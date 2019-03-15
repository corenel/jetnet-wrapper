#include <opencv2/opencv.hpp>
#include "jetnet.h"
#include "create_runner.h"
#include <iostream>
#include <ctime>
#include <iomanip>
#include <sstream>

using namespace jetnet;

/// \brief Process single frame and show detection result
/// \param frame frame from camera
/// \param runner YOLO model runner
/// \param pre YOLO model pre-processor
/// \param post YOLO model post-processor
/// \param class_names list of class names
/// \param batch_size batch size
/// \return -1 for error, 0 for normal running, 1 for exitting
int process_single_frame(
    cv::Mat &frame,
    YoloRunnerFactory::RunnerType &runner,
    YoloRunnerFactory::PreType &pre,
    YoloRunnerFactory::PostType &post,
    std::vector<std::string> &class_names,
    int batch_size = 1) {

  // create list of images to feed
  std::vector<cv::Mat> images;

  // process the same image multiple times if batch size > 1
  for (int i = 0; i < batch_size; ++i) {
    images.push_back(frame);
  }


  // register images to the preprocessor
  pre->register_images(images);

  // run pre/infer/post pipeline for a number of times depending on the profiling setting
  if (!(*runner)()) {
    std::cerr << "Failed to run network" << std::endl;
    return -1;
  }

  // get detections and visualise
  auto detections = post->get_detections();

  // image is read in RGB, convert to BGR for display with imshow and bbox rendering
  cv::Mat out;
  cv::cvtColor(images[0], out, cv::COLOR_RGB2BGR);
  draw_detections(detections[0], class_names, out);

  // show detection result
  cv::imshow("result", out);
  const int key = cv::waitKey(1);
  if (key == 'q'/*113*/) {
    // press 'q' for quitting
    return 1;
  } else if (key == 's') {
    // press 's' for saving current frame
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::ostringstream oss;
    oss << "frame_" << std::put_time(&tm, "%d%m%Y%H%M%S") << ".png";
    cv::imwrite(oss.str(), frame);
    std::cout << "Frame saved to " << oss.str() << std::endl;
  }

  return 0;

}

int main(int argc, char **argv) {
  std::string keys =
      "{help h usage ? |      | print this message                        }"
      "{@type          |<none>| Network type (yolov2, yolov3)             }"
      "{@modelfile     |<none>| Built and serialized TensorRT model file  }"
      "{@nameslist     |<none>| Class names list file                     }"
      "{@cameraid      |<none>| Index of camera                           }"
      "{profile        |      | Enable profiling                          }"
      "{t thresh       | 0.24 | Detection threshold                       }"
      "{nt nmsthresh   | 0.45 | Non-maxima suppression threshold          }"
      "{batch          | 1    | Batch size                                }"
      "{anchors        |      | Anchor prior file name                    }";

  // parse arguments
  cv::CommandLineParser parser(argc, argv, keys);
  parser.about("Jetnet YOLO runner");

  // show usage
  if (parser.has("help")) {
    parser.printMessage();
    return -1;
  }

  // init variables by arguments
  auto network_type = parser.get<std::string>("@type");
  auto input_model_file = parser.get<std::string>("@modelfile");
  auto input_names_file = parser.get<std::string>("@nameslist");
  auto input_camera_id = parser.get<int>("@cameraid");
  auto enable_profiling = parser.has("profile");
  auto threshold = parser.get<float>("thresh");
  auto nms_threshold = parser.get<float>("nmsthresh");
  auto batch_size = parser.get<int>("batch");
  auto anchors_file = parser.get<std::string>("anchors");

  // check parsing errors
  if (!parser.check()) {
    parser.printErrors();
    return 0;
  }

  // read class names
  std::vector<std::string> class_names;
  if (!read_text_file(class_names, input_names_file)) {
    std::cerr << "Failed to read names file" << std::endl;
    return -1;
  }

  // read anchors file
  std::vector<std::string> anchor_priors_str;
  std::vector<float> anchor_priors;
  if (!anchors_file.empty()) {
    if (!read_text_file(anchor_priors_str, anchors_file)) {
      std::cerr << "Failed to read anchor priors file" << std::endl;
      return -1;
    }
    for (auto str : anchor_priors_str)
      anchor_priors.push_back(std::stof(str));
  }

  // construct YOLO model
  YoloRunnerFactory runner_fact(class_names.size(), threshold, nms_threshold, batch_size,
                                anchor_priors, enable_profiling);
  YoloRunnerFactory::PreType pre;
  YoloRunnerFactory::RunnerType runner;
  YoloRunnerFactory::PostType post;

  std::tie(pre, runner, post) = runner_fact.create(network_type);

  if (!pre || !runner || !post) {
    std::cerr << "Failed to create runner" << std::endl;
    return -1;
  }

  if (!runner->init(input_model_file)) {
    std::cerr << "Failed to init runner" << std::endl;
    return -1;
  }

  // open camera
  cv::VideoCapture cap(input_camera_id);
  if (!cap.isOpened()) {
    return -1;
  }

  // print info about camera
  const auto camera_width = (size_t) cap.get(cv::CAP_PROP_FRAME_WIDTH);
  const auto camera_height = (size_t) cap.get(cv::CAP_PROP_FRAME_HEIGHT);
  const auto camera_fps = cap.get(cv::CAP_PROP_FPS);
  std::cout << "Resolution: " << camera_width << " x " << camera_height << std::endl;
  std::cout << "Camera FPS: " << camera_fps << std::endl;

  // start detection
  cv::Mat frame;
  while (cap.read(frame)) {
    auto exit_code = process_single_frame(frame, runner, pre, post, class_names);
    if (exit_code == 1) {
      break;
    }
  }
  cv::destroyAllWindows();

  // show profiling if enabled
  runner->print_profiling();

  std::cout << "Success!" << std::endl;
  return 0;
}
