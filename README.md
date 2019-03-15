# jetnet-wrapper
Wrapper for jetnet.

## Build

```bash
mkdir build && cd build
cmake ..
make
```

## Usage

```bash
Jetnet YOLO runner
Usage: jetnet-wrapper [params] type modelfile nameslist cameraid

	-?, -h, --help, --usage (value:true)
		print this message
	--anchors
		Anchor prior file name
	--batch (value:1)
		Batch size
	--nmsthresh, --nt (value:0.45)
		Non-maxima suppression threshold
	--profile
		Enable profiling
	-t, --thresh (value:0.24)
		Detection threshold

	type (value:<none>)
		Network type (yolov2, yolov3)
	modelfile (value:<none>)
		Built and serialized TensorRT model file
	nameslist (value:<none>)
		Class names list file
	cameraid (value:<none>)
		Index of camera
```

For example

```bash
./jetnet-wrapper \
  yolov3-tiny \
  yolov3-tiny-fp16.model \
  coco.names \
  1 \
  --profile
```

## Benchmark

|                           | Inference Time | Pre-processing Time | Post-processing Time |
| ------------------------- | -------------- | ------------------- | -------------------- |
| YOLOv3-Tiny w/ Jetson TX2 | 12.86ms        | 0.92ms              | 1.94ms               |
