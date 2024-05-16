#   Yolov8 Onnx OpenVINO

## 1. Model conversion
```shell    
# OpenVINO 2022.1.0  WARN Currently ONNX operator set version: 17 is unsupported. Falling back to: 15
yolo export model=best.pt format=onnx --opset=15
mo --input_model best.onnx
```

## 2. Download ONNX lib
```shell
# Get onnx lib 
##  CPU
wget -i https://github.com/microsoft/onnxruntime/releases/download/v1.17.3/onnxruntime-linux-x64-1.17.3.tgz
##  GPU
wget -i https://github.com/microsoft/onnxruntime/releases/download/v1.17.3/onnxruntime-linux-x64-gpu-1.17.3.tgz
```

##  3. Build
```shell
# CPU 
# Modified modules/flag_header.h +8  #define USE_CUDA 0
wget -i https://github.com/microsoft/onnxruntime/releases/download/v1.17.3/onnxruntime-linux-x64-1.17.3.tgz
tar -xvf onnxruntime-linux-x64-1.17.3.tgz
mkdir build && cd build
cmake  ..
make -j4


# GPU
# Modified modules/flag_header.h +8  #define USE_CUDA 1
wget -i https://github.com/microsoft/onnxruntime/releases/download/v1.17.3/onnxruntime-linux-x64-gpu-1.17.3.tgz
tar -xvf onnxruntime-linux-x64-gpu-1.17.3.tgz
mkdir build && cd build
cmake -DUSE_CUDA=ON ..
make -j4
```
