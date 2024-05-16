//
// Created by souffle on 24-4-29.
//

#ifndef YOLOV8_DEPLOY_MODULES_POSE_ONNX_POSE_H_
#define YOLOV8_DEPLOY_MODULES_POSE_ONNX_POSE_H_
#include <iostream>
#include<memory>
#include <opencv2/opencv.hpp>
#include<onnxruntime_cxx_api.h>
#include <fstream>
#include "pose/common.h"
#include "common.h"
#include "flag_header.h"

namespace pose::onnx {

class Pose {
 public:
  explicit Pose() : ortMemoryInfo_(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator,
                                                              OrtMemType::OrtMemTypeCPUOutput)) {};
  ~Pose() {
    delete ortSession_;
  };

  bool LoadModel(const std::string &modelPath,
                 const std::string &labelPath,
                 bool isCuda = false,
                 int cudaID = 0,
                 bool warmUp = true) {
    try {
      if (!CheckPath(modelPath) && !CheckPath(labelPath)) {
        return false;
      }
      if (!load_cls(labelPath)) {
        return false;
      }
      std::vector<std::string> available_providers = Ort::GetAvailableProviders();
      auto cuda_available = std::find(available_providers.begin(), available_providers.end(), "CUDAExecutionProvider");
      if (isCuda && (cuda_available == available_providers.end())) {
        std::cout << "Your ORT build without GPU. Change to CPU." << std::endl;
        std::cout << "************* Infer model on CPU! *************" << std::endl;
      } else if (isCuda && (cuda_available != available_providers.end())) {
#if USE_CUDA
        std::cout << "************* Infer model on GPU! *************" << std::endl;
#if ORT_API_VERSION < ORT_OLD_VISON
        OrtCUDAProviderOptions cudaOption;
            cudaOption.device_id = cudaID;
            ortSessionOptions_.AppendExecutionProvider_CUDA(cudaOption);
#else
            OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(ortSessionOptions_, cudaID);
#endif
#endif
      } else {
        std::cout << "************* Infer model on CPU! *************" << std::endl;
      }
      //

      ortSessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

#ifdef _WIN32
      std::wstring model_path(modelPath.begin(), modelPath.end());
        _OrtSession = new Ort::Session(_OrtEnv, model_path.c_str(), ortSessionOptions_);
#else
      ortSession_ = new Ort::Session(ortEnv_, modelPath.c_str(), ortSessionOptions_);
#endif

      Ort::AllocatorWithDefaultOptions allocator;
      //init input
      inputNodesNum_ = ortSession_->GetInputCount();
#if ORT_API_VERSION < ORT_OLD_VISON
      _inputName = _OrtSession->GetInputName(0, allocator);
        _inputNodeNames.push_back(_inputName);
#else
      inputName_ = std::move(ortSession_->GetInputNameAllocated(0, allocator));
      inputNodeNames_.push_back(inputName_.get());
#endif
      std::cout << inputNodeNames_[0] << std::endl;
      Ort::TypeInfo inputTypeInfo = ortSession_->GetInputTypeInfo(0);
      auto input_tensor_info = inputTypeInfo.GetTensorTypeAndShapeInfo();
      inputNodeDataType_ = input_tensor_info.GetElementType();
      inputTensorShape_ = input_tensor_info.GetShape();

      if (inputTensorShape_[0] == -1) {
        isDynamicShape_ = true;
        inputTensorShape_[0] = batchSize_;

      }
      if (inputTensorShape_[2] == -1 || inputTensorShape_[3] == -1) {
        isDynamicShape_ = true;
        inputTensorShape_[2] = netHeight_;
        inputTensorShape_[3] = netWidth_;
      }
      //init output
      outputNodesNum_ = ortSession_->GetOutputCount();
#if ORT_API_VERSION < ORT_OLD_VISON
      _output_name0 = _OrtSession->GetOutputName(0, allocator);
        _outputNodeNames.push_back(_output_name0);
#else
      output_name0_ = std::move(ortSession_->GetOutputNameAllocated(0, allocator));
      outputNodeNames_.push_back(output_name0_.get());
#endif
      Ort::TypeInfo type_info_output0(nullptr);
      type_info_output0 = ortSession_->GetOutputTypeInfo(0);  //output0

      auto tensor_info_output0 = type_info_output0.GetTensorTypeAndShapeInfo();
      outputNodeDataType_ = tensor_info_output0.GetElementType();
      outputTensorShape_ = tensor_info_output0.GetShape();

      //_outputMaskNodeDataType = tensor_info_output1.GetElementType(); //the same as output0
      //_outputMaskTensorShape = tensor_info_output1.GetShape();
      //if (_outputTensorShape[0] == -1)
      //{
      //	_outputTensorShape[0] = _batchSize;
      //	_outputMaskTensorShape[0] = _batchSize;
      //}
      //if (_outputMaskTensorShape[2] == -1) {
      //	//size_t ouput_rows = 0;
      //	//for (int i = 0; i < _strideSize; ++i) {
      //	//	ouput_rows += 3 * (_netWidth / _netStride[i]) * _netHeight / _netStride[i];
      //	//}
      //	//_outputTensorShape[1] = ouput_rows;

      //	_outputMaskTensorShape[2] = _segHeight;
      //	_outputMaskTensorShape[3] = _segWidth;
      //}

      //warm up
      if (isCuda && warmUp) {
        //draw run
        std::cout << "Start warming up" << std::endl;
        size_t input_tensor_length = pose::VectorProduct(inputTensorShape_);
        auto *temp = new float[input_tensor_length];
        std::vector<Ort::Value> input_tensors;
        std::vector<Ort::Value> output_tensors;
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            ortMemoryInfo_, temp, input_tensor_length, inputTensorShape_.data(),
            inputTensorShape_.size()));
        for (int i = 0; i < 3; ++i) {
          output_tensors = ortSession_->Run(Ort::RunOptions{nullptr},
                                            inputNodeNames_.data(),
                                            input_tensors.data(),
                                            inputNodeNames_.size(),
                                            outputNodeNames_.data(),
                                            outputNodeNames_.size());
        }

        delete[]temp;
      }
    }
    catch (const std::exception &) {
      return false;
    }
    return true;
  };

  bool BatchDetect(std::vector<cv::Mat> &srcImages,
                   std::vector<pose::InferenceResults> &outputs,
                   bool countTime = false) {
    auto ts = cv::getTickCount();
    std::vector<cv::Vec4d> params;
    std::vector<cv::Mat> input_images;
    cv::Size input_size(netWidth_, netHeight_);
    //preprocessing
    PreProcessing(srcImages, input_images, params);
    cv::Mat blob = cv::dnn::blobFromImages(input_images, 1 / 255.0, input_size, cv::Scalar(0, 0, 0), true, false);

    int64_t input_tensor_length = pose::VectorProduct(inputTensorShape_);
    std::vector<Ort::Value> input_tensors;
    std::vector<Ort::Value> output_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<float>(ortMemoryInfo_,
                                                            (float *) blob.data,
                                                            input_tensor_length,
                                                            inputTensorShape_.data(),
                                                            inputTensorShape_.size()));

    output_tensors = ortSession_->Run(Ort::RunOptions{nullptr},
                                      inputNodeNames_.data(),
                                      input_tensors.data(),
                                      inputNodeNames_.size(),
                                      outputNodeNames_.data(),
                                      outputNodeNames_.size()
    );
    //post-process
    float *all_data = output_tensors[0].GetTensorMutableData<float>();
    outputTensorShape_ = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    int net_width = (int) outputTensorShape_[1];

    int key_point_length = net_width - 5;
    int key_point_num = 17; //_bodyKeyPoints.size(), shape[x, y, confidence]
    if (key_point_num * 3 != key_point_length) {
      std::cout << "Pose should be shape [x, y, confidence] with 17-points" << std::endl;
      return false;
    }
    int64_t one_output_length = pose::VectorProduct(outputTensorShape_) / outputTensorShape_[0];
    for (int img_index = 0; img_index < srcImages.size(); ++img_index) {
      cv::Mat output0 = cv::Mat(cv::Size((int) outputTensorShape_[2], (int) outputTensorShape_[1]),
                                CV_32F,
                                all_data).t();  //[bs,116,8400]=>[bs,8400,116]
      all_data += one_output_length;
      float *pdata = (float *) output0.data;
      int rows = output0.rows;
      std::vector<int> class_ids;//结果id数组
      std::vector<float> confidences;//结果每个id对应置信度数组
      std::vector<cv::Rect> boxes;//每个id矩形框
      std::vector<std::vector<pose::PoseKeyPoint>> pose_key_points; //保存kpt

      for (int r = 0; r < rows; ++r) {
        float max_class_socre = pdata[4];
        if (max_class_socre >= classThreshold_) {
          //rect [x,y,w,h]
          float x = (pdata[0] - params[img_index][2]) / params[img_index][0];  //x
          float y = (pdata[1] - params[img_index][3]) / params[img_index][1];  //y
          float w = pdata[2] / params[img_index][0];  //w
          float h = pdata[3] / params[img_index][1];  //h
          int left = MAX(int(x -0.5 * w + 0.5), 0);
          int top = MAX(int(y -0.5 * h + 0.5), 0);
          class_ids.push_back(0);
          confidences.push_back(max_class_socre);
          boxes.push_back(cv::Rect(left, top, int(w + 0.5), int(h + 0.5)));
          std::vector<pose::PoseKeyPoint> temp_kpts;
          for (int kpt = 0; kpt < key_point_length; kpt += 3) {
            pose::PoseKeyPoint temp_kp;
            temp_kp.x = (pdata[5 + kpt] - params[img_index][2]) / params[img_index][0];
            temp_kp.y = (pdata[6 + kpt] - params[img_index][3]) / params[img_index][1];
            temp_kp.confidence = pdata[7 + kpt];
            temp_kpts.push_back(temp_kp);
          }
          pose_key_points.push_back(temp_kpts);
        }
        pdata += net_width;//下一行
      }

      std::vector<int> nms_result;
      cv::dnn::NMSBoxes(boxes, confidences, classThreshold_, nmsThreshold_, nms_result);
      std::vector<std::vector<float>> temp_mask_proposals;
      cv::Rect holeImgRect(0, 0, srcImages[img_index].cols, srcImages[img_index].rows);
      pose::InferenceResults temp_output;
      for (int i = 0; i < nms_result.size(); ++i) {
        int idx = nms_result[i];
        pose::InferenceResult result;
        result.id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx] & holeImgRect;
        result.keyPoints = pose_key_points[idx];
        temp_output.emplace_back(result);
      }
      outputs.emplace_back(temp_output);
    }
    if (countTime) {
      printf("inter time: %.2f ms\n", ((cv::getTickCount() - ts) / cv::getTickFrequency()) * 1000);
    }
    if (!outputs.empty())
      return true;
    else
      return false;

  }

  std::vector<std::string> GetLabels() {
    return cls_;
  }

 private:
  bool load_cls(const std::string &clsPath) {
    std::ifstream file(clsPath);
    if (file.is_open()) {
      std::string line;
      while (std::getline(file, line)) {
        // 去除特殊字符
        line.erase(std::remove_if(line.begin(), line.end(), [](char c) {
          return !std::isalnum(c) && !std::isspace(c); // 保留字母、数字和空格
        }), line.end());
        cls_.push_back(line);
      }
      file.close();
      return true;
    } else {
      std::cerr << "Unable to open file." << std::endl;
      return false;
    }
  };
  static int PreProcessing(const std::vector<cv::Mat> &srcImages,
                           std::vector<cv::Mat> &outImages,
                           std::vector<cv::Vec4d> &params) {
    outImages.clear();
    cv::Size input_size = cv::Size(netWidth_, netHeight_);
    for (const auto &temp_img : srcImages) {
      cv::Vec4d temp_param = {1, 1, 0, 0};
      if (temp_img.size() != input_size) {
        cv::Mat borderImg;
        LetterBox(temp_img, borderImg, temp_param, input_size, false, false, true, 32);
        outImages.push_back(borderImg);
        params.push_back(temp_param);
      } else {
        outImages.push_back(temp_img);
        params.push_back(temp_param);
      }
    }

    int lack_num = batchSize_ - srcImages.size();
    if (lack_num > 0) {
      for (int i = 0; i < lack_num; ++i) {
        cv::Mat temp_img = cv::Mat::zeros(input_size, CV_8UC3);
        cv::Vec4d temp_param = {1, 1, 0, 0};
        outImages.push_back(temp_img);
        params.push_back(temp_param);
      }
    }
    return 0;

  }

 private:
  inline static const int netWidth_ = 640;   //ONNX-net-input-width
  inline static const int netHeight_ = 640;  //ONNX-net-input-height

  inline static int batchSize_ = 1;  //if multi-batch,set this
  inline static bool isDynamicShape_ = false;//onnx support dynamic shape
  inline static float classThreshold_ = 0.25;
  inline static float nmsThreshold_ = 0.45;
  inline static float maskThreshold_ = 0.5;

  //ONNXRUNTIME
  Ort::Env ortEnv_ = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "Yolov8-Pose");
  Ort::SessionOptions ortSessionOptions_ = Ort::SessionOptions();
  Ort::Session *ortSession_ = nullptr;
  Ort::MemoryInfo ortMemoryInfo_;
#if ORT_API_VERSION < ORT_OLD_VISON
  char* inputName_, * output_name0_;
#else
  std::shared_ptr<char> inputName_, output_name0_;
#endif

  std::vector<char *> inputNodeNames_;   //输入节点名
  std::vector<char *> outputNodeNames_;  //输出节点名

  size_t inputNodesNum_ = 0;        //输入节点数
  size_t outputNodesNum_ = 0;       //输出节点数

  ONNXTensorElementDataType inputNodeDataType_; //数据类型
  ONNXTensorElementDataType outputNodeDataType_;
  std::vector<int64_t> inputTensorShape_; //输入张量shape
  std::vector<int64_t> outputTensorShape_;

  std::vector<std::string> cls_;

};
}
#endif //YOLOV8_DEPLOY_MODULES_POSE_ONNX_POSE_H_
