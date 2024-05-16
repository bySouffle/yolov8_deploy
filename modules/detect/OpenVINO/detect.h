//
// Created by souffle on 24-4-28.
//

#ifndef YOLOV8_DEPLOY_MODULES_DETECT_OPENVINO_DETECT_H_
#define YOLOV8_DEPLOY_MODULES_DETECT_OPENVINO_DETECT_H_

#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
#include "detect/common.h"
#include "common.h"

namespace detect::openvino {

class Detect {
 public:
  Detect() = default;
  ~Detect() = default;

  bool LoadModel(const std::string &modelPath,
                 const std::string &labelPath,
                 bool isGPU = false) {

    if (!CheckPath(modelPath) && !CheckPath(labelPath)) {
      return false;
    }
    if (!load_cls(labelPath)) {
      return false;
    }
    // Step 1. Initialize OpenVINO Runtime core
    std::vector<std::string> availableDevices = core.get_available_devices();
    for (auto &availableDevice : availableDevices) {
      printf("supported device name : %s \n", availableDevice.c_str());
    }
    // Step 2. Read a model
    std::shared_ptr<ov::Model> model;
    try {
      model = core.read_model(modelPath);
      printInputAndOutputsInfo(*model); // 打印模型信息

    } catch (ov::Exception &e) {
      fprintf(stderr, "read model error: %s\n", e.what());
      return false;
    }
    inputShape_ = ov::shape_size(model->input().get_shape());
    // Step 4. Inizialize Preprocessing for the model
    ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);
    //  Specify input image format
    ppp.input().tensor()
        .set_element_type(ov::element::u8)
        .set_layout("NHWC")
        .set_color_format(ov::preprocess::ColorFormat::BGR);
    // Specify preprocess pipeline to input image without resizing
    ppp.input().preprocess()
        .convert_element_type(ov::element::f32)
        .convert_color(ov::preprocess::ColorFormat::RGB).scale({255., 255., 255.});
    //  Specify model's input layout
    ppp.input().model().set_layout("NCHW");
    //  Specify output results format
    ppp.output().tensor().set_element_type(ov::element::f32);

    //  Embed above steps in the graph
    model = ppp.build();
    try {

      if (isGPU) {
        compiled_model = core.compile_model(model, "GPU");
        printf("use GPU\n");
      } else {
        compiled_model = core.compile_model(model, "CPU");
        printf("use CPU\n");
      }
    } catch (ov::Exception &e) {
      fprintf(stderr, "compile_model error: %s\n", e.what());
      return false;
    }
    infer_request = compiled_model.create_infer_request();
    return true;
  }

  bool OnceDetect(cv::Mat &srcImage, InferenceResults &output, bool countTime = false) {
    std::vector<cv::Mat> input_data = {srcImage};
    std::vector<InferenceResults> temp_output;
    if (BatchDetect(input_data, temp_output, countTime)) {
      output = temp_output[0];
      return true;
    } else return false;
  }

  bool BatchDetect(std::vector<cv::Mat> &srcImages, std::vector<InferenceResults> &outputs, bool countTime = false) {
    std::vector<cv::Vec4d> params;
    std::vector<cv::Mat> input_images;
    Preprocessing(srcImages, input_images, params);

    for (int i = 0; i < input_images.size(); ++i) {
      auto ts = cv::getTickCount();
      auto *input_data = (float *) input_images[i].data;
      auto input_tensor =
          ov::Tensor(compiled_model.input().get_element_type(), compiled_model.input().get_shape(), input_data);
      infer_request.set_input_tensor(input_tensor);
      infer_request.infer();

      const ov::Tensor &output_tensor = infer_request.get_output_tensor();
      ov::Shape output_shape = output_tensor.get_shape();
      auto *detections = output_tensor.data<float>();

      std::vector<int> class_ids;
      std::vector<float> confidences;
      std::vector<cv::Rect> boxes;
      int rows = static_cast<int>(output_shape[1]);
      int cols = static_cast<int>(output_shape[2]);
      const cv::Mat det_output(rows, cols, CV_32F, (float *) detections);

      for (int j = 0; j < det_output.cols; ++j) {
        const cv::Mat classes_scores = det_output.col(j).rowRange(4, rows);
        cv::Point class_id_point;
        double score;
        cv::minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);
        if (score > classThreshold_) {
          //rect [x,y,w,h]
          const float x = (det_output.at<float>(0, j) - params[i][2]) / params[i][0];
          const float y = (det_output.at<float>(1, j) - params[i][3]) / params[i][1];
          const float w = det_output.at<float>(2, j) / params[i][0];
          const float h = det_output.at<float>(3, j) / params[i][1];

          int left = MAX(int(x -0.5 * w + 0.5), 0);
          int top = MAX(int(y -0.5 * h + 0.5), 0);
          class_ids.push_back(class_id_point.x);
          confidences.push_back(score);
          boxes.push_back(cv::Rect(left, top, int(w + 0.5), int(h + 0.5)));
        }
      }
      std::vector<int> nms_result;
      cv::dnn::NMSBoxes(boxes, confidences, classThreshold_, nmsThreshold_, nms_result);
      std::vector<std::vector<float>> temp_mask_proposals;
      cv::Rect holeImgRect(0, 0, srcImages[i].cols, srcImages[i].rows);
      std::vector<InferenceResult> temp_output;
      for (int i = 0; i < nms_result.size(); ++i) {
        int idx = nms_result[i];
        InferenceResult result;
        result.id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx] & holeImgRect;
        temp_output.push_back(result);
      }
      outputs.push_back(temp_output);

      if (countTime) {
        printf("inter time: %.2f ms\n", ((cv::getTickCount() - ts) / cv::getTickFrequency()) * 1000);
      }

    }
    if (!outputs.empty())
      return true;
    else
      return false;
  }

  std::vector<std::string> GetLabels() {
    return cls_;
  }

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

  static int Preprocessing(const std::vector<cv::Mat> &srcImgs,
                           std::vector<cv::Mat> &outSrcImgs,
                           std::vector<cv::Vec4d> &params) {
    outSrcImgs.clear();
    cv::Size input_size = cv::Size(netWidth_, netHeight_);
    for (const auto &temp_img : srcImgs) {
      cv::Vec4d temp_param = {1, 1, 0, 0};
      if (temp_img.size() != input_size) {

        cv::Mat borderImg;
        LetterBox(temp_img, borderImg, temp_param, input_size, false, false, true, 32);
        //std::cout << borderImg.size() << std::endl;
        outSrcImgs.push_back(borderImg);
        params.push_back(temp_param);
      } else {
        outSrcImgs.push_back(temp_img);
        params.push_back(temp_param);
      }
    }

    int lack_num = batchSize_ - srcImgs.size();
    if (lack_num > 0) {
      for (int i = 0; i < lack_num; ++i) {
        cv::Mat temp_img = cv::Mat::zeros(input_size, CV_8UC3);
        cv::Vec4d temp_param = {1, 1, 0, 0};
        outSrcImgs.push_back(temp_img);
        params.push_back(temp_param);
      }
    }
    return 0;

  }

 private:
  inline static const int netWidth_ = 640;   //ONNX-net-input-width
  inline static const int netHeight_ = 640;  //ONNX-net-input-height

  inline static int batchSize_ = 1;  //if multi-batch,set this
  inline static int inputShape_ = netWidth_ * netHeight_;  //if multi-batch,set this

  inline static bool isDynamicShape_ = false;//onnx support dynamic shape
  inline static float classThreshold_ = 0.5;
  inline static float nmsThreshold_ = 0.45;
  inline static float maskThreshold_ = 0.5;

  std::vector<std::string> cls_;

 private:
  //  >>>>>>>>>>>>>>>>>>>>>>>>>>> openvino ov  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  ov::Core core;
  ov::CompiledModel compiled_model;
  ov::InferRequest infer_request;

  //  <<<<<<<<<<<<<<<<<<<<<<<<<<< openvino ov  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

};
}
#endif //YOLOV8_DEPLOY_MODULES_DETECT_OPENVINO_DETECT_H_
