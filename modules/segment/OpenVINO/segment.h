//
// Created by souffle on 24-4-28.
//

#ifndef YOLOV8_DEPLOY_MODULES_SEGMENT_OPENVINO_SEGMENT_H_
#define YOLOV8_DEPLOY_MODULES_SEGMENT_OPENVINO_SEGMENT_H_

#include <vector>
#include <string>
#include<openvino/openvino.hpp>
#include <fstream>
#include <numeric>
#include "segment/common.h"
#include "common.h"
#include "flag_header.h"
namespace segment::openvino {

class Segment {
 public:
  explicit Segment() = default;
  ~Segment() = default;

  bool LoadModel(const std::string &modelPath,
                 const std::string &labelPath,
                 bool isGPU = false,
                 bool warmUp = true) {
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
    ppp.output(0).tensor().set_element_type(ov::element::f32);
    ppp.output(1).tensor().set_element_type(ov::element::f32);

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
  };

  std::vector<std::string> GetLabels() {
    return cls_;
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
    PreProcessing(srcImages, input_images, params);

    for (int i = 0; i < input_images.size(); ++i) {
      auto ts = cv::getTickCount();
      auto *input_data = (float *) input_images[i].data;
      auto input_tensor =
          ov::Tensor(compiled_model.input().get_element_type(), compiled_model.input().get_shape(), input_data);
      infer_request.set_input_tensor(input_tensor);
      infer_request.infer();

      /// 处理推理计算结果
      // 获得推理结果, 输出结点是[116,8400], 一共8400个结果, 每个结果116个维度.
      // 116=4+80+32, 4是预测框的[cx, cy, w, h], 80是每个类别的置信度, 32是分割需要用到的
      const ov::Tensor output0 = infer_request.get_tensor("output0");
      const auto *output0_buffer = output0.data<const float>();
      const ov::Shape output0_shape = output0.get_shape();
      const int output0_rows = static_cast<int >(output0_shape[1]);
      const int output0_cols = static_cast<int >(output0_shape[2]);

      const ov::Tensor output1 = infer_request.get_tensor("output1");
      const ov::Shape output1_shape = output1.get_shape();

      int score_cls_length = output0_rows - 4 - (int) output1_shape[1];

      std::vector<int> mask_proto_shape =
          {(int) output1_shape[0], (int) output1_shape[1], (int) output1_shape[2], (int) output1_shape[3]};

      // Detect Matrix: 116 x 8400 -> 8400 x 116
      // 一共8400个结果, 每个结果116个维度.
      // 116=4+80+32, 4是预测框的[cx, cy, w, h]; 80是每个类别的置信度; 32需要与Proto Matrix相乘得到分割mask, 所以这里转置了矩阵
      const cv::Mat detect_buffer = cv::Mat(output0_rows, output0_cols, CV_32F, (float *) output0_buffer).t();
      // Proto Matrix: 1x32x160x160 -> 32x25600
      const cv::Mat proto_buffer(1, (int) (output1_shape[2] * output1_shape[3]), CV_32F, output1.data());

      const float conf_threshold = 0.5;
      const float nms_threshold = 0.5;
      std::vector<cv::Rect> mask_boxes;
      std::vector<cv::Rect> boxes;
      std::vector<int> class_ids;
      std::vector<float> confidences;
      std::vector<std::vector<float>> picked_proposals;  //output0[:,:, 5 + _className.size():net_width]===> for mask
      for (int j = 0; j < detect_buffer.rows; ++j) {
        const cv::Mat result = detect_buffer.row(j);
        /// 处理检测部分的结果
        // 取置信度最大的那个标签  [4 xywh][80 cls][32 mask]
        const cv::Mat classes_scores = result.colRange(4, (int) (output0_rows - output1_shape[1]));
        cv::Point class_id_point;
        double score;
        cv::minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

        if (score > classThreshold_) {
          std::vector<float> temp_proto = result.colRange(4 + score_cls_length, output0_rows);
          picked_proposals.push_back(temp_proto);
          //rect [x,y,w,h]
          const float x = (result.at<float>(0, 0) - params[i][2]) / params[i][0];
          const float y = (result.at<float>(0, 1) - params[i][3]) / params[i][1];
          const float w = result.at<float>(0, 2) / params[i][0];
          const float h = result.at<float>(0, 3) / params[i][1];

          int left = MAX(int(x -0.5 * w + 0.5), 0);
          int top = MAX(int(y -0.5 * h + 0.5), 0);
          class_ids.push_back(class_id_point.x);
          confidences.push_back(score);
          boxes.emplace_back(left, top, int(w + 0.5), int(h + 0.5));
        }
      }

      std::vector<int> nms_result;
      cv::dnn::NMSBoxes(boxes, confidences, classThreshold_, nmsThreshold_, nms_result);
      std::vector<std::vector<float>> temp_mask_proposals;
      cv::Rect holeImgRect(0, 0, srcImages[i].cols, srcImages[i].rows);
      std::vector<InferenceResult> temp_output;
      for (auto j = 0; j < nms_result.size(); j++) {
        int idx = nms_result[j];
        InferenceResult result;
        result.id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx] & holeImgRect;
        temp_mask_proposals.push_back(picked_proposals[idx]);
        temp_output.push_back(result);
      }

      MaskParams mask_params;
      mask_params.params = params[i];
      mask_params.srcImgShape = srcImages[i].size();
      mask_params.netHeight = netHeight_;
      mask_params.netWidth = netWidth_;
      mask_params.maskThreshold = maskThreshold_;
      cv::Mat mask_proto = cv::Mat(mask_proto_shape,
                                   CV_32F,
                                   output1.data<float>() + i * output1.get_size());
      for (int j = 0; j < temp_mask_proposals.size(); ++j) {
        GetMask2(cv::Mat(temp_mask_proposals[j]).t(), mask_proto, temp_output[j], mask_params);
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

#endif //YOLOV8_DEPLOY_MODULES_SEGMENT_OPENVINO_SEGMENT_H_
