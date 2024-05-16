//
// Created by souffle on 24-4-26.
//

#ifndef YOLOV8_DEPLOY_MODEL_COMMON_H_
#define YOLOV8_DEPLOY_MODEL_COMMON_H_

#include <filesystem>
#include<opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <random>

#define ORT_OLD_VISON 13  //ort1.12.0 之前的版本为旧版本API

inline static bool CheckPath(const std::string &path) {
  std::filesystem::path model(path);
  if (!std::filesystem::exists(model)) {
    fprintf(stderr, "Model path does not exist, please check: %s\n", path.data());
    return false;
  } else
    return true;
}

inline static void printInputAndOutputsInfo(const ov::Model &network) {
  std::cout << "model name: " << network.get_friendly_name() << std::endl;

  const std::vector<ov::Output<const ov::Node>> inputs = network.inputs();
  for (const ov::Output<const ov::Node> &input : inputs) {
    std::cout << "    inputs" << std::endl;

    const std::string name = input.get_names().empty() ? "NONE" : input.get_any_name();
    std::cout << "        input name: " << name << std::endl;

    const ov::element::Type type = input.get_element_type();
    std::cout << "        input type: " << type << std::endl;

    const ov::Shape &shape = input.get_shape();
    std::cout << "        input shape: " << shape << std::endl;
  }

  const std::vector<ov::Output<const ov::Node>> outputs = network.outputs();
  for (const ov::Output<const ov::Node> &output : outputs) {
    std::cout << "    outputs" << std::endl;

    const std::string name = output.get_names().empty() ? "NONE" : output.get_any_name();
    std::cout << "        output name: " << name << std::endl;

    const ov::element::Type type = output.get_element_type();
    std::cout << "        output type: " << type << std::endl;

    const ov::Shape &shape = output.get_shape();
    std::cout << "        output shape: " << shape << std::endl;
  }
}

inline static void LetterBox(const cv::Mat &image,
                             cv::Mat &outImage,
                             cv::Vec4d &params,
                             const cv::Size &newShape,
                             bool autoShape,
                             bool scaleFill,
                             bool scaleUp,
                             int stride,
                             const cv::Scalar &color = cv::Scalar(114, 114, 114)) {

  cv::Size shape = image.size();
  float r = std::min((float) newShape.height / (float) shape.height,
                     (float) newShape.width / (float) shape.width);
  if (!scaleUp)
    r = std::min(r, 1.0f);

  float ratio[2]{r, r};
  int new_un_pad[2] = {(int) std::round((float) shape.width * r), (int) std::round((float) shape.height * r)};

  auto dw = (float) (newShape.width - new_un_pad[0]);
  auto dh = (float) (newShape.height - new_un_pad[1]);

  if (autoShape) {
    dw = (float) ((int) dw % stride);
    dh = (float) ((int) dh % stride);
  } else if (scaleFill) {
    dw = 0.0f;
    dh = 0.0f;
    new_un_pad[0] = newShape.width;
    new_un_pad[1] = newShape.height;
    ratio[0] = (float) newShape.width / (float) shape.width;
    ratio[1] = (float) newShape.height / (float) shape.height;
  }

  dw /= 2.0f;
  dh /= 2.0f;

  if (shape.width != new_un_pad[0] && shape.height != new_un_pad[1]) {
    cv::resize(image, outImage, cv::Size(new_un_pad[0], new_un_pad[1]));
  } else {
    outImage = image.clone();
  }

  int top = int(std::round(dh - 0.1f));
  int bottom = int(std::round(dh + 0.1f));
  int left = int(std::round(dw - 0.1f));
  int right = int(std::round(dw + 0.1f));
  params[0] = ratio[0];
  params[1] = ratio[1];
  params[2] = left;
  params[3] = top;
  cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}

inline static std::vector<cv::Scalar> RandColor() {
  std::vector<cv::Scalar> color;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 255);
  for (int i = 0; i < 80; i++) {
    int b = dis(gen);
    int g = dis(gen);
    int r = dis(gen);
    color.emplace_back(b, g, r);
  };
  return color;
}

#endif //YOLOV8_DEPLOY_MODEL_COMMON_H_
