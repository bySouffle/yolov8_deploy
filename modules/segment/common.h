//
// Created by souffle on 24-4-28.
//

#ifndef YOLOV8_DEPLOY_MODULES_SEGMENT_COMMON_H_
#define YOLOV8_DEPLOY_MODULES_SEGMENT_COMMON_H_
#include <opencv2/opencv.hpp>
#include <numeric>

namespace segment {

typedef struct InferenceResult {
  int id{};
  float confidence{};
  cv::Rect box{};
  cv::RotatedRect rotatedBox;
  cv::Mat boxMask;
} InferenceResult_t;
using InferenceResults = std::vector<InferenceResult_t>;

template<typename T>
T VectorProduct(const std::vector<T> &v) {
  return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
};

struct MaskParams {
  //int segChannels = 32;
  //int segWidth = 160;
  //int segHeight = 160;
  int netWidth = 640;
  int netHeight = 640;
  float maskThreshold = 0.5;
  cv::Size srcImgShape;
  cv::Vec4d params;
};

inline static void GetMask(const cv::Mat &maskProposals,
                           const cv::Mat &maskProtos,
                           InferenceResults &output,
                           const MaskParams &maskParams) {
  //std::cout << maskProtos.size << std::endl;

  int net_width = maskParams.netWidth;
  int net_height = maskParams.netHeight;
  int seg_channels = maskProtos.size[1];
  int seg_height = maskProtos.size[2];
  int seg_width = maskProtos.size[3];
  float mask_threshold = maskParams.maskThreshold;
  cv::Vec4f params = maskParams.params;
  cv::Size src_img_shape = maskParams.srcImgShape;

  cv::Mat protos = maskProtos.reshape(0, {seg_channels, seg_width * seg_height});

  cv::Mat matmul_res = (maskProposals * protos).t();
  cv::Mat masks = matmul_res.reshape(output.size(), {seg_width, seg_height});
  std::vector<cv::Mat> maskChannels;
  split(masks, maskChannels);
  for (int i = 0; i < output.size(); ++i) {
    cv::Mat dest, mask;
    //sigmoid
    cv::exp(-maskChannels[i], dest);
    dest = 1.0 / (1.0 + dest);

    cv::Rect roi(int(params[2] / net_width * seg_width),
                 int(params[3] / net_height * seg_height),
                 int(seg_width - params[2] / 2),
                 int(seg_height - params[3] / 2));
    dest = dest(roi);
    resize(dest, mask, src_img_shape, cv::INTER_NEAREST);

    //crop
    cv::Rect temp_rect = output[i].box;
    mask = mask(temp_rect) > mask_threshold;
    output[i].boxMask = mask;
  }
}
inline static void GetMask2(const cv::Mat &maskProposals,
                            const cv::Mat &maskProtos,
                            InferenceResult &output,
                            const MaskParams &maskParams) {
  int net_width = maskParams.netWidth;
  int net_height = maskParams.netHeight;
  int seg_channels = maskProtos.size[1];
  int seg_height = maskProtos.size[2];
  int seg_width = maskProtos.size[3];
  float mask_threshold = maskParams.maskThreshold;
  cv::Vec4f params = maskParams.params;
  cv::Size src_img_shape = maskParams.srcImgShape;

  cv::Rect temp_rect = output.box;
  //crop from mask_protos
  int rang_x = floor((temp_rect.x * params[0] + params[2]) / net_width * seg_width);
  int rang_y = floor((temp_rect.y * params[1] + params[3]) / net_height * seg_height);
  int rang_w = ceil(((temp_rect.x + temp_rect.width) * params[0] + params[2]) / net_width * seg_width) - rang_x;
  int rang_h = ceil(((temp_rect.y + temp_rect.height) * params[1] + params[3]) / net_height * seg_height) - rang_y;

  rang_w = MAX(rang_w, 1);
  rang_h = MAX(rang_h, 1);
  if (rang_x + rang_w > seg_width) {
    if (seg_width - rang_x > 0)
      rang_w = seg_width - rang_x;
    else
      rang_x -= 1;
  }
  if (rang_y + rang_h > seg_height) {
    if (seg_height - rang_y > 0)
      rang_h = seg_height - rang_y;
    else
      rang_y -= 1;
  }

  std::vector<cv::Range> roi_rangs;
  roi_rangs.push_back(cv::Range(0, 1));
  roi_rangs.push_back(cv::Range::all());
  roi_rangs.push_back(cv::Range(rang_y, rang_h + rang_y));
  roi_rangs.push_back(cv::Range(rang_x, rang_w + rang_x));

  //crop
  cv::Mat temp_mask_protos = maskProtos(roi_rangs).clone();
  cv::Mat protos = temp_mask_protos.reshape(0, {seg_channels, rang_w * rang_h});
  cv::Mat matmul_res = (maskProposals * protos).t();
  cv::Mat masks_feature = matmul_res.reshape(1, {rang_h, rang_w});
  cv::Mat dest, mask;

  //sigmoid
  cv::exp(-masks_feature, dest);
  dest = 1.0 / (1.0 + dest);

  int left = floor((net_width / seg_width * rang_x - params[2]) / params[0]);
  int top = floor((net_height / seg_height * rang_y - params[3]) / params[1]);
  int width = ceil(net_width / seg_width * rang_w / params[0]);
  int height = ceil(net_height / seg_height * rang_h / params[1]);

  resize(dest, mask, cv::Size(width, height), cv::INTER_NEAREST);
  cv::Rect mask_rect = temp_rect - cv::Point(left, top);
  mask_rect &= cv::Rect(0, 0, width, height);
  mask = mask(mask_rect) > mask_threshold;
  if (mask.rows != temp_rect.height
      || mask.cols != temp_rect.width) { //https://github.com/UNeedCryDear/yolov8-opencv-onnxruntime-cpp/pull/30
    resize(mask, mask, temp_rect.size(), cv::INTER_NEAREST);
  }
  output.boxMask = mask;

}

inline static void DrawRotatedBox(cv::Mat &srcImg, cv::RotatedRect box, cv::Scalar color, int thinkness) {
  cv::Point2f p[4];
  box.points(p);
  for (int l = 0; l < 4; ++l) {
    line(srcImg, p[l], p[(l + 1) % 4], color, thinkness, 8);
  }
}

inline static void DrawPred(cv::Mat &img,
                            InferenceResults result,
                            std::vector<std::string> classNames,
                            std::vector<cv::Scalar> color,
                            const std::string &savePath,
                            bool isVideo = false) {
  cv::Mat mask = img.clone();
  for (int i = 0; i < result.size(); i++) {
    int left = 0, top = 0;

    int color_num = i;
    if (result[i].box.area() > 0) {
      rectangle(img, result[i].box, color[result[i].id], 2, 8);
      left = result[i].box.x;
      top = result[i].box.y;
    }
    if (result[i].rotatedBox.size.width * result[i].rotatedBox.size.height > 0) {
      DrawRotatedBox(img, result[i].rotatedBox, color[result[i].id], 2);
      left = result[i].rotatedBox.center.x;
      top = result[i].rotatedBox.center.y;
    }
    if (result[i].boxMask.rows && result[i].boxMask.cols > 0)
      mask(result[i].box).setTo(color[result[i].id], result[i].boxMask);
    std::string label = classNames[result[i].id] + ":" + std::to_string(result[i].confidence);
    int baseLine;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = MAX(top, labelSize.height);
    //rectangle(frame, cv::Point(left, top - int(1.5 * labelSize.height)), cv::Point(left + int(1.5 * labelSize.width), top + baseLine), cv::Scalar(0, 255, 0), FILLED);
//    putText(img, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 1, color[result[i].id], 2);
  }
  cv::addWeighted(img, 0.5, mask, 0.5, 0, img); //add mask to src
  cv::imwrite(savePath, img);
}
}

#endif //YOLOV8_DEPLOY_MODULES_SEGMENT_COMMON_H_
