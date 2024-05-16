//
// Created by souffle on 24-4-28.
//

#ifndef YOLOV8_DEPLOY_MODULES_DETECT_COMMON_H_
#define YOLOV8_DEPLOY_MODULES_DETECT_COMMON_H_
#include <opencv2/opencv.hpp>

namespace detect {

typedef struct InferenceResult {
  int id{};
  float confidence{};
  cv::Rect box{};
  cv::RotatedRect rotatedBox{};
  cv::Mat boxMask{};
} InferenceResult_t;
using InferenceResults = std::vector<InferenceResult_t>;

template<typename T>
T VectorProduct(const std::vector<T> &v) {
  return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
};

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
    putText(img, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 1, color[result[i].id], 2);
  }
  cv::addWeighted(img, 0.5, mask, 0.5, 0, img); //add mask to src
  cv::imwrite(savePath, img);
}
}
#endif //YOLOV8_DEPLOY_MODULES_DETECT_COMMON_H_
