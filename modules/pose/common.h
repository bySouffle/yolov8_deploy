//
// Created by souffle on 24-4-29.
//

#ifndef YOLOV8_DEPLOY_MODULES_POSE_COMMON_H_
#define YOLOV8_DEPLOY_MODULES_POSE_COMMON_H_
#include <opencv2/opencv.hpp>
#include <numeric>
namespace pose {

struct PoseParams {
  float kptThreshold = 0.5;
  int kptRadius = 5;
  bool isDrawKptLine = true; //If True, the function will draw lines connecting keypoint for human pose.Default is True.
  cv::Scalar personColor = cv::Scalar(0, 0, 255);
  std::vector<std::vector<int>> skeleton = {
      {16, 14}, {14, 12}, {17, 15}, {15, 13},
      {12, 13}, {6, 12}, {7, 13}, {6, 7}, {6, 8}, {7, 9},
      {8, 10}, {9, 11}, {2, 3}, {1, 2}, {1, 3}, {2, 4},
      {3, 5}, {4, 6}, {5, 7}
  };
  std::vector<cv::Scalar> posePalette =
      {
          cv::Scalar(255, 128, 0),
          cv::Scalar(255, 153, 51),
          cv::Scalar(255, 178, 102),
          cv::Scalar(230, 230, 0),
          cv::Scalar(255, 153, 255),
          cv::Scalar(153, 204, 255),
          cv::Scalar(255, 102, 255),
          cv::Scalar(255, 51, 255),
          cv::Scalar(102, 178, 255),
          cv::Scalar(51, 153, 255),
          cv::Scalar(255, 153, 153),
          cv::Scalar(255, 102, 102),
          cv::Scalar(255, 51, 51),
          cv::Scalar(153, 255, 153),
          cv::Scalar(102, 255, 102),
          cv::Scalar(51, 255, 51),
          cv::Scalar(0, 255, 0),
          cv::Scalar(0, 0, 255),
          cv::Scalar(255, 0, 0),
          cv::Scalar(255, 255, 255),
      };
  std::vector<int> limbColor = {9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16};
  std::vector<int> kptColor = {16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9};
  std::map<unsigned int, std::string> kptBodyNames{
      {0, "Nose"},
      {1, "left_eye"}, {2, "right_eye"},
      {3, "left_ear"}, {4, "right_ear"},
      {5, "left_shoulder"}, {6, "right_shoulder"},
      {7, "left_elbow"}, {8, "right_elbow"},
      {9, "left_wrist"}, {10, "right_wrist"},
      {11, "left_hip"}, {12, "right_hip"},
      {13, "left_knee"}, {14, "right_knee"},
      {15, "left_ankle"}, {16, "right_ankle"}
  };
};

struct PoseKeyPoint {
  float x = 0;
  float y = 0;
  float confidence = 0;
};

typedef struct InferenceResult {
  int id{};
  float confidence{};
  cv::Rect box{};
  cv::RotatedRect rotatedBox;
  cv::Mat boxMask;
  std::vector<PoseKeyPoint> keyPoints; //pose key points

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
                            const PoseParams &poseParams,
                            const std::string &savePath,
                            bool isVideo = false) {
  for (int i = 0; i < result.size(); i++) {
    int left, top;
    int color_num = i;
    if (result[i].box.area() > 0) {
      rectangle(img, result[i].box, poseParams.personColor, 2, 8);
      left = result[i].box.x;
      top = result[i].box.y;
    } else
      continue;

    std::string label = "person:" + std::to_string(result[i].confidence);
    int baseLine;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = MAX(top, labelSize.height);
    //rectangle(frame, cv::Point(left, top - int(1.5 * labelSize.height)), cv::Point(left + int(1.5 * labelSize.width), top + baseLine), cv::Scalar(0, 255, 0), FILLED);
    putText(img, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 1, poseParams.personColor, 2);
    if (result[i].keyPoints.size() != poseParams.kptBodyNames.size())
      continue;
    for (int j = 0; j < result[i].keyPoints.size(); ++j) {
      PoseKeyPoint kpt = result[i].keyPoints[j];
      if (kpt.confidence < poseParams.kptThreshold)
        continue;
      cv::Scalar kptColor = poseParams.posePalette[poseParams.kptColor[j]];
      cv::circle(img, cv::Point(kpt.x, kpt.y), poseParams.kptRadius, kptColor, -1, 8);
    }
    if (poseParams.isDrawKptLine) {
      for (int j = 0; j < poseParams.skeleton.size(); ++j) {
        PoseKeyPoint kpt0 = result[i].keyPoints[poseParams.skeleton[j][0] - 1];
        PoseKeyPoint kpt1 = result[i].keyPoints[poseParams.skeleton[j][1] - 1];
        if (kpt0.confidence < poseParams.kptThreshold || kpt1.confidence < poseParams.kptThreshold)
          continue;
        cv::Scalar kptColor = poseParams.posePalette[poseParams.limbColor[j]];
        cv::line(img, cv::Point(kpt0.x, kpt0.y), cv::Point(kpt1.x, kpt1.y), kptColor, 2, 8);
      }
    }
  }
  cv::imwrite(savePath, img);

}

}

#endif //YOLOV8_DEPLOY_MODULES_POSE_COMMON_H_
