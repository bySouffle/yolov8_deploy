#include <iostream>
#include "modules/detect/OpenVINO/detect.h"
#include "modules/detect/onnx/detect.h"
#include "modules/segment/OpenVINO/segment.h"
#include "modules/segment/onnx/segment.h"
#include "modules/pose/OpenVINO/pose.h"
#include "modules/pose/onnx/pose.h"
int main() {
  //  model
  std::string detect_model = "../models/yolov8n.onnx";
  std::string pose_model = "../models/yolov8n-pose.onnx";
  std::string seg_model = "../models/yolov8n-seg.onnx";

  //  label
  std::string labels = "../models/labels.txt";

  //  image
  std::string detect_img = "../images/bus.jpg";
  std::string save_img = "../images/bus_detect.jpg";


  cv::Mat src = cv::imread(detect_img);
  cv::Mat img = src.clone();
  std::vector<cv::Mat> images{src};
  //  detect
  std::vector<detect::InferenceResults> result;
  ////    onnx
//  detect::onnx::Detect detect_onnx;
//
//  detect_onnx.LoadModel(detect_model,
//                        labels,
//                        true);
//
//  detect_onnx.BatchDetect(images, result, true);
//  detect::DrawPred(src, result[0], detect_onnx.GetLabels(), RandColor(), save_img);
  ////    ov
//  detect::openvino::Detect detect;
//
//  detect.LoadModel(detect_model,
//                   labels,
//                   false);
//
//
//  detect.BatchDetect(images, result, true);
//  detect::DrawPred(src, result[0], detect.GetLabels(), RandColor(), save_img);

  //  seg
  std::vector<segment::InferenceResults> result_seg;
  ////    onnx
  segment::onnx::Segment segment;
  segment.LoadModel(
      seg_model,
      labels,
      true);
  segment.BatchDetect(images,result_seg, true);
  segment::DrawPred(src, result_seg[0],segment.GetLabels(),RandColor(),save_img);
  ////    ov
//  segment::openvino::Segment segment;
//  segment.LoadModel(
//      seg_model,
//      labels,
//      false);
//  segment.BatchDetect(images,result_seg, true);
//  segment::DrawPred(src, result_seg[0],segment.GetLabels(),RandColor(),save_img);

  //  pose
//  std::vector<pose::InferenceResults> result_pose;
//  ////    onnx
//    pose::onnx::Pose pose;
//    pose.LoadModel(
//          pose_model,
//          labels,
//          false);
//    pose.BatchDetect(images,result_pose, true);
//    pose::DrawPred(src, result_pose[0], pose::PoseParams(),save_img);
  ////    ov
//  pose::openvino::Pose pose;
//  pose.LoadModel(
//      pose_model,
//      labels,
//      false);
//  pose.BatchDetect(images,result_pose, true);
//  pose::DrawPred(src, result_pose[0], pose::PoseParams(),save_img);

  return 0;
}
