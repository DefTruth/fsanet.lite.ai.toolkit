//
// Created by DefTruth on 2021/12/30.
//

#include "lite/lite.h"

static void test_default()
{
  std::string var_onnx_path = "../hub/onnx/cv/fsanet-var.onnx";
  std::string conv_onnx_path = "../hub/onnx/cv/fsanet-1x1.onnx";
  std::string test_img_path = "../resources/1.jpg";
  std::string save_img_path = "../logs/1.jpg";

  auto *var_fsanet = new lite::cv::face::pose::FSANet(var_onnx_path);
  auto *conv_fsanet = new lite::cv::face::pose::FSANet(conv_onnx_path);
  cv::Mat img_bgr = cv::imread(test_img_path);
  lite::types::EulerAngles var_euler_angles, conv_euler_angles;

  // 1. detect euler angles.
  var_fsanet->detect(img_bgr, var_euler_angles);
  conv_fsanet->detect(img_bgr, conv_euler_angles);

  lite::types::EulerAngles euler_angles;

  euler_angles.yaw = (var_euler_angles.yaw + conv_euler_angles.yaw) / 2.0f;
  euler_angles.pitch = (var_euler_angles.pitch + conv_euler_angles.pitch) / 2.0f;
  euler_angles.roll = (var_euler_angles.roll + conv_euler_angles.roll) / 2.0f;
  euler_angles.flag = var_euler_angles.flag && conv_euler_angles.flag;

  if (euler_angles.flag)
  {
    lite::utils::draw_axis_inplace(img_bgr, euler_angles);

    cv::imwrite(save_img_path, img_bgr);

    std::cout << "Default Version"
              << " yaw: " << euler_angles.yaw
              << " pitch: " << euler_angles.pitch
              << " roll: " << euler_angles.roll << std::endl;
  }

  delete var_fsanet;
  delete conv_fsanet;

}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
  std::string var_onnx_path = "../hub/onnx/cv/fsanet-var.onnx";
  std::string conv_onnx_path = "../hub/onnx/cv/fsanet-1x1.onnx";
  std::string test_img_path = "../resources/2.jpg";
  std::string save_img_path = "../logs/2_onnx.jpg";

  auto *var_fsanet = new lite::onnxruntime::cv::face::pose::FSANet(var_onnx_path);
  auto *conv_fsanet = new lite::onnxruntime::cv::face::pose::FSANet(conv_onnx_path);
  cv::Mat img_bgr = cv::imread(test_img_path);
  lite::types::EulerAngles var_euler_angles, conv_euler_angles;

  // 1. detect euler angles.
  var_fsanet->detect(img_bgr, var_euler_angles);
  conv_fsanet->detect(img_bgr, conv_euler_angles);

  lite::types::EulerAngles euler_angles;

  euler_angles.yaw = (var_euler_angles.yaw + conv_euler_angles.yaw) / 2.0f;
  euler_angles.pitch = (var_euler_angles.pitch + conv_euler_angles.pitch) / 2.0f;
  euler_angles.roll = (var_euler_angles.roll + conv_euler_angles.roll) / 2.0f;
  euler_angles.flag = var_euler_angles.flag && conv_euler_angles.flag;

  if (euler_angles.flag)
  {
    lite::utils::draw_axis_inplace(img_bgr, euler_angles);
    cv::imwrite(save_img_path, img_bgr);

    std::cout << "ONNXRuntime Version"
              << " yaw: " << euler_angles.yaw
              << " pitch: " << euler_angles.pitch
              << " roll: " << euler_angles.roll << std::endl;
  }

  delete var_fsanet;
  delete conv_fsanet;
#endif
}

static void test_mnn()
{
#ifdef ENABLE_MNN
  std::string var_mnn_path = "../hub/mnn/cv/fsanet-var.mnn";
  std::string conv_mnn_path = "../hub/mnn/cv/fsanet-1x1.mnn";
  std::string test_img_path = "../resources/2.jpg";
  std::string save_img_path = "../logs/2_mnn.jpg";

  auto *var_fsanet = new lite::mnn::cv::face::pose::FSANet(var_mnn_path);
  auto *conv_fsanet = new lite::mnn::cv::face::pose::FSANet(conv_mnn_path);
  cv::Mat img_bgr = cv::imread(test_img_path);
  lite::types::EulerAngles var_euler_angles, conv_euler_angles;

  // 1. detect euler angles.
  var_fsanet->detect(img_bgr, var_euler_angles);
  conv_fsanet->detect(img_bgr, conv_euler_angles);

  lite::types::EulerAngles euler_angles;

  euler_angles.yaw = (var_euler_angles.yaw + conv_euler_angles.yaw) / 2.0f;
  euler_angles.pitch = (var_euler_angles.pitch + conv_euler_angles.pitch) / 2.0f;
  euler_angles.roll = (var_euler_angles.roll + conv_euler_angles.roll) / 2.0f;
  euler_angles.flag = var_euler_angles.flag && conv_euler_angles.flag;

  if (euler_angles.flag)
  {
    lite::utils::draw_axis_inplace(img_bgr, euler_angles);
    cv::imwrite(save_img_path, img_bgr);

    std::cout << "MNN Version"
              << " yaw: " << euler_angles.yaw
              << " pitch: " << euler_angles.pitch
              << " roll: " << euler_angles.roll << std::endl;
  }

  delete var_fsanet;
  delete conv_fsanet;
#endif
}

static void test_ncnn()
{
#ifdef ENABLE_NCNN
#endif
}

static void test_tnn()
{
#ifdef ENABLE_TNN
  std::string var_proto_path = "../hub/tnn/cv/fsanet-var.opt.tnnproto";
  std::string var_model_path = "../hub/tnn/cv/fsanet-var.opt.tnnmodel";
  std::string conv_proto_path = "../hub/tnn/cv/fsanet-1x1.opt.tnnproto";
  std::string conv_model_path = "../hub/tnn/cv/fsanet-1x1.opt.tnnmodel";
  std::string test_img_path = "../resources/2.jpg";
  std::string save_img_path = "../logs/2_tnn.jpg";

  auto *var_fsanet = new lite::tnn::cv::face::pose::FSANet(var_proto_path, var_model_path);
  auto *conv_fsanet = new lite::tnn::cv::face::pose::FSANet(conv_proto_path, conv_model_path);
  cv::Mat img_bgr = cv::imread(test_img_path);
  lite::types::EulerAngles var_euler_angles, conv_euler_angles;

  // 1. detect euler angles.
  var_fsanet->detect(img_bgr, var_euler_angles);
  conv_fsanet->detect(img_bgr, conv_euler_angles);

  lite::types::EulerAngles euler_angles;

  euler_angles.yaw = (var_euler_angles.yaw + conv_euler_angles.yaw) / 2.0f;
  euler_angles.pitch = (var_euler_angles.pitch + conv_euler_angles.pitch) / 2.0f;
  euler_angles.roll = (var_euler_angles.roll + conv_euler_angles.roll) / 2.0f;
  euler_angles.flag = var_euler_angles.flag && conv_euler_angles.flag;

  if (euler_angles.flag)
  {
    lite::utils::draw_axis_inplace(img_bgr, euler_angles);
    cv::imwrite(save_img_path, img_bgr);

    std::cout << "TNN Version"
              << " yaw: " << euler_angles.yaw
              << " pitch: " << euler_angles.pitch
              << " roll: " << euler_angles.roll << std::endl;
  }

  delete var_fsanet;
  delete conv_fsanet;
#endif
}

static void test_lite()
{
  test_default();
  test_onnxruntime();
  test_mnn();
  test_ncnn();
  test_tnn();
}

int main(__unused int argc, __unused char *argv[])
{
  test_lite();
  return 0;
}
