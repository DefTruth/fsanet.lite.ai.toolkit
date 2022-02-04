# fsanet.lite.ai.toolkit
使用 [🍅🍅 Lite.AI.ToolKit](https://github.com/DefTruth/lite.ai.toolkit) C++工具箱来跑FSANet头部姿态估计的一些案例, 包含ONNXRuntime C++、MNN和TNN版本。FSANet的权重文件大小只有 **1Mb** ，是一个非常轻量级的头部姿态估计模型。

<div align='center'>
  <img src='resources/1_onnx.jpg' height="224px" width="224px">
  <img src='resources/2_onnx.jpg' height="224px" width="224px">
  <img src='resources/3_onnx.jpg' height="224px" width="224px">
</div>  


Star⭐️🌟什么的，就随缘吧~ 🙃🤪🍀

## 2. C++版本源码

FSANet C++ 版本的源码包含ONNXRuntime、MNN和TNN三个版本，源码可以在 [lite.ai.toolkit](https://github.com/DefTruth/lite.ai.toolkit) 工具箱中找到。本项目主要介绍如何基于 [lite.ai.toolkit](https://github.com/DefTruth/lite.ai.toolkit) 工具箱，直接使用FSANet来跑人脸检测。需要说明的是，本项目是基于MacOS下编译的 [liblite.ai.toolkit.v0.1.0.dylib](https://github.com/DefTruth/yolox.lite.ai.toolkit/blob/main/lite.ai.toolkit/lib) 来实现的，对于使用MacOS的用户，可以直接下载本项目包含的*liblite.ai.toolkit.v0.1.0*动态库和其他依赖库进行使用。而非MacOS用户，则需要从[lite.ai.toolkit](https://github.com/DefTruth/lite.ai.toolkit) 中下载源码进行编译。[lite.ai.toolkit](https://github.com/DefTruth/lite.ai.toolkit) c++工具箱目前包含80+流行的开源模型，就不多介绍了，只是平时顺手捏的，整合了自己学习过程中接触到的一些模型，感兴趣的同学可以去看看。
* [fsanet.cpp](https://github.com/DefTruth/lite.ai.toolkit/blob/main/lite/ort/cv/fsanet.cpp)
* [fsanet.h](https://github.com/DefTruth/lite.ai.toolkit/blob/main/lite/ort/cv/fsanet.h)
* [mnn_fsanet.cpp](https://github.com/DefTruth/lite.ai.toolkit/blob/main/lite/mnn/cv/mnn_fsanet.cpp)
* [mnn_fsanet.h](https://github.com/DefTruth/lite.ai.toolkit/blob/main/lite/mnn/cv/mnn_fsanet.h)
* [tnn_fsanet.cpp](https://github.com/DefTruth/lite.ai.toolkit/blob/main/lite/tnn/cv/tnn_fsanet.cpp)
* [tnn_fsanet.h](https://github.com/DefTruth/lite.ai.toolkit/blob/main/lite/tnn/cv/tnn_fsanet.h)

ONNXRuntime C++、MNN和TNN版本的推理实现均已测试通过，欢迎白嫖~  

## 3. 模型文件

### 3.1 ONNX模型文件
可以从我提供的链接下载 ([Baidu Drive](https://pan.baidu.com/s/1elUGcx7CZkkjEoYhTMwTRQ) code: 8gin) , 也可以从本直接仓库下载。


|                 Class                 |      Pretrained ONNX Files      |              Rename or Converted From (Repo)              | Size  |
| :-----------------------------------: | :-----------------------------: | :-------------------------------------------------------: | :---: |  
| *lite::cv::face::pose::FSANet* |    fsanet-var.onnx    | [...fsanet...](https://github.com/omasaht/headpose-fsanet-pytorch) | 1.2Mb |
| *lite::cv::face::pose::FSANet* |    fsanet-1x1.onnx    | [...fsanet...](https://github.com/omasaht/headpose-fsanet-pytorch) | 1.2Mb |


### 3.2 MNN模型文件
MNN模型文件下载地址，([Baidu Drive](https://pan.baidu.com/s/1KyO-bCYUv6qPq2M8BH_Okg) code: 9v63), 也可以从本直接仓库下载。

|                 Class                 |      Pretrained MNN Files      |              Rename or Converted From (Repo)              | Size  |
| :-----------------------------------: | :-----------------------------: | :-------------------------------------------------------: | :---: |
| *lite::mnn::cv::face::pose::FSANet* |    fsanet-var.mnn    | [...fsanet...](https://github.com/omasaht/headpose-fsanet-pytorch) | 1.2Mb |
| *lite::mnn::cv::face::pose::FSANet* |    fsanet-1x1.mnn   | [...fsanet...](https://github.com/omasaht/headpose-fsanet-pytorch) | 1.2Mb |

### 3.3 TNN模型文件
TNN模型文件下载地址，([Baidu Drive](https://pan.baidu.com/s/1lvM2YKyUbEc5HKVtqITpcw) code: 6o6k), 也可以从本直接仓库下载。

|                 Class                 |      Pretrained TNN Files      |              Rename or Converted From (Repo)              | Size  |
| :-----------------------------------: | :-----------------------------: | :-------------------------------------------------------: | :---: |
| *lite::tnn::cv::face::pose::FSANet* |    fsanet-var.opt.tnnproto&tnnmodel    | [...fsanet...](https://github.com/omasaht/headpose-fsanet-pytorch) | 1.2Mb |
| *lite::tnn::cv::face::pose::FSANet* |    fsanet-1x1.opt.tnnproto&tnnmodel    | [...fsanet...](https://github.com/omasaht/headpose-fsanet-pytorch) | 1.2Mb |


## 4. 接口文档

在[lite.ai.toolkit](https://github.com/DefTruth/lite.ai.toolkit) 中，FSANet的实现类为：

```c++
class LITE_EXPORTS lite::cv::face::detect::FSANet;
class LITE_EXPORTS lite::mnn::cv::face::detect::FSANet;
class LITE_EXPORTS lite::tnn::cv::face::detect::FSANet;
```  

该类型目前包含1公共接口`detect`用于进行头部姿态检测。
```c++
public:
  void detect(const cv::Mat &mat, types::EulerAngles &euler_angles);
```
`detect`接口的输入参数说明：
* mat: cv::Mat类型，BGR格式，一张包含人脸头部的图片（不包含过多的背景）。
* euler_angles: types::EulerAngles类型，包含被检测到的欧拉角(yaw,pitch,roll)，值域为[-90,+90]; 

## 5. 使用案例
这里测试使用的是fsanet-var和fsanet-1x1的模型取均值，你可以只使用其中一个模型。

### 5.1 ONNXRuntime版本
```c++
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
```  

### 5.2 MNN版本
```c++
#include "lite/lite.h"

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
```  

### 5.3 TNN版本
```c++
#include "lite/lite.h"

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
```  


* 输出结果为:

<div align='center'>
  <img src='resources/1_onnx.jpg' height="224px" width="224px">
  <img src='resources/2_onnx.jpg' height="224px" width="224px">
  <img src='resources/3_onnx.jpg' height="224px" width="224px">
</div> 

## 6. 编译运行
在MacOS下可以直接编译运行本项目，无需下载其他依赖库。其他系统则需要从[lite.ai.toolkit](https://github.com/DefTruth/lite.ai.toolkit) 中下载源码先编译*lite.ai.toolkit.v0.1.0*动态库。
```shell
git clone --depth=1 https://github.com/DefTruth/fsanet.lite.ai.toolkit.git
cd fsanet.lite.ai.toolkit 
sh ./build.sh
```  

* CMakeLists.txt设置

```cmake
cmake_minimum_required(VERSION 3.17)
project(fsanet.lite.ai.toolkit)

set(CMAKE_CXX_STANDARD 11)

# setting up lite.ai.toolkit
set(LITE_AI_DIR ${CMAKE_SOURCE_DIR}/lite.ai.toolkit)
set(LITE_AI_INCLUDE_DIR ${LITE_AI_DIR}/include)
set(LITE_AI_LIBRARY_DIR ${LITE_AI_DIR}/lib)
include_directories(${LITE_AI_INCLUDE_DIR})
link_directories(${LITE_AI_LIBRARY_DIR})

set(OpenCV_LIBS
        opencv_highgui
        opencv_core
        opencv_imgcodecs
        opencv_imgproc
        opencv_video
        opencv_videoio
        )
# add your executable
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/examples/build)

add_executable(lite_fsanet examples/test_lite_fsanet.cpp)
target_link_libraries(lite_fsanet
        lite.ai.toolkit
        onnxruntime
        MNN  # need, if built lite.ai.toolkit with ENABLE_MNN=ON,  default OFF
        ncnn # need, if built lite.ai.toolkit with ENABLE_NCNN=ON, default OFF
        TNN  # need, if built lite.ai.toolkit with ENABLE_TNN=ON,  default OFF
        ${OpenCV_LIBS})  # link lite.ai.toolkit & other libs.
```

* building && testing information:
```shell
[ 50%] Building CXX object CMakeFiles/lite_fsanet.dir/examples/test_lite_fsanet.cpp.o
[100%] Linking CXX executable lite_fsanet
[100%] Built target lite_fsanet
Testing Start ...
LITEORT_DEBUG LogId: ../hub/onnx/cv/fsanet-var.onnx
=============== Input-Dims ==============
input_node_dims: 1
input_node_dims: 3
input_node_dims: 64
input_node_dims: 64
=============== Output-Dims ==============
Output: 0 Name: output Dim: 0 :1
Output: 0 Name: output Dim: 1 :3
========================================
LITEORT_DEBUG LogId: ../hub/onnx/cv/fsanet-1x1.onnx
=============== Input-Dims ==============
input_node_dims: 1
input_node_dims: 3
input_node_dims: 64
input_node_dims: 64
=============== Output-Dims ==============
Output: 0 Name: output Dim: 0 :1
Output: 0 Name: output Dim: 1 :3
========================================
Default Version yaw: -1.68474 pitch: -5.54399 roll: -0.131204
LITEORT_DEBUG LogId: ../hub/onnx/cv/fsanet-var.onnx
=============== Input-Dims =============
...
Testing Successful !
```  
