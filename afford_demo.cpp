#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

#include <string>
#include <inference_engine.hpp>

#include "seg_model.h"

void demo_img()
{
    auto seg_m = afford_seg("components_20220427.xml");
    cv::Mat img = cv::imread("../img/2.jpg");

    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    cv::Mat res;
    res = seg_m.run_img(img);

    cv::imshow("res", res);

    cv::waitKey(0);
}

void demo_webcam()
{
    auto seg_m = afford_seg("components_20220427.xml");
    cv::Mat img;
    cv::VideoCapture cap(0);
    while(1)
    {
        cap >> img;
        cv::resize(img, img, cv::Size(224,224),0,0,cv::INTER_LINEAR);
        // cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

        auto res = seg_m.run_img(img);
        cv::imshow("img", img);
        cv::imshow("res", res);
        cv::waitKey(1);
    }

}


void test_openvino(const char* model_path)
{
    std::string input_name_ = "data";

    InferenceEngine::Core ie;
    InferenceEngine::CNNNetwork model = ie.ReadNetwork(model_path);

    InferenceEngine::InputsDataMap inputs_map(model.getInputsInfo());
    input_name_ = inputs_map.begin()->first;
    InferenceEngine::InputInfo::Ptr input_info = inputs_map.begin()->second;
}



int main()
{
    demo_img();
    // demo_webcam();
    return 0;
}