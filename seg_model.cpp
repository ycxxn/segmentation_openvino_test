#include "seg_model.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <algorithm>

using namespace InferenceEngine;

afford_seg::afford_seg(const char* model_path)
{
    InferenceEngine::Core ie;
    InferenceEngine::CNNNetwork model = ie.ReadNetwork(model_path);
    // prepare input settings
    InferenceEngine::InputsDataMap inputs_map(model.getInputsInfo());
    input_name_ = inputs_map.begin()->first;
    InferenceEngine::InputInfo::Ptr input_info = inputs_map.begin()->second;
    //input_info->setPrecision(InferenceEngine::Precision::FP32);
    //input_info->setLayout(InferenceEngine::Layout::NCHW);



    //prepare output settings
    InferenceEngine::OutputsDataMap outputs_map(model.getOutputsInfo());
    for (auto &output_info : outputs_map)
    {
        //std::cout<< "Output:" << output_info.first <<std::endl;
        output_info.second->setPrecision(InferenceEngine::Precision::FP32);
    }

    //get network
    network_ = ie.LoadNetwork(model, "CPU");
    infer_request_ = network_.CreateInferRequest();
}


afford_seg::~afford_seg()
{
    ;
}

void afford_seg::preprocess(cv::Mat& image, InferenceEngine::Blob::Ptr& blob)
{
    int img_w = image.cols;
    int img_h = image.rows;
    int channels = 3;

    // size_t num_channels = blob->getTensorDesc().getDims()[1];
    // size_t h = blob->getTensorDesc().getDims()[2];
    // size_t w = blob->getTensorDesc().getDims()[3];
    // size_t image_size = h*w;
    // // cv::Mat blob_image;
    // cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    // image.convertTo(image, CV_32F);
    // // image = image / 255.0;
    // cv::subtract(image, cv::Scalar(0.485, 0.456, 0.406),image);
    // cv::divide(image, cv::Scalar(0.229, 0.224, 0.225), image);


    InferenceEngine::MemoryBlob::Ptr mblob = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob);
    if (!mblob)
    {
        THROW_IE_EXCEPTION << "We expect blob to be inherited from MemoryBlob in matU8ToBlob, "
            << "but by fact we were not able to cast inputBlob to MemoryBlob";
    }
    // locked memory holder should be alive all time while access to its buffer happens
    auto mblobHolder = mblob->wmap();

    float *blob_data = mblobHolder.as<float *>();


    for (size_t c = 0; c < channels; c++)
    {
        for (size_t  h = 0; h < img_h; h++)
        {
            for (size_t w = 0; w < img_w; w++)
            {
                blob_data[c * img_w * img_h + h * img_w + w] =
                    (float)image.at<cv::Vec3b>(h, w)[c];
            }
        }
    }
}

void afford_seg::run(cv::Mat img)
{
    InferenceEngine::Blob::Ptr input_blob = infer_request_.GetBlob(input_name_);
    preprocess(img, input_blob);

    cv::Mat mask(224, 224, CV_8UC1, cv::Scalar::all(0));
    // do inference
    infer_request_.Infer();

    // const InferenceEngine::Blob::Ptr pred_blob = infer_request_.GetBlob(output_name_);
    auto output = infer_request_.GetBlob(output_name_);

    // pred_blob->buffer();
    const float* probs = static_cast<PrecisionTrait<Precision::FP32>::value_type*>(output->buffer());
    const InferenceEngine::SizeVector outputDims = output->getTensorDesc().getDims();

    std::cout << outputDims[0] << "x" << outputDims[1] << "x";
    std::cout << outputDims[2] << "x" << outputDims[3] << std::endl;

    // for(int i=0;i<224;i++)
    // {
    //     std::cout << probs[i] << "\t" ;
    //     if(i%6==5)
    //     {
    //         std::cout << "\n" ;
    //     }
    // }
    std::cout << probs[0]<< "\t";
    std::cout << probs[224*224] << "\t";
    std::cout << probs[224*224*2] << "\t";
    std::cout << probs[224*224*3] << "\t";
    std::cout << probs[224*224*4] << "\t";
    std::cout << probs[224*224*5] << std::endl;

    std::vector<float> v = {0,0,0,0,0,0};

    for(int i=0; i<224; i++)
    {
        for(int j=0; j<224; j++)
        {
            // probs[i*224+j]

            // std::cout << i*224+j << "\n";
            for(int c=0; c<6; c++)
            {
                v[c] = probs[c*224*224+i*224+j];
            }
            auto it = std::minmax_element(v.begin(), v.end());
            int max_idx = std::distance(v.begin(), it.second);

            std::cout << max_idx << " ";

            mask.at<uchar>(i,j) = max_idx*50;
        }
        std::cout << "\n";
    }

    // std::cout << probs[5*224*224+223*224+223];

    cv::imwrite("out.png", mask);
    // for(int c=0; c<6; c++)
    // {
    //     v[c] = probs[c*224*224];
    // }
    // auto it = std::minmax_element(v.begin(), v.end());
    // int max_idx = std::distance(v.begin(), it.second);
 
    // std::cout << max_idx << std::endl;

    // std::cout << probs[0][0] << std::endl;
    // std::cout << probs[0] << std::endl;
    // std::cout << probs[1] << std::endl;
    // std::cout << probs[2] << std::endl;
    // std::cout << probs[3] << std::endl;
    // std::cout << probs[4] << std::endl;
    // std::cout << probs[5] << std::endl;
    // std::cout << "-------"<< std::endl;
    // std::cout << probs[6] << std::endl;
    // std::cout << probs[7] << std::endl;
    // std::cout << probs[8] << std::endl;
    // std::cout << probs[9] << std::endl;
    // std::cout << probs[10] << std::endl;
    // std::cout << probs[11] << std::endl;
    // std::cout << "-------"<< std::endl;
    // std::cout << probs[12] << std::endl;
    // std::cout << probs[13] << std::endl;
    // std::cout << probs[14] << std::endl;
    // std::cout << probs[15] << std::endl;
    // std::cout << probs[16] << std::endl;
    // std::cout << probs[17] << std::endl;

    // std::cout << outputDims ; 

}


cv::Mat afford_seg::run_img(cv::Mat img)
{
    InferenceEngine::Blob::Ptr input_blob = infer_request_.GetBlob(input_name_);
    preprocess(img, input_blob);

    cv::Mat mask(224, 224, CV_8UC1, cv::Scalar::all(0));
    // do inference
    infer_request_.Infer();

    // const InferenceEngine::Blob::Ptr pred_blob = infer_request_.GetBlob(output_name_);
    auto output = infer_request_.GetBlob(output_name_);

    // pred_blob->buffer();
    const float* probs = static_cast<PrecisionTrait<Precision::FP32>::value_type*>(output->buffer());
    const InferenceEngine::SizeVector outputDims = output->getTensorDesc().getDims();

    std::cout << outputDims[0] << "x" << outputDims[1] << "x";
    std::cout << outputDims[2] << "x" << outputDims[3] << std::endl;

    std::vector<float> v = {0,0,0,0,0,0};

    for(int i=0; i<224; i++)
    {
        for(int j=0; j<224; j++)
        {
            // probs[i*224+j]

            // std::cout << i*224+j << "\n";
            for(int c=0; c<6; c++)
            {
                v[c] = probs[c*224*224+i*224+j];
            }
            auto it = std::minmax_element(v.begin(), v.end());
            int max_idx = std::distance(v.begin(), it.second);

            mask.at<uchar>(i,j) = max_idx*50;
        }
        
    }
    std::cout << probs[0] << std::endl;
    std::cout << probs[1] << std::endl;
    std::cout << probs[2] << std::endl;
    std::cout << probs[3] << std::endl;
    std::cout << probs[4] << std::endl;
    std::cout << probs[5] << std::endl;

    std::cout << probs[0] << "\t";
    std::cout << probs[1*224*224] << "\t";
    std::cout << probs[2*224*224] << "\t";
    std::cout << probs[3*224*224] << "\t";
    std::cout << probs[4*224*224] << "\t";
    std::cout << probs[5*224*224] << "\n";
    return mask;
}