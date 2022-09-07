#include <string>
#include <opencv2/core.hpp>
#include <inference_engine.hpp>


class afford_seg
{
public:
    afford_seg(const char* param);
    ~afford_seg();

    InferenceEngine::ExecutableNetwork network_;
    InferenceEngine::InferRequest infer_request_;

    void run(cv::Mat);

    cv::Mat run_img(cv::Mat);

private:
    void preprocess(cv::Mat& image, InferenceEngine::Blob::Ptr& blob);
    std::string input_name_ = "data";
    std::string output_name_ = "output";
};