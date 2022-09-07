auto scores = infer_request.GetBlob("scores");
auto boxes = infer_request.GetBlob("boxes");
auto clazzes = infer_request.GetBlob("classes");
auto raw_masks = infer_request.GetBlob("raw_masks");
const float* score_data = static_cast<PrecisionTrait<Precision::FP32>::value_type*>(scores->buffer());
const float* boxes_data = static_cast<PrecisionTrait<Precision::FP32>::value_type*>(boxes->buffer());
const float* clazzes_data = static_cast<PrecisionTrait<Precision::FP32>::value_type*>(clazzes->buffer());
const auto raw_masks_data = static_cast<PrecisionTrait<Precision::FP32>::value_type*>(raw_masks->buffer());
const SizeVector scores_outputDims = scores->getTensorDesc().getDims();
const SizeVector boxes_outputDims = boxes->getTensorDesc().getDims();
const SizeVector mask_outputDims = raw_masks->getTensorDesc().getDims();
const int max_count = scores_outputDims[0];
const int object_size = boxes_outputDims[1];
printf("mask NCHW=[%d, %d, %d, %d]
", mask_outputDims[0], mask_outputDims[1], mask_outputDims[2], mask_outputDims[3]);
int mask_h = mask_outputDims[2];
int mask_w = mask_outputDims[3];
size_t box_stride = mask_h * mask_w * mask_outputDims[1];
for (int n = 0; n < max_count; n++) {
    float confidence = score_data[n];
    float xmin = boxes_data[n*object_size] * w_rate;
    float ymin = boxes_data[n*object_size + 1] * h_rate;
    float xmax = boxes_data[n*object_size + 2] * w_rate;
    float ymax = boxes_data[n*object_size + 3] * h_rate;
    if (confidence > 0.5) {
        cv::Scalar color(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        cv::Rect box;
        float x1 = std::min(std::max(0.0f, xmin), static_cast<float>(im_w));
        float y1 = std::min(std::max(0.0f,ymin), static_cast<float>(im_h));
        float x2 = std::min(std::max(0.0f, xmax), static_cast<float>(im_w));
        float y2 = std::min(std::max(0.0f, ymax), static_cast<float>(im_h));
        box.x = static_cast<int>(x1);
        box.y = static_cast<int>(y1);
        box.width = static_cast<int>(x2 - x1);
        box.height = static_cast<int>(y2 - y1);
        int label = static_cast<int>(clazzes_data[n]);
        std::cout <<"confidence: "<< confidence<<" class name: "<< coco_labels[label] << std::endl;
        // 解析mask
        float* mask_arr = raw_masks_data + box_stride * n + mask_h * mask_w * label;
        cv::Mat mask_mat(mask_h, mask_w, CV_32FC1, mask_arr);
        cv::Mat roi_img = src(box);
        cv::Mat resized_mask_mat(box.height, box.width, CV_32FC1);
        cv::resize(mask_mat, resized_mask_mat, cv::Size(box.width, box.height));
        cv::Mat uchar_resized_mask(box.height, box.width, CV_8UC3,color);
        roi_img.copyTo(uchar_resized_mask, resized_mask_mat <= 0.5);
        cv::addWeighted(uchar_resized_mask, 0.7, roi_img, 0.3, 0.0f, roi_img);
        cv::putText(src, coco_labels[label].c_str(), box.tl()+(box.br()-box.tl())/2, cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 0, 255), 1, 8);
    }
}