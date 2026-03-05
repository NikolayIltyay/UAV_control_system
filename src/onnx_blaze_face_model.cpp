#include "onnx_blaze_face_model.hpp"
#include <iostream>

namespace
{
    constexpr int MODEL_WIDTH = 128;
    constexpr int MODEL_HEIGHT = 128;

    constexpr float SCORE_THRESHOLD = 0.4f;
    constexpr float IOU_THRESHOLD = 0.3f;
    constexpr int64_t MAX_DETECTIONS = 2;
}

BlazeFaceModel::BlazeFaceModel(const std::string &modelPath)
    : engine(modelPath)
{
}

std::vector<Detection> BlazeFaceModel::infer(const cv::Mat &image)
{
    int orig_w = image.cols;
    int orig_h = image.rows;

    if (orig_w <= 0 || orig_h <= 0)
    {
        std::cout << "Invalid image size";
        return {};
    }

    int delta = orig_w / 2 - orig_h / 2;
    int x_start = std::max(delta, 0);
    int y_start = std::max(-delta, 0);

    int minSide = std::min(orig_w, orig_h);

    cv::Rect roi(x_start, y_start, minSide, minSide);
    cv::Mat resized = image;
    resized = resized(roi);

    cv::resize(resized, resized, cv::Size(MODEL_WIDTH, MODEL_HEIGHT));
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    resized.convertTo(resized, CV_32F, 1.0f / 255.0f);

    std::vector<int64_t> input_shape = {1, MODEL_HEIGHT, MODEL_WIDTH, 3};

    std::vector<float> input_tensor_values(resized.total() * 3);
    std::memcpy(input_tensor_values.data(),
                resized.data,
                input_tensor_values.size() * sizeof(float));

    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value image_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor_values.data(),
        input_tensor_values.size(),
        input_shape.data(),
        input_shape.size());

    std::vector<int64_t> scalar_shape = {1};

    float score_threshold = SCORE_THRESHOLD;
    float iou_threshold = IOU_THRESHOLD;
    int64_t max_detections = MAX_DETECTIONS;

    Ort::Value score_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        &score_threshold,
        1,
        scalar_shape.data(),
        scalar_shape.size());

    Ort::Value iou_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        &iou_threshold,
        1,
        scalar_shape.data(),
        scalar_shape.size());

    Ort::Value max_det_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info,
        &max_detections,
        1,
        scalar_shape.data(),
        scalar_shape.size());

    std::vector<Ort::Value> input_tensors;

    input_tensors.push_back(std::move(image_tensor));
    input_tensors.push_back(std::move(score_tensor));
    input_tensors.push_back(std::move(max_det_tensor));
    input_tensors.push_back(std::move(iou_tensor));

    auto output_tensors = engine.run(input_tensors);

    float *output_data =
        output_tensors[0].GetTensorMutableData<float>();

    auto output_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> output_shape = output_info.GetShape();

    int64_t num_detections = 0;
    int64_t elements_per_detection = 16;

    if (output_shape.size() == 3)
    {
        // shape: [1, N, 16]
        num_detections = output_shape[1];
    }
    else if (output_shape.size() == 2)
    {
        // shape: [1, 16]
        if (output_shape[1] == 16)
            num_detections = 1;
        else
            num_detections = 0;
    }
    else
    {
        std::cout << "Unexpected output shape: ";
        for (auto s : output_shape)
            std::cout << s << " ";
        std::cout << std::endl;
        return {};
    }

    if (num_detections == 0)
        return {};

    std::vector<Detection> outlines;

    auto getImageCoord = [output_data, &minSide](int i, int idxOffset, int offset)
    {
        int coord = static_cast<int>(output_data[idxOffset + i] * (float)minSide);
        return std::max(0, std::min(minSide, coord)) + offset;
    };

    for (int64_t i = 0; i < num_detections; ++i)
    {
        int offset = static_cast<int>(i * elements_per_detection);

        float score = output_data[offset + 15];

        if (score < SCORE_THRESHOLD)
            continue;

        int y1 = getImageCoord(0, offset, y_start);
        int x1 = getImageCoord(1, offset, x_start);
        int y2 = getImageCoord(2, offset, y_start);
        int x2 = getImageCoord(3, offset, x_start);

        int leftEye_x = getImageCoord(4, offset, x_start);
        int leftEye_y = getImageCoord(5, offset, y_start);

        int rightEye_x = getImageCoord(6, offset, x_start);
        int rightEye_y = getImageCoord(7, offset, y_start);

        int Nose_x = getImageCoord(8, offset, x_start);
        int Nose_y = getImageCoord(9, offset, y_start);

        int Mouth_x = getImageCoord(10, offset, x_start);
        int Mouth_y = getImageCoord(11, offset, y_start);

        int leftEar_x = getImageCoord(12, offset, x_start);
        int leftEar_y = getImageCoord(13, offset, y_start);

        int rightEar_x = getImageCoord(14, offset, x_start);
        int rightEar_y = getImageCoord(15, offset, y_start);

        outlines.emplace_back(Detection{cv::Rect(x1, y1, x2 - x1, y2 - y1),
                                        cv::Point2i(leftEye_x, leftEye_y),
                                        cv::Point2i(rightEye_x, rightEye_y),
                                        cv::Point2i(Nose_x, Nose_y),
                                        cv::Point2i(Mouth_x, Mouth_y),
                                        cv::Point2i(leftEar_x, leftEar_y),
                                        cv::Point2i(rightEar_x, rightEar_y)});
    }

    return outlines;
}