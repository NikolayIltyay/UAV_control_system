#include "onnx_detection_model_rt.hpp"
#include <iostream>

namespace
{
    constexpr int MODEL_WIDTH = 128;
    constexpr int MODEL_HEIGHT = 128;

    constexpr float SCORE_THRESHOLD = 0.4f;
    constexpr float IOU_THRESHOLD = 0.3f;
    constexpr int64_t MAX_DETECTIONS = 10;

    void printModelInfo(const Ort::Session &session)
    {
        std::cout << "Number of inputs: " << session.GetInputCount() << std::endl;
        Ort::AllocatorWithDefaultOptions allocator;

        for (size_t i = 0; i < session.GetInputCount(); ++i)
        {
            auto name = session.GetInputNameAllocated(i, allocator);
            auto type_info = session.GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            auto shape = tensor_info.GetShape();

            std::cout << "Input " << i << ": " << name.get() << " Shape: ";
            for (auto s : shape)
                std::cout << s << " ";
            std::cout << std::endl;
        }
    }
}

ONNXDetectionModel::~ONNXDetectionModel()
{
}

int ONNXDetectionModel::initialize(const std::string &modelPath)
{
    // ---------------------------
    // Initialize ONNX Runtime
    // ---------------------------
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    _session = Ort::Session(
        _env,
        modelPath.c_str(),
        session_options);

    printModelInfo(_session);

    auto inputCount = _session.GetInputCount();
    auto outputCount = _session.GetOutputCount();

    Ort::AllocatorWithDefaultOptions allocator;

    for (decltype(inputCount) i = 0; i < inputCount; i++)
    {
        _inputNamesStore.emplace_back(_session.GetInputNameAllocated(i, allocator));
        _inputNames.push_back(_inputNamesStore.back().get());
    }

    for (decltype(outputCount) i = 0; i < outputCount; i++)
    {
        _outputNamesStore.emplace_back(_session.GetOutputNameAllocated(i, allocator));
        _outputNames.push_back(_outputNamesStore.back().get());
    }

    return 0;
}

std::vector<Detection> ONNXDetectionModel::getOutlines(const cv::Mat &image)
{
    int orig_w = image.cols;
    int orig_h = image.rows;

    // ---------------------------
    // Preprocess
    // ---------------------------
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(MODEL_WIDTH, MODEL_HEIGHT));
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

    // ---------------------------
    // Create scalar tensors
    // ---------------------------
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

    Ort::Value input_tensors[] = {
        std::move(image_tensor),
        std::move(score_tensor),
        std::move(max_det_tensor),
        std::move(iou_tensor)};

    Ort::RunOptions run_options;

    // ---------------------------
    // Run Inference
    // ---------------------------
    auto output_tensors = _session.Run(
        run_options,
        _inputNames.data(),
        input_tensors,
        4,
        _outputNames.data(),
        1);

    float *output_data =
        output_tensors[0].GetTensorMutableData<float>();

    // ---------------------------
    // Draw detections (robust)
    // ---------------------------
    auto output_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> output_shape = output_info.GetShape();

    int64_t num_detections = 0;
    int64_t elements_per_detection = 16;

    // Determine detection count
    if (output_shape.size() == 3)
    {
        // shape: [1, N, 16]
        num_detections = output_shape[1];
    }
    else if (output_shape.size() == 2)
    {
        // shape: [1, 16] OR [1, 0]
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

    for (int64_t i = 0; i < num_detections; ++i)
    {
        int64_t offset = i * elements_per_detection;

        float score = output_data[offset + 15];

        if (score < SCORE_THRESHOLD)
            continue;

        int y1 = static_cast<int>(output_data[offset + 0] * (float)orig_h);
        int x1 = static_cast<int>(output_data[offset + 1] * (float)orig_w);
        int y2 = static_cast<int>(output_data[offset + 2] * (float)orig_h);
        int x2 = static_cast<int>(output_data[offset + 3] * (float)orig_w);

        x1 = std::max(0, std::min(orig_w, x1));
        y1 = std::max(0, std::min(orig_h, y1));
        x2 = std::max(0, std::min(orig_w, x2));
        y2 = std::max(0, std::min(orig_h, y2));

        outlines.emplace_back(Detection{cv::Rect(x1, y1, x2 - x1, y2 - y1), score});
    }

    return outlines;
}