#pragma once

#include "onnx_i_model_adapter.hpp"
#include "onnx_engine.hpp"

class BlazeFaceModel : public iModelAdapter
{
public:
    BlazeFaceModel(const std::string &modelPath);

    virtual std::vector<Detection> infer(const cv::Mat &image) override;

private:
    ONNXEngine engine;

    std::vector<float> imageBuffer;

    Ort::Value imageTensor{nullptr};
    Ort::Value scoreTensor{nullptr};
    Ort::Value iouTensor{nullptr};
    Ort::Value maxTensor{nullptr};

    std::vector<Ort::Value> inputTensors;

    std::vector<int64_t> imageShape{1,128,128,3};
    std::vector<int64_t> scalarShape{1};

    Ort::MemoryInfo memoryInfo;

    float score_threshold {0.4f};
    float iou_threshold {0.3f};
    int64_t max_detections {10};
};