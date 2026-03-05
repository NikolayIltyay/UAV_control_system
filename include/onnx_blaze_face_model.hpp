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
};