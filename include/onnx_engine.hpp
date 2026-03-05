#pragma once

#include <string>

#include <onnxruntime_cxx_api.h>

class ONNXEngine
{
public:
    ONNXEngine(const std::string &modelPath);

    std::vector<Ort::Value> run(const std::vector<Ort::Value> &inputs);

private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    OrtCUDAProviderOptions cuda_options;
    std::unique_ptr<Ort::Session> session;

    std::vector<const char *> inputNames;
    std::vector<const char *> outputNames;

    std::vector<Ort::AllocatedStringPtr> inputNamesStore;
    std::vector<Ort::AllocatedStringPtr> outputNamesStore;
};
