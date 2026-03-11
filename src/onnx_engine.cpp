#include "onnx_engine.hpp"
#include <iostream>

namespace
{
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

ONNXEngine::ONNXEngine(const std::string &modelPath)
    : env(ORT_LOGGING_LEVEL_WARNING, "onnx_engine")
{
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    session = std::make_unique<Ort::Session>(
        env,
        modelPath.c_str(),
        session_options);

    printModelInfo(*session);

    Ort::AllocatorWithDefaultOptions allocator;

    size_t inputCount = session->GetInputCount();
    size_t outputCount = session->GetOutputCount();

    for (size_t i = 0; i < inputCount; i++)
    {
        inputNamesStore.emplace_back(session->GetInputNameAllocated(i, allocator));
        inputNames.push_back(inputNamesStore.back().get());
    }

    for (size_t i = 0; i < outputCount; i++)
    {
        outputNamesStore.emplace_back(session->GetOutputNameAllocated(i, allocator));
        outputNames.push_back(outputNamesStore.back().get());
    }
}

std::vector<Ort::Value> ONNXEngine::run(const std::vector<Ort::Value> &inputs)
{
    Ort::RunOptions runOptions;

    return session->Run(
        runOptions,
        inputNames.data(),
        inputs.data(),
        inputs.size(),
        outputNames.data(),
        outputNames.size());
}