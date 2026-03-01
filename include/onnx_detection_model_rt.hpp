#pragma once
#include "idetection_model_rt.hpp"
#include <onnxruntime_cxx_api.h>

class ONNXDetectionModel : public iDetectionModelRT
{
public:
    ~ONNXDetectionModel();

    virtual int initialize(const std::string &modelPath) override;

    virtual std::vector<Detection> getOutlines(const cv::Mat &image) override;

    private:
    Ort::Session _session{nullptr};
    Ort::Env _env{ORT_LOGGING_LEVEL_WARNING, "blazeface"};

    std::vector<Ort::AllocatedStringPtr> _inputNamesStore;
    std::vector<const char*> _inputNames;

    std::vector<Ort::AllocatedStringPtr> _outputNamesStore;
    std::vector<const char*> _outputNames;
};
