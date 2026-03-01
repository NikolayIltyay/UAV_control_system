#pragma once
#include <string>
#include <optional>
#include <opencv2/opencv.hpp>

struct Detection
{
    cv::Rect _rect;
    float _score{0.0f};
};

class iDetectionModelRT
{
public:
    virtual ~iDetectionModelRT() {};

    virtual int initialize(const std::string &modelPath) = 0;

    virtual std::vector<Detection> getOutlines(const cv::Mat &image) = 0;
};