#pragma once
#include <string>
#include <optional>
#include <opencv2/opencv.hpp>

struct Detection
{
    cv::Rect _rect;
    cv::Point2i _leftEye;
    cv::Point2i _rightEye;
    cv::Point2i _nose;
    cv::Point2i _mouth;
    cv::Point2i _leftEar;
    cv::Point2i _rightEar;
};

class iDetectionModelRT
{
public:
    virtual ~iDetectionModelRT() {};

    virtual int initialize(const std::string &modelPath) = 0;

    virtual std::vector<Detection> getOutlines(const cv::Mat &image) = 0;
};