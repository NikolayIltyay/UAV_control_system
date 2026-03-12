#pragma once
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