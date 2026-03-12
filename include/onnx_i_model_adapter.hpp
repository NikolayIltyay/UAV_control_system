#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include "data_types.hpp"



class iModelAdapter
{
public:
    virtual std::vector<Detection> infer(const cv::Mat &image) = 0;
};