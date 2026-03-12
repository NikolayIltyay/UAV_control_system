#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include "data_types.hpp"


void drawDetections(const std::vector<Detection>& detections, cv::Mat& frame);