#pragma once
#include <opencv2/core/mat.hpp>

enum class IMAGE_FORMAT
{
    MJPEG,
    YUYV
};

class iCameraCapture
{
    public:

    virtual ~iCameraCapture(){};

    virtual int startStreaming(const char* dev, unsigned int wdth, unsigned int height, IMAGE_FORMAT format) = 0;
    virtual cv::Mat getFrame() = 0;
    virtual int stopStreaming() = 0;
};