#pragma once
#include "icamera_capture.hpp"
#include <optional>
#include <opencv2/opencv.hpp>


class CaptureCameraOpencv : public iCameraCapture
{
public:
    CaptureCameraOpencv();
    ~CaptureCameraOpencv();

    virtual int startStreaming(const char *dev, unsigned int wdth, unsigned int height, IMAGE_FORMAT format) override;
    virtual cv::Mat getFrame() override;
    virtual int stopStreaming() override;

private:

static std::optional<int> mapFormat(IMAGE_FORMAT format);

private:
cv::VideoCapture _cap{};

};