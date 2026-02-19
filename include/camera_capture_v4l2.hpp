#pragma once
#include "icamera_capture.hpp"
#include <vector>
#include <optional>

class CaptureCameraV4L2 : public iCameraCapture
{
public:
    CaptureCameraV4L2();
    ~CaptureCameraV4L2();

    virtual int startStreaming(const char *dev, unsigned int wdth, unsigned int height, IMAGE_FORMAT format) override;
    virtual cv::Mat getFrame() override;
    virtual int stopStreaming() override;

private:
    int xioctl(int fd, unsigned long int request, void *arg);
    static std::optional<unsigned int> mapFormat(IMAGE_FORMAT format);

private:
    struct Buffer
    {
        void *start;
        size_t length;
    };
    std::vector<Buffer> _buffers;
    int _fd{-1};
};
