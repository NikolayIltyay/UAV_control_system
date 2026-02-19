#include "camera_capture_opencv.hpp"

CaptureCameraOpencv::CaptureCameraOpencv()
{
}

CaptureCameraOpencv::~CaptureCameraOpencv()
{
}

int CaptureCameraOpencv::startStreaming(const char *dev, unsigned int wdth, unsigned int height, IMAGE_FORMAT format)
{
    _cap.open(dev, cv::CAP_V4L2);
    if (!_cap.isOpened())
    {
        std::cerr << "Cannot open camera device." << dev << std::endl;
        return -1;
    }

    auto opencv_format = mapFormat(format);

    if (!opencv_format)
    {
        std::cerr << "mapFormat failed" << std::endl;
        return -1;
    }

    if (!_cap.set(cv::CAP_PROP_FOURCC, opencv_format.value()))
        std::cout << "WARNING: Failed to set format" << std::endl;

    _cap.set(cv::CAP_PROP_FRAME_WIDTH, wdth);
    _cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);


    _cap.set(cv::CAP_PROP_FPS, 30);

    return 0;
}

cv::Mat CaptureCameraOpencv::getFrame()
{
    cv::Mat frame;

    if (!_cap.read(frame))
        std::cerr << "Failed to read frame from camera." << std::endl;

    return frame;
}

int CaptureCameraOpencv::stopStreaming()
{
    _cap.release();
    return 0;
}

std::optional<int> CaptureCameraOpencv::mapFormat(IMAGE_FORMAT format)
{
    switch (format)
    {
    case IMAGE_FORMAT::MJPEG:
        return cv::VideoWriter::fourcc('M', 'J', 'P', 'G');

    case IMAGE_FORMAT::YUYV:
        return cv::VideoWriter::fourcc('Y', 'U', 'Y', 'V');

    default:
        return std::nullopt;
    }
}