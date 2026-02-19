#include "camera_capture_v4l2.hpp"
#include <iostream>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <opencv2/opencv.hpp>

CaptureCameraV4L2::CaptureCameraV4L2()
{
}

CaptureCameraV4L2::~CaptureCameraV4L2()
{
    stopStreaming();
}

int CaptureCameraV4L2::startStreaming(const char *dev, unsigned int wdth, unsigned int height, IMAGE_FORMAT format)
{
    _fd = open(dev, O_RDWR);
    if (_fd < 0)
    {
        std::cerr << "open" << std::endl;
        return -1;
    }

    // --------------------------------------------------
    // Set format
    // --------------------------------------------------
    v4l2_format fmt{};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = wdth;
    fmt.fmt.pix.height = height;

    auto v4l2_format = mapFormat(format);

    if (!v4l2_format)
    {
        std::cerr << "mapFormat failed" << std::endl;
        return -1;
    }

    fmt.fmt.pix.pixelformat = v4l2_format.value();
    fmt.fmt.pix.field = V4L2_FIELD_ANY;

    if (xioctl(_fd, VIDIOC_S_FMT, &fmt))
        return -1;

    // --------------------------------------------------
    // Request buffers
    // --------------------------------------------------
    v4l2_requestbuffers req{};
    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    if (xioctl(_fd, VIDIOC_REQBUFS, &req))
        return -1;

    _buffers.resize(req.count);

    // --------------------------------------------------
    // Map buffers
    // --------------------------------------------------
    for (unsigned i = 0; i < req.count; ++i)
    {
        v4l2_buffer buf{};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;

        if (xioctl(_fd, VIDIOC_QUERYBUF, &buf))
            return -1;

        _buffers[i].length = buf.length;
        _buffers[i].start = mmap(NULL, buf.length,
                                 PROT_READ | PROT_WRITE,
                                 MAP_SHARED,
                                 _fd, buf.m.offset);

        if (_buffers[i].start == MAP_FAILED)
        {
            std::cerr << "mmap failed" << std::endl;
            return -1;
        }
    }

    // --------------------------------------------------
    // Queue buffers
    // --------------------------------------------------
    for (unsigned i = 0; i < _buffers.size(); ++i)
    {
        v4l2_buffer buf{};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        if (xioctl(_fd, VIDIOC_QBUF, &buf))
            return -1;
    }

    // --------------------------------------------------
    // Start streaming
    // --------------------------------------------------
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(_fd, VIDIOC_STREAMON, &type))
        return -1;

    return 0;
}

cv::Mat CaptureCameraV4L2::getFrame()
{
    v4l2_buffer buf{};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    if (xioctl(_fd, VIDIOC_DQBUF, &buf))
        return cv::Mat();

    // MJPEG -> decode to BGR using OpenCV
    std::vector<uint8_t> jpeg(
        (uint8_t *)_buffers[buf.index].start,
        (uint8_t *)_buffers[buf.index].start + buf.bytesused);

    if (xioctl(_fd, VIDIOC_QBUF, &buf))
        std::cerr << "VIDIOC_QBUF failed" << std::endl;

    return cv::imdecode(jpeg, cv::IMREAD_COLOR);
}

int CaptureCameraV4L2::stopStreaming()
{
    int result = 0;

    if (_fd != -1)
    {
        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (xioctl(_fd, VIDIOC_STREAMOFF, &type))
            result = -1;

        close(_fd);
        _fd = -1;
    }

    for (auto &b : _buffers)
        munmap(b.start, b.length);

    _buffers.clear();

    return result;
}

int CaptureCameraV4L2::xioctl(int fd, unsigned long int request, void *arg)
{
    if (ioctl(fd, request, arg) == -1)
    {
        std::cerr << "ioctl failed" << std::endl;
        return -1;
    }

    return 0;
}

std::optional<unsigned int> CaptureCameraV4L2::mapFormat(IMAGE_FORMAT format)
{
    switch (format)
    {
    case IMAGE_FORMAT::MJPEG:
        return V4L2_PIX_FMT_MJPEG;

    case IMAGE_FORMAT::YUYV:
        return V4L2_PIX_FMT_YUYV;

    default:
        return std::nullopt;
    }
}