#include <iostream>
#include <vector>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <opencv2/opencv.hpp>

struct Buffer {
    void* start;
    size_t length;
};

static void xioctl(int fd, unsigned long int request, void* arg)
{
    if (ioctl(fd, request, arg) == -1) {
        perror("ioctl");
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv)
{
    const char* dev = (argc > 1) ? argv[1] : "/dev/video0";

    int fd = open(dev, O_RDWR);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    // --------------------------------------------------
    // Set format MJPEG 640x480
    // --------------------------------------------------
    v4l2_format fmt{};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = 1280;
    fmt.fmt.pix.height = 720;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
    fmt.fmt.pix.field = V4L2_FIELD_ANY;

    xioctl(fd, VIDIOC_S_FMT, &fmt);

    // --------------------------------------------------
    // Request buffers
    // --------------------------------------------------
    v4l2_requestbuffers req{};
    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    xioctl(fd, VIDIOC_REQBUFS, &req);

    std::vector<Buffer> buffers(req.count);

    // --------------------------------------------------
    // Map buffers
    // --------------------------------------------------
    for (unsigned i = 0; i < req.count; ++i)
    {
        v4l2_buffer buf{};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;

        xioctl(fd, VIDIOC_QUERYBUF, &buf);

        buffers[i].length = buf.length;
        buffers[i].start = mmap(NULL, buf.length,
                                PROT_READ | PROT_WRITE,
                                MAP_SHARED,
                                fd, buf.m.offset);

        if (buffers[i].start == MAP_FAILED) {
            perror("mmap");
            return 1;
        }
    }

    // --------------------------------------------------
    // Queue buffers
    // --------------------------------------------------
    for (unsigned i = 0; i < buffers.size(); ++i)
    {
        v4l2_buffer buf{};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        xioctl(fd, VIDIOC_QBUF, &buf);
    }

    // --------------------------------------------------
    // Start streaming
    // --------------------------------------------------
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    xioctl(fd, VIDIOC_STREAMON, &type);

        // FPS counters
    size_t frames = 0;
    auto t0 = std::chrono::steady_clock::now();

    // --------------------------------------------------
    // Capture loop
    // --------------------------------------------------
    while (true)
    {
        v4l2_buffer buf{};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;

        xioctl(fd, VIDIOC_DQBUF, &buf);

        // MJPEG -> decode to BGR using OpenCV
        std::vector<uint8_t> jpeg(
            (uint8_t*)buffers[buf.index].start,
            (uint8_t*)buffers[buf.index].start + buf.bytesused
        );

        cv::Mat frame = cv::imdecode(jpeg, cv::IMREAD_COLOR);
        if (!frame.empty())
            cv::imshow("V4L2 Camera", frame);

        if (cv::waitKey(1) == 27)
            break;

        frames++;
        auto t1 = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        if (sec >= 1.0) {
            std::cout << "FPS: " << frames / sec << std::endl;
            frames = 0;
            t0 = t1;
        }

        xioctl(fd, VIDIOC_QBUF, &buf);
    }

    // --------------------------------------------------
    // Stop
    // --------------------------------------------------
    xioctl(fd, VIDIOC_STREAMOFF, &type);

    for (auto& b : buffers)
        munmap(b.start, b.length);

    close(fd);
    return 0;
}

