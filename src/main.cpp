#include <iostream>
#include <opencv2/opencv.hpp>
#include "fps_logger.hpp"
#include "camera_capture_factory.hpp"
#include "icamera_capture.hpp"



int main(int argc, char** argv)
{
    const char *dev = (argc > 1) ? argv[1] : "/dev/video0";

    auto capture = CameraCaptureFactory::getCameraCapture();

    if (capture->startStreaming(dev, 1280, 720, IMAGE_FORMAT::MJPEG))
    {
        std::cerr << "start streaming failed" << std::endl;
        return -1;
    }

    FpsLogger fpsLog;
    while (true)
    {
        cv::Mat frame = capture->getFrame();
        if (!frame.empty())
            cv::imshow("Camera", frame);

        if (cv::waitKey(1) == 27)
            break;

        fpsLog.update();
    }

    if (capture->stopStreaming())
    {
        std::cerr << "stop streaming failed" << std::endl;
        return -1;
    }

    return 0;
}

