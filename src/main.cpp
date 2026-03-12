#include <iostream>
#include <atomic>
#include <csignal>
#include <opencv2/opencv.hpp>
#include "fps_logger.hpp"
#include "camera_capture_factory.hpp"
#include "icamera_capture.hpp"
#include "onnx_blaze_face_model.hpp"
#include "utils.hpp"

std::atomic<bool> running = true;

void signalHandler(int)
{
    running = false;
}

int main(int argc, char **argv)
{
    std::signal(SIGINT, signalHandler);

    const auto model = (argc > 2) ? argv[2] : nullptr;

    if(!model)
    {
        std::cerr << "model is not specified" << std::endl;
        return -1;
    }

    BlazeFaceModel blazeModel(model);

    auto dev = (argc > 1) ? argv[1] : "/dev/video0";
    auto capture = CameraCaptureFactory::getCameraCapture();
    unsigned int width = 1280;
    unsigned int height = 720;

    if (capture->startStreaming(dev, width, height, IMAGE_FORMAT::MJPEG))
    {
        std::cerr << "start streaming failed" << std::endl;
        return -1;
    }

    FpsLogger fpsLog;



    cv::VideoWriter writer(
        "output.mp4",
        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
        30,
        cv::Size(width, height));

    cv::Mat frame;

    while (running)
    {
        frame = capture->getFrame();
        if (frame.empty())
            continue;

        auto outlines = blazeModel.infer(frame);
        drawDetections(outlines, frame);


        writer.write(frame);

        fpsLog.update();
    }

    writer.release();
    capture->stopStreaming();
    return 0;
}
