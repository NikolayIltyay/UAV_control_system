#include <iostream>
#include <thread>
#include <atomic>
#include <csignal>
#include <opencv2/opencv.hpp>
#include "fps_logger.hpp"
#include "onnx_blaze_face_model.hpp"
#include "utils.hpp"

std::atomic<bool> running = true;
std::atomic<bool> runCapture = true;
std::atomic<bool> runInference = true;

std::atomic<std::shared_ptr<cv::Mat>> lastCaptureFrame;
std::atomic<std::shared_ptr<cv::Mat>> frameToRender;

void signalHandler(int)
{
    runCapture = false;
    runInference = false;
    running = false;
}

void captureCamera(const char *dev, unsigned int wdth, unsigned int height)
{
    cv::VideoCapture capture;
    capture.open(dev, cv::CAP_V4L2);
    if (!capture.isOpened())
    {
        std::cerr << "Cannot open camera device." << dev << std::endl;
        return;
    }

    if (!capture.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G')))
        std::cout << "WARNING: Failed to set format" << std::endl;

    capture.set(cv::CAP_PROP_FRAME_WIDTH, wdth);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, height);

    capture.set(cv::CAP_PROP_FPS, 30);

    while (runCapture)
    {
        cv::Mat frame;

        if (!capture.read(frame))
        {
            std::cout << "Failed to read frame from camera." << std::endl;
            continue;
        }

        lastCaptureFrame.store(std::make_shared<cv::Mat>(frame.clone()), std::memory_order_release);
    }

    capture.release();
}

void inference(const char *model)
{
    BlazeFaceModel blazeModel(model);

    while (runInference)
    {
        std::shared_ptr<cv::Mat> lastFrame = lastCaptureFrame.exchange(nullptr, std::memory_order_acquire);
        if (!lastFrame)
            continue;

        auto outlines = blazeModel.infer(*lastFrame);
        drawDetections(outlines, *lastFrame);

        frameToRender.store(std::move(lastFrame), std::memory_order_release);
    }
}

int main(int argc, char **argv)
{
    std::signal(SIGINT, signalHandler);

    const auto model = (argc > 2) ? argv[2] : nullptr;

    if (!model)
    {
        std::cerr << "model is not specified" << std::endl;
        return -1;
    }

    auto dev = (argc > 1) ? argv[1] : "/dev/video0";

    unsigned int width = 1280;
    unsigned int height = 720;

    std::thread captureCameraThread(captureCamera, dev, width, height);
    std::thread inferenceThread(inference, model);

    FpsLogger fpsLog;

    cv::VideoWriter writer(
        "output.mp4",
        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
        30,
        cv::Size(width, height));

    std::shared_ptr<cv::Mat> frame;

    while (running)
    {
        frame = frameToRender.exchange(nullptr, std::memory_order_acquire);
        if (!frame)
            continue;

        writer.write(*frame);
        fpsLog.update();
    }

    writer.release();
    captureCameraThread.join();
    inferenceThread.join();

    return 0;
}
