#include <iostream>
#include <thread>
#include <atomic>
#include <csignal>
#include <sys/syscall.h>
#include <sstream>

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

void captureCamera(const char *dev, unsigned int width, unsigned int height)
{
    std::cout << "captureCamera TID = " << syscall(SYS_gettid) << std::endl;

    std::stringstream ss;
    ss << "libcamerasrc ! video/x-raw,format=BGR,width=";
    ss << width;
    ss << ",height=";
    ss << height;
    ss << ",framerate=30/1 ! appsink";

    cv::VideoCapture capture(
        ss.str(),
        cv::CAP_GSTREAMER);

    if (!capture.isOpened())
    {
        std::cerr << "Cannot open camera device " << dev << std::endl;
        return;
    }

    while (runCapture)
    {
        cv::Mat frame;

        if (!capture.read(frame))
        {
            std::cout << "Failed to read frame from camera." << std::endl;
            return;
        }

        lastCaptureFrame.store(std::make_shared<cv::Mat>(frame.clone()), std::memory_order_release);
    }

    capture.release();
}


void inference(const char *model)
{
    std::cout << "inference TID = " << syscall(SYS_gettid) << std::endl;
    BlazeFaceModel blazeModel(model);

    while (runInference)
    {
        std::shared_ptr<cv::Mat> lastFrame = lastCaptureFrame.exchange(nullptr, std::memory_order_acquire);
        if (!lastFrame)
        {
            std::this_thread::sleep_for(std::chrono::duration(std::chrono::milliseconds(1)));
            continue;
        }         

        auto outlines = blazeModel.infer(*lastFrame);
        drawDetections(outlines, *lastFrame);

        frameToRender.store(std::move(lastFrame), std::memory_order_release);
    }
}

int main(int argc, char **argv)
{
    std::cout << "main TID = " << syscall(SYS_gettid) << std::endl;

    std::signal(SIGINT, signalHandler);

    const auto model = (argc > 2) ? argv[2] : nullptr;

    if (!model)
    {
        std::cerr << "model is not specified" << std::endl;
        return -1;
    }

    auto dev = (argc > 1) ? argv[1] : "/dev/video0";

    unsigned int width = 1920;
    unsigned int height = 1080;

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
        {
            std::this_thread::sleep_for(std::chrono::duration(std::chrono::milliseconds(1)));
            continue;
        }

        writer.write(*frame);
        fpsLog.update();
    }

    writer.release();
    captureCameraThread.join();
    inferenceThread.join();

    return 0;
}
