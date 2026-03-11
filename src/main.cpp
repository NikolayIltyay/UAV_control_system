#include <iostream>
#include <opencv2/opencv.hpp>
#include "fps_logger.hpp"
#include "camera_capture_factory.hpp"
#include "icamera_capture.hpp"
#include "onnx_blaze_face_model.hpp"



int main(int argc, char **argv)
{
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

    bool renderFrame= false;
    bool renderDots = false;
    bool renderTarget = true;

    cv::VideoWriter writer(
        "output.mp4",
        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
        30,
        cv::Size(width, height));

    cv::Mat frame;

    while (true)
    {
        cv::Mat frame = capture->getFrame();
        if (frame.empty())
            continue;

        auto outlines = blazeModel.infer(frame);

        for (const auto &outline : outlines)
        {
            if (renderFrame)
                cv::rectangle(frame,
                              outline._rect.tl(),
                              outline._rect.br(),
                              cv::Scalar(0, 255, 0),
                              2);

            if (renderDots)
            {
                cv::circle(frame, outline._leftEye, 4, cv::Scalar(255, 0, 0), -1);
                cv::circle(frame, outline._rightEye, 4, cv::Scalar(255, 0, 0), -1);

                cv::circle(frame, outline._leftEar, 4, cv::Scalar(255, 0, 0), -1);
                cv::circle(frame, outline._rightEar, 4, cv::Scalar(255, 0, 0), -1);

                cv::circle(frame, outline._mouth, 4, cv::Scalar(255, 0, 0), -1);
                cv::circle(frame, outline._nose, 4, cv::Scalar(255, 0, 0), -1);
            }

            if (renderTarget)
            {
                auto centerPoint = (outline._leftEye + outline._rightEye) / 2;
                auto segmentLength = frame.rows / 5;
                cv::Point2i dX(segmentLength, 0);
                cv::Point2i dY(0, segmentLength);

                cv::line(frame, centerPoint - dY, centerPoint + dY, cv::Scalar(0, 0, 255), 2);
                cv::line(frame, centerPoint - dX, centerPoint + dX, cv::Scalar(0, 0, 255), 2);
                cv::circle(frame, centerPoint, segmentLength / 2, cv::Scalar(0, 0, 255), 2);
                cv::circle(frame, centerPoint, segmentLength / 4, cv::Scalar(0, 0, 255), 2);
            }
        }

        writer.write(frame);

        cv::imshow("Face Detection", frame);

        if (cv::waitKey(1) == 27)
            break;

        fpsLog.update();
    }

    capture->stopStreaming();
    return 0;
}
