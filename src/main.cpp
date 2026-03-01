#include <iostream>
#include <opencv2/opencv.hpp>
#include "fps_logger.hpp"
#include "camera_capture_factory.hpp"
#include "icamera_capture.hpp"
#include "onnx_detection_model_rt.hpp"



int main(int argc, char **argv)
{
    ONNXDetectionModel onnxRT;
    onnxRT.initialize("models/End-to-end-BlazeFace-Onnx/mpipe_bface_boxes_ops16.onnx");


    auto dev = (argc > 1) ? argv[1] : "/dev/video2";

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
        if (frame.empty())
            continue;

        auto outlines = onnxRT.getOutlines(frame);

        for (const auto& outline : outlines)
        {

            cv::rectangle(frame,
                          outline._rect.tl(),
                          outline._rect.br(),
                          cv::Scalar(0, 255, 0),
                          2);

            char text[32];
            snprintf(text, sizeof(text), "%.2f", outline._score);

            cv::putText(frame,
                        text,
                        cv::Point(outline._rect.tl().x, (outline._rect.tl().y - 5)),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.5,
                        cv::Scalar(0, 255, 0),
                        1);
        }

        cv::imshow("Face Detection", frame);

        if (cv::waitKey(1) == 27)
            break;

        fpsLog.update();
    }

    capture->stopStreaming();
    return 0;
}
