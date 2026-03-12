#include "utils.hpp"

namespace
{
    bool renderFrame = false;
    bool renderDots = false;
    bool renderTarget = true;
}

void drawDetections(const std::vector<Detection> &detections, cv::Mat& frame)
{
    for (const auto &detection : detections)
    {
        if (renderFrame)
            cv::rectangle(frame,
                          detection._rect.tl(),
                          detection._rect.br(),
                          cv::Scalar(0, 255, 0),
                          2);

        if (renderDots)
        {
            cv::circle(frame, detection._leftEye, 4, cv::Scalar(255, 0, 0), -1);
            cv::circle(frame, detection._rightEye, 4, cv::Scalar(255, 0, 0), -1);

            cv::circle(frame, detection._leftEar, 4, cv::Scalar(255, 0, 0), -1);
            cv::circle(frame, detection._rightEar, 4, cv::Scalar(255, 0, 0), -1);

            cv::circle(frame, detection._mouth, 4, cv::Scalar(255, 0, 0), -1);
            cv::circle(frame, detection._nose, 4, cv::Scalar(255, 0, 0), -1);
        }

        if (renderTarget)
        {
            auto centerPoint = (detection._leftEye + detection._rightEye) / 2;
            auto segmentLength = frame.rows / 5;
            cv::Point2i dX(segmentLength, 0);
            cv::Point2i dY(0, segmentLength);

            cv::line(frame, centerPoint - dY, centerPoint + dY, cv::Scalar(0, 0, 255), 2);
            cv::line(frame, centerPoint - dX, centerPoint + dX, cv::Scalar(0, 0, 255), 2);
            cv::circle(frame, centerPoint, segmentLength / 2, cv::Scalar(0, 0, 255), 2);
            cv::circle(frame, centerPoint, segmentLength / 4, cv::Scalar(0, 0, 255), 2);
        }
    }
}