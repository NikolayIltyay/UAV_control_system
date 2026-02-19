#include "camera_capture_factory.hpp"

#if CAMERA_CAPTURE_BACKEND == V4L2
#include "camera_capture_v4l2.hpp"
#elif CAMERA_CAPTURE_BACKEND == OPENCV
#include "camera_capture_opencv.hpp"
#endif

std::unique_ptr<iCameraCapture> CameraCaptureFactory::getCameraCapture()
{
#if CAMERA_CAPTURE_BACKEND == V4L2
    return std::make_unique<CaptureCameraV4L2>();
#elif CAMERA_CAPTURE_BACKEND == OPENCV
    return std::make_unique<CaptureCameraOpencv>();
#endif
}