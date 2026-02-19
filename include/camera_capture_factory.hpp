#pragma once
#include <memory>

class iCameraCapture;

class CameraCaptureFactory
{
    public:
    static std::unique_ptr<iCameraCapture> getCameraCapture();
};