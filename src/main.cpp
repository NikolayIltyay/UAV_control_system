#include <iostream>
#include <opencv2/opencv.hpp>
#include "fps_logger.hpp"
#include "camera_capture_factory.hpp"
#include "icamera_capture.hpp"
#include <onnxruntime_cxx_api.h>

constexpr int MODEL_WIDTH = 128;
constexpr int MODEL_HEIGHT = 128;

constexpr float SCORE_THRESHOLD = 0.4f;
constexpr float IOU_THRESHOLD = 0.3f;
constexpr int64_t MAX_DETECTIONS = 10;

int main(int argc, char **argv)
{
    // ---------------------------
    // 1. Initialize ONNX Runtime
    // ---------------------------
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "blazeface");
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    Ort::Session session(
        env,
        "models/End-to-end-BlazeFace-Onnx/mpipe_bface_boxes_ops16.onnx",
        session_options);

    std::cout << "Number of inputs: " << session.GetInputCount() << std::endl;

    Ort::AllocatorWithDefaultOptions allocator;

    // Get all input names
    auto input_name0 = session.GetInputNameAllocated(0, allocator);
    auto input_name1 = session.GetInputNameAllocated(1, allocator);
    auto input_name2 = session.GetInputNameAllocated(2, allocator);
    auto input_name3 = session.GetInputNameAllocated(3, allocator);

    auto output_name = session.GetOutputNameAllocated(0, allocator);

    const char *input_names[] = {
        input_name0.get(),
        input_name1.get(),
        input_name2.get(),
        input_name3.get()};

    for (size_t i = 0; i < session.GetInputCount(); ++i)
    {
        auto name = session.GetInputNameAllocated(i, allocator);
        auto type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();

        std::cout << "Input " << i << ": " << name.get() << " Shape: ";
        for (auto s : shape)
            std::cout << s << " ";
        std::cout << std::endl;
    }

    const char *output_names[] = {output_name.get()};

    const char *dev = (argc > 1) ? argv[1] : "/dev/video2";

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

        int orig_w = frame.cols;
        int orig_h = frame.rows;

        // ---------------------------
        // 2. Preprocess
        // ---------------------------
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(MODEL_WIDTH, MODEL_HEIGHT));
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
        resized.convertTo(resized, CV_32F, 1.0f / 255.0f);

        std::vector<int64_t> input_shape = {1, MODEL_HEIGHT, MODEL_WIDTH, 3};

        std::vector<float> input_tensor_values(resized.total() * 3);
        std::memcpy(input_tensor_values.data(),
                    resized.data,
                    input_tensor_values.size() * sizeof(float));

        Ort::MemoryInfo memory_info =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value image_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_tensor_values.data(),
            input_tensor_values.size(),
            input_shape.data(),
            input_shape.size());

        // ---------------------------
        // 3. Create scalar tensors
        // ---------------------------
        std::vector<int64_t> scalar_shape = {1};

        float score_threshold = SCORE_THRESHOLD;
        float iou_threshold = IOU_THRESHOLD;
        int64_t max_detections = MAX_DETECTIONS;

        Ort::Value score_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            &score_threshold,
            1,
            scalar_shape.data(),
            scalar_shape.size());

        Ort::Value iou_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            &iou_threshold,
            1,
            scalar_shape.data(),
            scalar_shape.size());

        Ort::Value max_det_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info,
            &max_detections,
            1,
            scalar_shape.data(),
            scalar_shape.size());

        Ort::Value input_tensors[] = {
            std::move(image_tensor),
            std::move(score_tensor),   // conf_threshold
            std::move(max_det_tensor), // max_detections
            std::move(iou_tensor)      // iou_threshold
        };

        Ort::RunOptions run_options;

        // ---------------------------
        // 4. Run Inference
        // ---------------------------
        auto output_tensors = session.Run(
            run_options,
            input_names,
            input_tensors,
            4,
            output_names,
            1);

        float *output_data =
            output_tensors[0].GetTensorMutableData<float>();

        // ---------------------------
        // 5. Draw detections (robust)
        // ---------------------------
        auto output_info = output_tensors[0].GetTensorTypeAndShapeInfo();
        std::vector<int64_t> output_shape = output_info.GetShape();

        int64_t batch = output_shape[0];
        int64_t num_detections = 0;
        int64_t elements_per_detection = 16;

        // Determine detection count
        if (output_shape.size() == 3)
        {
            // shape: [1, N, 16]
            num_detections = output_shape[1];
        }
        else if (output_shape.size() == 2)
        {
            // shape: [1, 16] OR [1, 0]
            if (output_shape[1] == 16)
                num_detections = 1;
            else
                num_detections = 0;
        }
        else
        {
            std::cout << "Unexpected output shape: ";
            for (auto s : output_shape)
                std::cout << s << " ";
            std::cout << std::endl;
            continue;
        }

        if (num_detections == 0)
        {
            // No faces detected
            cv::imshow("Face Detection", frame);
            continue;
        }

        for (int64_t i = 0; i < num_detections; ++i)
        {
            int64_t offset = i * elements_per_detection;

            float score = output_data[offset + 15];

            if (score < SCORE_THRESHOLD)
                continue;

            float y1 = output_data[offset + 0] * orig_h;
            float x1 = output_data[offset + 1] * orig_w;
            float y2 = output_data[offset + 2] * orig_h;
            float x2 = output_data[offset + 3] * orig_w;

            // Clamp coordinates (important!)
            x1 = std::max(0.0f, std::min((float)orig_w, x1));
            y1 = std::max(0.0f, std::min((float)orig_h, y1));
            x2 = std::max(0.0f, std::min((float)orig_w, x2));
            y2 = std::max(0.0f, std::min((float)orig_h, y2));

            cv::rectangle(frame,
                          cv::Point((int)x1, (int)y1),
                          cv::Point((int)x2, (int)y2),
                          cv::Scalar(0, 255, 0),
                          2);

            // Optional: show confidence
            char text[32];
            snprintf(text, sizeof(text), "%.2f", score);

            cv::putText(frame,
                        text,
                        cv::Point((int)x1, (int)y1 - 5),
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
