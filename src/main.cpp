#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

int main()
{
    const int device_id = 0;

    cv::VideoCapture cap(device_id, cv::CAP_V4L2);
    if (!cap.isOpened())
    {
        std::cerr << "ERROR: Cannot open camera device " << device_id << std::endl;
        return EXIT_FAILURE;
    }

    // Request MJPEG
    if (!cap.set(cv::CAP_PROP_FOURCC,
                 cv::VideoWriter::fourcc('M','J','P','G')))
    {
        std::cerr << "WARNING: Failed to set MJPEG format" << std::endl;
    }

    // Set resolution
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 720);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    // Set FPS (driver may ignore)
    cap.set(cv::CAP_PROP_FPS, 30);

    std::cout << "Camera initialized." << std::endl;

    cv::Mat frame;
    int frame_count = 0;
    auto start_time = std::chrono::steady_clock::now();

    while (true)
    {
        if (!cap.read(frame))
        {
            std::cerr << "ERROR: Failed to read frame from camera." << std::endl;
            break;
        }

        if (frame.empty())
        {
            std::cerr << "WARNING: Captured empty frame." << std::endl;
            continue;
        }

        frame_count++;


        if (frame_count == 30)
        {
            auto end_time = std::chrono::steady_clock::now();
            double seconds =
                std::chrono::duration<double>(end_time - start_time).count();

            std::cout << "Measured FPS: "
                      << frame_count / seconds << std::endl;

            frame_count = 0;
            start_time = std::chrono::steady_clock::now();
        }


        cv::imshow("Camera", frame);

        // Exit on ESC
        if (cv::waitKey(1) == 27)
            break;
    }

    cap.release();
    cv::destroyAllWindows();

    std::cout << "Camera released. Exiting." << std::endl;

    return EXIT_SUCCESS;
}

