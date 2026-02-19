#pragma once

#include <chrono>

class FpsLogger {
public:
    explicit FpsLogger(double logIntervalSeconds = 1.0);

    void update();

private:
    double _logIntervalSeconds;
    
    size_t _frames{0}; 
    std::chrono::time_point<std::chrono::steady_clock> _lastLogTime;
};