#include "fps_logger.hpp"
#include <iostream>

FpsLogger::FpsLogger(double logIntervalSeconds)
    : _logIntervalSeconds{logIntervalSeconds}, 
      _lastLogTime{std::chrono::steady_clock::now()} 
{
}

void FpsLogger::update() 
{
    _frames++;
    auto currentTime = std::chrono::steady_clock::now();
    double secondsElapsed = std::chrono::duration<double>(currentTime - _lastLogTime).count();

    if (secondsElapsed >= _logIntervalSeconds) 
    {
        std::cout << "FPS: " << static_cast<double>(_frames) / secondsElapsed << std::endl;
        
        _frames = 0;
        _lastLogTime = currentTime; 
    }
}