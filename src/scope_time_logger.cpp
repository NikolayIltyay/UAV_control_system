#include "scope_time_logger.hpp"
#include <iostream>

ScopeTimeLogger::ScopeTimeLogger(const std::string &name)
    : m_name(name), m_start(std::chrono::high_resolution_clock::now())
{
}

ScopeTimeLogger::~ScopeTimeLogger()
{
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - m_start);

    std::cout << "[Timer] " << m_name << " took "
              << duration.count() << "ms" << std::endl;
}