#include <chrono>
#include <string>

class ScopeTimeLogger
{
public:
    ScopeTimeLogger(const std::string &name);

    ~ScopeTimeLogger();

private:
    std::string m_name;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
};