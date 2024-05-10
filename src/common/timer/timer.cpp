#include "timer.h"

// Timer implementations
// Author: Will Morris
// Credit: https://www.learncpp.com/cpp-tutorial/timing-your-code/
namespace carver {

void Timer::reset() {
    beginning = Clock::now();
}

double Timer::elapsed() const {
    return std::chrono::duration_cast<Second>(Clock::now() - beginning).count();
}

} // carver
