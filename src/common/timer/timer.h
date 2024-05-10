#ifndef HPCARVER_TIMER_H
#define HPCARVER_TIMER_H

#include <chrono>

namespace carver {

/**
 * Timing module for tests.
 * Credit: https://www.learncpp.com/cpp-tutorial/timing-your-code/
 *
 * @author Will Morris
 */
class Timer {

private:
    using Clock = std::chrono::steady_clock;
    using Second = std::chrono::duration<double, std::ratio<1> >;

    std::chrono::time_point<Clock> beginning { Clock::now() };

public:
    /**
     * Reset this timer.
     */
    void reset();

    /**
     * Return elapsed time on this timer.
     */
     double elapsed() const;
};

}

#endif //HPCARVER_TIMER_H
