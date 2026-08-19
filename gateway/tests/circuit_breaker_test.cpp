#include <gtest/gtest.h>

#include <chrono>

#include "frontier_forge/circuit_breaker.hpp"

namespace frontier_forge {
namespace {

using namespace std::chrono_literals;

TEST(CircuitBreakerTest, OpensAfterThresholdAndClosesAfterSuccessfulProbe) {
  CircuitBreaker breaker({.failure_threshold = 2, .recovery_timeout = 100ms});
  const auto now = SteadyClock::time_point{};
  EXPECT_TRUE(breaker.allow(now));
  breaker.record_failure(now);
  EXPECT_TRUE(breaker.allow(now));
  breaker.record_failure(now);
  EXPECT_EQ(breaker.snapshot(now).state, CircuitState::open);
  EXPECT_FALSE(breaker.allow(now + 99ms));
  EXPECT_TRUE(breaker.allow(now + 100ms));
  EXPECT_EQ(breaker.snapshot(now + 100ms).state, CircuitState::half_open);
  EXPECT_FALSE(breaker.allow(now + 100ms));
  breaker.record_success();
  EXPECT_EQ(breaker.snapshot(now + 100ms).state, CircuitState::closed);
}

TEST(CircuitBreakerTest, FailedProbeReopensCircuit) {
  CircuitBreaker breaker({.failure_threshold = 1, .recovery_timeout = 50ms});
  const auto now = SteadyClock::time_point{};
  breaker.record_failure(now);
  ASSERT_TRUE(breaker.allow(now + 50ms));
  breaker.record_failure(now + 50ms);
  EXPECT_EQ(breaker.snapshot(now + 50ms).state, CircuitState::open);
  EXPECT_FALSE(breaker.peek_available(now + 90ms));
  EXPECT_TRUE(breaker.peek_available(now + 100ms));
}

} // namespace
} // namespace frontier_forge
