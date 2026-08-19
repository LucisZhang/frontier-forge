#include <gtest/gtest.h>

#include <chrono>

#include "frontier_forge/rate_limiter.hpp"

namespace frontier_forge {
namespace {

using namespace std::chrono_literals;

TEST(PerClientRateLimiterTest, EnforcesRequestAndTokenBucketsPerClient) {
  PerClientRateLimiter limiter({.requests_per_second = 2.0,
                                .request_burst = 2.0,
                                .tokens_per_second = 100.0,
                                .token_burst = 100.0,
                                .max_clients = 10,
                                .idle_ttl = 1min});
  const auto now = SteadyClock::time_point{};
  EXPECT_TRUE(limiter.allow("alice", 40, now).allowed);
  EXPECT_TRUE(limiter.allow("alice", 40, now).allowed);
  const auto denied = limiter.allow("alice", 40, now);
  EXPECT_FALSE(denied.allowed);
  EXPECT_GE(denied.retry_after, 500ms);
  EXPECT_TRUE(limiter.allow("bob", 100, now).allowed);
  EXPECT_TRUE(limiter.allow("alice", 40, now + 500ms).allowed);
}

TEST(PerClientRateLimiterTest, EvictsOldestClientAtCardinalityBound) {
  PerClientRateLimiter limiter({.requests_per_second = 1.0,
                                .request_burst = 1.0,
                                .tokens_per_second = 10.0,
                                .token_burst = 10.0,
                                .max_clients = 2,
                                .idle_ttl = 1min});
  const auto now = SteadyClock::time_point{};
  EXPECT_TRUE(limiter.allow("a", 1, now).allowed);
  EXPECT_TRUE(limiter.allow("b", 1, now).allowed);
  EXPECT_TRUE(limiter.allow("c", 1, now + 2min).allowed);
  EXPECT_LE(limiter.client_count(), 2U);
}

} // namespace
} // namespace frontier_forge
