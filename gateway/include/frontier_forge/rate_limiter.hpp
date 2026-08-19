#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>

#include "frontier_forge/admission.hpp"

namespace frontier_forge {

struct RateLimitConfig {
  double requests_per_second{20.0};
  double request_burst{40.0};
  double tokens_per_second{32768.0};
  double token_burst{65536.0};
  std::size_t max_clients{10000};
  std::chrono::minutes idle_ttl{10};
};

struct RateLimitResult {
  bool allowed{};
  std::chrono::milliseconds retry_after{};
};

class PerClientRateLimiter {
public:
  explicit PerClientRateLimiter(RateLimitConfig config);

  [[nodiscard]] RateLimitResult
  allow(std::string_view client_id, std::uint64_t estimated_tokens,
        SteadyClock::time_point now = SteadyClock::now());
  [[nodiscard]] std::size_t client_count() const;

private:
  struct Bucket {
    double request_tokens{};
    double work_tokens{};
    SteadyClock::time_point updated_at{};
    SteadyClock::time_point last_seen{};
  };

  void refill_locked(Bucket &bucket, SteadyClock::time_point now) const;
  void prune_locked(SteadyClock::time_point now);

  RateLimitConfig config_;
  mutable std::mutex mutex_;
  std::unordered_map<std::string, Bucket> clients_;
};

} // namespace frontier_forge
