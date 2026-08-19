#pragma once

#include <chrono>
#include <cstddef>
#include <mutex>

#include "frontier_forge/admission.hpp"

namespace frontier_forge {

enum class CircuitState { closed, open, half_open };

[[nodiscard]] const char *to_string(CircuitState state) noexcept;

struct CircuitBreakerConfig {
  std::size_t failure_threshold{3};
  std::chrono::milliseconds recovery_timeout{2000};
};

struct CircuitSnapshot {
  CircuitState state{CircuitState::closed};
  std::size_t consecutive_failures{};
  bool probe_in_flight{};
  std::chrono::milliseconds retry_after{};
};

class CircuitBreaker {
public:
  explicit CircuitBreaker(CircuitBreakerConfig config);

  [[nodiscard]] bool
  peek_available(SteadyClock::time_point now = SteadyClock::now()) const;
  [[nodiscard]] bool allow(SteadyClock::time_point now = SteadyClock::now());
  void record_success();
  void record_failure(SteadyClock::time_point now = SteadyClock::now());
  [[nodiscard]] CircuitSnapshot
  snapshot(SteadyClock::time_point now = SteadyClock::now()) const;

private:
  CircuitBreakerConfig config_;
  mutable std::mutex mutex_;
  CircuitState state_{CircuitState::closed};
  std::size_t consecutive_failures_{};
  bool probe_in_flight_{};
  SteadyClock::time_point opened_at_{};
};

} // namespace frontier_forge
