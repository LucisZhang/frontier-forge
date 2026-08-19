#include "frontier_forge/circuit_breaker.hpp"

#include <algorithm>
#include <stdexcept>

namespace frontier_forge {

const char *to_string(CircuitState state) noexcept {
  switch (state) {
  case CircuitState::closed:
    return "closed";
  case CircuitState::open:
    return "open";
  case CircuitState::half_open:
    return "half_open";
  }
  return "unknown";
}

CircuitBreaker::CircuitBreaker(CircuitBreakerConfig config)
    : config_(std::move(config)) {
  if (config_.failure_threshold == 0 ||
      config_.recovery_timeout <= std::chrono::milliseconds::zero()) {
    throw std::invalid_argument("circuit-breaker values must be positive");
  }
}

bool CircuitBreaker::peek_available(SteadyClock::time_point now) const {
  std::scoped_lock lock(mutex_);
  if (state_ == CircuitState::closed) {
    return true;
  }
  if (state_ == CircuitState::half_open) {
    return !probe_in_flight_;
  }
  return now - opened_at_ >= config_.recovery_timeout;
}

bool CircuitBreaker::allow(SteadyClock::time_point now) {
  std::scoped_lock lock(mutex_);
  if (state_ == CircuitState::closed) {
    return true;
  }
  if (state_ == CircuitState::open) {
    if (now - opened_at_ < config_.recovery_timeout) {
      return false;
    }
    state_ = CircuitState::half_open;
    probe_in_flight_ = true;
    return true;
  }
  if (probe_in_flight_) {
    return false;
  }
  probe_in_flight_ = true;
  return true;
}

void CircuitBreaker::record_success() {
  std::scoped_lock lock(mutex_);
  state_ = CircuitState::closed;
  consecutive_failures_ = 0;
  probe_in_flight_ = false;
}

void CircuitBreaker::record_failure(SteadyClock::time_point now) {
  std::scoped_lock lock(mutex_);
  probe_in_flight_ = false;
  if (state_ == CircuitState::half_open) {
    state_ = CircuitState::open;
    opened_at_ = now;
    consecutive_failures_ = config_.failure_threshold;
    return;
  }
  if (state_ == CircuitState::open) {
    opened_at_ = now;
    return;
  }
  ++consecutive_failures_;
  if (consecutive_failures_ >= config_.failure_threshold) {
    state_ = CircuitState::open;
    opened_at_ = now;
  }
}

CircuitSnapshot CircuitBreaker::snapshot(SteadyClock::time_point now) const {
  std::scoped_lock lock(mutex_);
  std::chrono::milliseconds retry_after{};
  if (state_ == CircuitState::open &&
      now - opened_at_ < config_.recovery_timeout) {
    retry_after = std::chrono::duration_cast<std::chrono::milliseconds>(
        config_.recovery_timeout - (now - opened_at_));
  }
  return {.state = state_,
          .consecutive_failures = consecutive_failures_,
          .probe_in_flight = probe_in_flight_,
          .retry_after = retry_after};
}

} // namespace frontier_forge
