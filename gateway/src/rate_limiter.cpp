#include "frontier_forge/rate_limiter.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace frontier_forge {

PerClientRateLimiter::PerClientRateLimiter(RateLimitConfig config)
    : config_(std::move(config)) {
  if (config_.requests_per_second <= 0.0 || config_.request_burst < 1.0 ||
      config_.tokens_per_second <= 0.0 || config_.token_burst < 1.0 ||
      config_.max_clients == 0) {
    throw std::invalid_argument("rate-limit values must be positive");
  }
}

RateLimitResult PerClientRateLimiter::allow(std::string_view client_id,
                                            std::uint64_t estimated_tokens,
                                            SteadyClock::time_point now) {
  std::scoped_lock lock(mutex_);
  prune_locked(now);
  const std::string key =
      client_id.empty() ? "anonymous" : std::string(client_id);
  auto [it, inserted] =
      clients_.try_emplace(key, Bucket{.request_tokens = config_.request_burst,
                                       .work_tokens = config_.token_burst,
                                       .updated_at = now,
                                       .last_seen = now});
  auto &bucket = it->second;
  if (!inserted) {
    refill_locked(bucket, now);
    bucket.last_seen = now;
  }

  const double work = static_cast<double>(estimated_tokens);
  if (bucket.request_tokens >= 1.0 && bucket.work_tokens >= work) {
    bucket.request_tokens -= 1.0;
    bucket.work_tokens -= work;
    return {.allowed = true};
  }

  const double request_wait =
      bucket.request_tokens >= 1.0
          ? 0.0
          : (1.0 - bucket.request_tokens) / config_.requests_per_second;
  const double token_wait =
      bucket.work_tokens >= work
          ? 0.0
          : (work - bucket.work_tokens) / config_.tokens_per_second;
  const double seconds = std::max(request_wait, token_wait);
  const auto max_ms =
      static_cast<double>(std::numeric_limits<std::int64_t>::max());
  return {.allowed = false,
          .retry_after = std::chrono::milliseconds(static_cast<std::int64_t>(
              std::ceil(std::min(max_ms, seconds * 1000.0))))};
}

std::size_t PerClientRateLimiter::client_count() const {
  std::scoped_lock lock(mutex_);
  return clients_.size();
}

void PerClientRateLimiter::refill_locked(Bucket &bucket,
                                         SteadyClock::time_point now) const {
  const auto elapsed = std::chrono::duration<double>(
                           std::max(now, bucket.updated_at) - bucket.updated_at)
                           .count();
  bucket.request_tokens =
      std::min(config_.request_burst,
               bucket.request_tokens + elapsed * config_.requests_per_second);
  bucket.work_tokens =
      std::min(config_.token_burst,
               bucket.work_tokens + elapsed * config_.tokens_per_second);
  bucket.updated_at = now;
}

void PerClientRateLimiter::prune_locked(SteadyClock::time_point now) {
  if (clients_.size() < config_.max_clients) {
    return;
  }
  for (auto it = clients_.begin(); it != clients_.end();) {
    if (now - it->second.last_seen >= config_.idle_ttl) {
      it = clients_.erase(it);
    } else {
      ++it;
    }
  }
  while (clients_.size() >= config_.max_clients) {
    auto oldest = std::min_element(
        clients_.begin(), clients_.end(), [](const auto &lhs, const auto &rhs) {
          return lhs.second.last_seen < rhs.second.last_seen;
        });
    if (oldest == clients_.end()) {
      break;
    }
    clients_.erase(oldest);
  }
}

} // namespace frontier_forge
