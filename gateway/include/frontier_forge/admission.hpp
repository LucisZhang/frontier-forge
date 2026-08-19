#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

namespace frontier_forge {

using SteadyClock = std::chrono::steady_clock;

enum class UpstreamRoute : std::uint8_t { primary, fallback };

[[nodiscard]] const char *to_string(UpstreamRoute route) noexcept;

struct RouteAvailability {
  bool primary{true};
  bool fallback{false};
};

struct RequestCost {
  std::uint64_t prompt_tokens{};
  std::uint64_t output_tokens{};

  [[nodiscard]] std::uint64_t total_tokens() const noexcept;
};

enum class AdmissionKind : std::uint8_t {
  admitted,
  queued,
  rejected_overload,
  rejected_deadline,
  rejected_oversize,
  rejected_unavailable,
  cancelled,
};

[[nodiscard]] const char *to_string(AdmissionKind kind) noexcept;

struct AdmissionLease {
  std::uint64_t id{};
  UpstreamRoute route{UpstreamRoute::primary};
  std::uint64_t tokens{};
};

struct AdmissionResult {
  AdmissionKind kind{AdmissionKind::rejected_overload};
  std::optional<AdmissionLease> lease;
  std::optional<std::uint64_t> ticket;
  std::chrono::milliseconds retry_after{};
  std::string reason;
};

struct AdmissionSnapshot {
  std::size_t active_requests{};
  std::uint64_t active_primary_tokens{};
  std::uint64_t active_fallback_tokens{};
  std::size_t queue_requests{};
  std::uint64_t queue_tokens{};
  std::size_t max_observed_queue_requests{};
  std::uint64_t max_observed_queue_tokens{};
  std::chrono::milliseconds service_time_ewma{};
};

struct AdmissionConfig {
  std::uint64_t primary_token_capacity{32768};
  std::uint64_t fallback_token_capacity{16384};
  std::size_t max_queue_requests{64};
  std::uint64_t max_queue_tokens{131072};
  double degrade_utilization{0.80};
  std::chrono::milliseconds initial_service_time{1000};
  std::chrono::milliseconds minimum_execution_budget{50};
  std::chrono::milliseconds minimum_retry_after{100};
};

class AdmissionController {
public:
  explicit AdmissionController(AdmissionConfig config);

  [[nodiscard]] AdmissionResult
  admit(RequestCost cost, SteadyClock::time_point deadline, bool allow_degrade,
        RouteAvailability availability,
        SteadyClock::time_point now = SteadyClock::now());

  [[nodiscard]] AdmissionResult
  poll(std::uint64_t ticket, RouteAvailability availability,
       SteadyClock::time_point now = SteadyClock::now());

  [[nodiscard]] bool cancel(std::uint64_t ticket);
  void complete(const AdmissionLease &lease,
                std::chrono::milliseconds service_time = {});

  [[nodiscard]] AdmissionSnapshot snapshot() const;
  [[nodiscard]] const AdmissionConfig &config() const noexcept {
    return config_;
  }

private:
  struct QueueEntry {
    std::uint64_t id{};
    std::uint64_t tokens{};
    SteadyClock::time_point deadline{};
    bool allow_degrade{};
  };

  [[nodiscard]] AdmissionLease make_lease_locked(UpstreamRoute route,
                                                 std::uint64_t tokens);
  void promote_locked(RouteAvailability availability,
                      SteadyClock::time_point now);
  void expire_locked(SteadyClock::time_point now);
  [[nodiscard]] std::chrono::milliseconds
  predicted_wait_locked(std::uint64_t additional_tokens) const;
  [[nodiscard]] std::chrono::milliseconds retry_after_locked() const;
  [[nodiscard]] bool route_fits_locked(UpstreamRoute route,
                                       std::uint64_t tokens) const;

  AdmissionConfig config_;
  mutable std::mutex mutex_;
  std::uint64_t next_id_{1};
  std::uint64_t active_primary_tokens_{};
  std::uint64_t active_fallback_tokens_{};
  std::deque<QueueEntry> queue_;
  std::uint64_t queue_tokens_{};
  std::unordered_map<std::uint64_t, AdmissionLease> active_;
  std::unordered_map<std::uint64_t, AdmissionLease> ready_;
  std::unordered_map<std::uint64_t, AdmissionResult> terminal_;
  std::size_t max_observed_queue_requests_{};
  std::uint64_t max_observed_queue_tokens_{};
  double service_time_ewma_ms_{};
};

} // namespace frontier_forge
