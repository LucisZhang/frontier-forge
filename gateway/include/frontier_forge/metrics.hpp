#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>

#include "frontier_forge/admission.hpp"
#include "frontier_forge/circuit_breaker.hpp"

namespace frontier_forge {

enum class MetricDecision : std::size_t {
  primary,
  fallback,
  queued,
  reject_overload,
  reject_deadline,
  reject_rate_limit,
  reject_unavailable,
  reject_bad_request,
  count,
};

class GatewayMetrics {
public:
  void record_decision(MetricDecision decision) noexcept;
  void record_response(unsigned status) noexcept;
  void record_upstream_failure(UpstreamRoute route) noexcept;
  void observe_latency(std::chrono::microseconds latency) noexcept;

  void set_primary_healthy(bool healthy) noexcept;
  void set_fallback_healthy(bool healthy) noexcept;
  void set_primary_circuit(CircuitState state) noexcept;
  void set_fallback_circuit(CircuitState state) noexcept;

  [[nodiscard]] std::string render(const AdmissionSnapshot &admission) const;

private:
  static constexpr std::array<double, 10> latency_buckets_seconds_{
      0.005, 0.010, 0.025, 0.050, 0.100, 0.250, 0.500, 1.000, 2.500, 5.000};

  std::array<std::atomic<std::uint64_t>,
             static_cast<std::size_t>(MetricDecision::count)>
      decisions_{};
  std::atomic<std::uint64_t> responses_2xx_{};
  std::atomic<std::uint64_t> responses_4xx_{};
  std::atomic<std::uint64_t> responses_5xx_{};
  std::array<std::atomic<std::uint64_t>, 2> upstream_failures_{};
  std::array<std::atomic<std::uint64_t>, latency_buckets_seconds_.size()>
      latency_buckets_{};
  std::atomic<std::uint64_t> latency_count_{};
  std::atomic<std::uint64_t> latency_sum_microseconds_{};
  std::atomic<int> primary_healthy_{1};
  std::atomic<int> fallback_healthy_{0};
  std::atomic<int> primary_circuit_{};
  std::atomic<int> fallback_circuit_{};
};

} // namespace frontier_forge
