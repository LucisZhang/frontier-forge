#include "frontier_forge/metrics.hpp"

#include <iomanip>
#include <sstream>
#include <string_view>

namespace frontier_forge {

namespace {

constexpr std::array<std::string_view,
                     static_cast<std::size_t>(MetricDecision::count)>
    kDecisionNames{"primary",
                   "fallback",
                   "queued",
                   "reject_overload",
                   "reject_deadline",
                   "reject_rate_limit",
                   "reject_unavailable",
                   "reject_bad_request"};

std::size_t route_index(UpstreamRoute route) {
  return route == UpstreamRoute::primary ? 0U : 1U;
}

int state_value(CircuitState state) {
  switch (state) {
  case CircuitState::closed:
    return 0;
  case CircuitState::half_open:
    return 1;
  case CircuitState::open:
    return 2;
  }
  return 2;
}

} // namespace

void GatewayMetrics::record_decision(MetricDecision decision) noexcept {
  decisions_[static_cast<std::size_t>(decision)].fetch_add(
      1, std::memory_order_relaxed);
}

void GatewayMetrics::record_response(unsigned status) noexcept {
  if (status >= 200 && status < 300) {
    responses_2xx_.fetch_add(1, std::memory_order_relaxed);
  } else if (status >= 400 && status < 500) {
    responses_4xx_.fetch_add(1, std::memory_order_relaxed);
  } else if (status >= 500) {
    responses_5xx_.fetch_add(1, std::memory_order_relaxed);
  }
}

void GatewayMetrics::record_upstream_failure(UpstreamRoute route) noexcept {
  upstream_failures_[route_index(route)].fetch_add(1,
                                                   std::memory_order_relaxed);
}

void GatewayMetrics::observe_latency(
    std::chrono::microseconds latency) noexcept {
  const auto micros =
      static_cast<std::uint64_t>(std::max<std::int64_t>(0, latency.count()));
  const double seconds = static_cast<double>(micros) / 1'000'000.0;
  for (std::size_t index = 0; index < latency_buckets_seconds_.size();
       ++index) {
    if (seconds <= latency_buckets_seconds_[index]) {
      latency_buckets_[index].fetch_add(1, std::memory_order_relaxed);
    }
  }
  latency_count_.fetch_add(1, std::memory_order_relaxed);
  latency_sum_microseconds_.fetch_add(micros, std::memory_order_relaxed);
}

void GatewayMetrics::set_primary_healthy(bool healthy) noexcept {
  primary_healthy_.store(healthy ? 1 : 0, std::memory_order_relaxed);
}

void GatewayMetrics::set_fallback_healthy(bool healthy) noexcept {
  fallback_healthy_.store(healthy ? 1 : 0, std::memory_order_relaxed);
}

void GatewayMetrics::set_primary_circuit(CircuitState state) noexcept {
  primary_circuit_.store(state_value(state), std::memory_order_relaxed);
}

void GatewayMetrics::set_fallback_circuit(CircuitState state) noexcept {
  fallback_circuit_.store(state_value(state), std::memory_order_relaxed);
}

std::string GatewayMetrics::render(const AdmissionSnapshot &admission) const {
  std::ostringstream out;
  out << "# HELP forge_gateway_queue_depth Requests waiting for token budget.\n"
      << "# TYPE forge_gateway_queue_depth gauge\n"
      << "forge_gateway_queue_depth " << admission.queue_requests << '\n'
      << "# HELP forge_gateway_queue_tokens Estimated tokens waiting in "
         "queue.\n"
      << "# TYPE forge_gateway_queue_tokens gauge\n"
      << "forge_gateway_queue_tokens " << admission.queue_tokens << '\n'
      << "# HELP forge_gateway_active_requests Requests holding admission "
         "leases.\n"
      << "# TYPE forge_gateway_active_requests gauge\n"
      << "forge_gateway_active_requests " << admission.active_requests << '\n'
      << "# HELP forge_gateway_active_tokens Estimated active tokens by "
         "route.\n"
      << "# TYPE forge_gateway_active_tokens gauge\n"
      << "forge_gateway_active_tokens{route=\"primary\"} "
      << admission.active_primary_tokens << '\n'
      << "forge_gateway_active_tokens{route=\"fallback\"} "
      << admission.active_fallback_tokens << '\n'
      << "# HELP forge_gateway_queue_high_watermark Maximum observed queue "
         "depth.\n"
      << "# TYPE forge_gateway_queue_high_watermark gauge\n"
      << "forge_gateway_queue_high_watermark "
      << admission.max_observed_queue_requests << '\n'
      << "# HELP forge_gateway_routing_decisions_total Admission decisions.\n"
      << "# TYPE forge_gateway_routing_decisions_total counter\n";
  for (std::size_t index = 0; index < decisions_.size(); ++index) {
    out << "forge_gateway_routing_decisions_total{decision=\""
        << kDecisionNames[index] << "\"} "
        << decisions_[index].load(std::memory_order_relaxed) << '\n';
  }
  out << "# HELP forge_gateway_responses_total Responses by status class.\n"
      << "# TYPE forge_gateway_responses_total counter\n"
      << "forge_gateway_responses_total{class=\"2xx\"} "
      << responses_2xx_.load(std::memory_order_relaxed) << '\n'
      << "forge_gateway_responses_total{class=\"4xx\"} "
      << responses_4xx_.load(std::memory_order_relaxed) << '\n'
      << "forge_gateway_responses_total{class=\"5xx\"} "
      << responses_5xx_.load(std::memory_order_relaxed) << '\n'
      << "# HELP forge_gateway_upstream_failures_total Transport and 5xx "
         "failures.\n"
      << "# TYPE forge_gateway_upstream_failures_total counter\n"
      << "forge_gateway_upstream_failures_total{route=\"primary\"} "
      << upstream_failures_[0].load(std::memory_order_relaxed) << '\n'
      << "forge_gateway_upstream_failures_total{route=\"fallback\"} "
      << upstream_failures_[1].load(std::memory_order_relaxed) << '\n'
      << "# HELP forge_gateway_replica_healthy Health-check state.\n"
      << "# TYPE forge_gateway_replica_healthy gauge\n"
      << "forge_gateway_replica_healthy{route=\"primary\"} "
      << primary_healthy_.load(std::memory_order_relaxed) << '\n'
      << "forge_gateway_replica_healthy{route=\"fallback\"} "
      << fallback_healthy_.load(std::memory_order_relaxed) << '\n'
      << "# HELP forge_gateway_circuit_state Circuit state: 0 closed, 1 "
         "half-open, 2 open.\n"
      << "# TYPE forge_gateway_circuit_state gauge\n"
      << "forge_gateway_circuit_state{route=\"primary\"} "
      << primary_circuit_.load(std::memory_order_relaxed) << '\n'
      << "forge_gateway_circuit_state{route=\"fallback\"} "
      << fallback_circuit_.load(std::memory_order_relaxed) << '\n'
      << "# HELP forge_gateway_request_duration_seconds End-to-end latency.\n"
      << "# TYPE forge_gateway_request_duration_seconds histogram\n";
  for (std::size_t index = 0; index < latency_buckets_seconds_.size();
       ++index) {
    out << "forge_gateway_request_duration_seconds_bucket{le=\"" << std::fixed
        << std::setprecision(3) << latency_buckets_seconds_[index] << "\"} "
        << latency_buckets_[index].load(std::memory_order_relaxed) << '\n';
  }
  const auto count = latency_count_.load(std::memory_order_relaxed);
  out << "forge_gateway_request_duration_seconds_bucket{le=\"+Inf\"} " << count
      << '\n'
      << "forge_gateway_request_duration_seconds_count " << count << '\n'
      << "forge_gateway_request_duration_seconds_sum " << std::fixed
      << std::setprecision(6)
      << static_cast<double>(
             latency_sum_microseconds_.load(std::memory_order_relaxed)) /
             1'000'000.0
      << '\n';
  return out.str();
}

} // namespace frontier_forge
