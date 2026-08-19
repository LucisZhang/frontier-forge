#include <gtest/gtest.h>

#include <chrono>
#include <string>

#include "frontier_forge/metrics.hpp"

namespace frontier_forge {
namespace {

TEST(GatewayMetricsTest, RendersRequiredPrometheusFamilies) {
  GatewayMetrics metrics;
  metrics.record_decision(MetricDecision::queued);
  metrics.record_decision(MetricDecision::fallback);
  metrics.record_response(200);
  metrics.record_response(503);
  metrics.record_upstream_failure(UpstreamRoute::primary);
  metrics.observe_latency(std::chrono::milliseconds(25));
  metrics.set_primary_healthy(false);
  metrics.set_primary_circuit(CircuitState::open);
  const auto text = metrics.render({.active_requests = 2,
                                    .active_primary_tokens = 100,
                                    .active_fallback_tokens = 20,
                                    .queue_requests = 3,
                                    .queue_tokens = 90,
                                    .max_observed_queue_requests = 4,
                                    .max_observed_queue_tokens = 120});
  EXPECT_NE(text.find("forge_gateway_queue_depth 3"), std::string::npos);
  EXPECT_NE(text.find("decision=\"fallback\"} 1"), std::string::npos);
  EXPECT_NE(text.find("route=\"primary\"} 1"), std::string::npos);
  EXPECT_NE(text.find("class=\"5xx\"} 1"), std::string::npos);
  EXPECT_NE(text.find("request_duration_seconds_bucket"), std::string::npos);
}

} // namespace
} // namespace frontier_forge
