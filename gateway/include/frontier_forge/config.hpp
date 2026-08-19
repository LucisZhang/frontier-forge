#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>

#include "frontier_forge/admission.hpp"
#include "frontier_forge/circuit_breaker.hpp"
#include "frontier_forge/rate_limiter.hpp"
#include "frontier_forge/token_estimator.hpp"

namespace frontier_forge {

struct UpstreamConfig {
  std::string host{"127.0.0.1"};
  std::uint16_t port{8080};
  std::string model;
  std::string health_target{"/health"};
  std::size_t connection_pool_size{8};
};

struct GatewayConfig {
  std::string listen_host{"127.0.0.1"};
  std::uint16_t listen_port{9000};
  std::size_t io_threads{2};
  UpstreamConfig primary;
  bool fallback_enabled{};
  UpstreamConfig fallback{.host = "127.0.0.1", .port = 8081};
  AdmissionConfig admission;
  RateLimitConfig rate_limit;
  CircuitBreakerConfig circuit_breaker;
  TokenEstimatorConfig token_estimator;
  std::chrono::milliseconds default_request_timeout{30000};
  std::chrono::milliseconds maximum_request_timeout{120000};
  std::chrono::milliseconds connect_timeout{2000};
  std::chrono::milliseconds health_interval{1000};
  std::chrono::milliseconds health_timeout{500};
  std::chrono::milliseconds queue_poll_interval{5};
  std::size_t maximum_request_body_bytes{4 * 1024 * 1024};
  std::size_t maximum_response_body_bytes{64 * 1024 * 1024};
};

struct CommandLineResult {
  GatewayConfig config;
  bool show_help{};
};

[[nodiscard]] CommandLineResult parse_command_line(int argc, char **argv);
[[nodiscard]] std::string command_line_help();

} // namespace frontier_forge
