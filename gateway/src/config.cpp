#include "frontier_forge/config.hpp"

#include <charconv>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace frontier_forge {

namespace {

template <typename Integer>
Integer parse_integer(std::string_view name, std::string_view value) {
  Integer parsed{};
  const auto [end, error] =
      std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (error != std::errc{} || end != value.data() + value.size()) {
    throw std::invalid_argument(std::string(name) + " expects an integer");
  }
  return parsed;
}

double parse_double(std::string_view name, std::string_view value) {
  std::size_t consumed = 0;
  const auto parsed = std::stod(std::string(value), &consumed);
  if (consumed != value.size() || !std::isfinite(parsed)) {
    throw std::invalid_argument(std::string(name) + " expects a number");
  }
  return parsed;
}

std::string_view require_value(int argc, char **argv, int &index,
                               std::string_view option) {
  if (index + 1 >= argc) {
    throw std::invalid_argument(std::string(option) + " requires a value");
  }
  ++index;
  return argv[index];
}

} // namespace

CommandLineResult parse_command_line(int argc, char **argv) {
  CommandLineResult result;
  for (int index = 1; index < argc; ++index) {
    const std::string_view option = argv[index];
    if (option == "--help" || option == "-h") {
      result.show_help = true;
      continue;
    }
    if (option == "--listen-host") {
      result.config.listen_host = require_value(argc, argv, index, option);
    } else if (option == "--listen-port") {
      result.config.listen_port = parse_integer<std::uint16_t>(
          option, require_value(argc, argv, index, option));
    } else if (option == "--io-threads") {
      result.config.io_threads = parse_integer<std::size_t>(
          option, require_value(argc, argv, index, option));
    } else if (option == "--primary-host") {
      result.config.primary.host = require_value(argc, argv, index, option);
    } else if (option == "--primary-port") {
      result.config.primary.port = parse_integer<std::uint16_t>(
          option, require_value(argc, argv, index, option));
    } else if (option == "--primary-model") {
      result.config.primary.model = require_value(argc, argv, index, option);
    } else if (option == "--primary-health-path") {
      result.config.primary.health_target =
          require_value(argc, argv, index, option);
    } else if (option == "--fallback-host") {
      result.config.fallback_enabled = true;
      result.config.fallback.host = require_value(argc, argv, index, option);
    } else if (option == "--fallback-port") {
      result.config.fallback_enabled = true;
      result.config.fallback.port = parse_integer<std::uint16_t>(
          option, require_value(argc, argv, index, option));
    } else if (option == "--fallback-model") {
      result.config.fallback_enabled = true;
      result.config.fallback.model = require_value(argc, argv, index, option);
    } else if (option == "--fallback-health-path") {
      result.config.fallback_enabled = true;
      result.config.fallback.health_target =
          require_value(argc, argv, index, option);
    } else if (option == "--disable-fallback") {
      result.config.fallback_enabled = false;
    } else if (option == "--pool-size") {
      const auto size = parse_integer<std::size_t>(
          option, require_value(argc, argv, index, option));
      result.config.primary.connection_pool_size = size;
      result.config.fallback.connection_pool_size = size;
    } else if (option == "--primary-token-capacity") {
      result.config.admission.primary_token_capacity =
          parse_integer<std::uint64_t>(
              option, require_value(argc, argv, index, option));
    } else if (option == "--fallback-token-capacity") {
      result.config.admission.fallback_token_capacity =
          parse_integer<std::uint64_t>(
              option, require_value(argc, argv, index, option));
    } else if (option == "--max-queue-requests") {
      result.config.admission.max_queue_requests = parse_integer<std::size_t>(
          option, require_value(argc, argv, index, option));
    } else if (option == "--max-queue-tokens") {
      result.config.admission.max_queue_tokens = parse_integer<std::uint64_t>(
          option, require_value(argc, argv, index, option));
    } else if (option == "--degrade-utilization") {
      result.config.admission.degrade_utilization =
          parse_double(option, require_value(argc, argv, index, option));
    } else if (option == "--request-timeout-ms") {
      result.config.default_request_timeout =
          std::chrono::milliseconds(parse_integer<std::int64_t>(
              option, require_value(argc, argv, index, option)));
    } else if (option == "--health-interval-ms") {
      result.config.health_interval =
          std::chrono::milliseconds(parse_integer<std::int64_t>(
              option, require_value(argc, argv, index, option)));
    } else if (option == "--rate-requests-per-second") {
      result.config.rate_limit.requests_per_second =
          parse_double(option, require_value(argc, argv, index, option));
    } else if (option == "--rate-request-burst") {
      result.config.rate_limit.request_burst =
          parse_double(option, require_value(argc, argv, index, option));
    } else if (option == "--rate-tokens-per-second") {
      result.config.rate_limit.tokens_per_second =
          parse_double(option, require_value(argc, argv, index, option));
    } else if (option == "--rate-token-burst") {
      result.config.rate_limit.token_burst =
          parse_double(option, require_value(argc, argv, index, option));
    } else if (option == "--circuit-failures") {
      result.config.circuit_breaker.failure_threshold =
          parse_integer<std::size_t>(option,
                                     require_value(argc, argv, index, option));
    } else if (option == "--circuit-recovery-ms") {
      result.config.circuit_breaker.recovery_timeout =
          std::chrono::milliseconds(parse_integer<std::int64_t>(
              option, require_value(argc, argv, index, option)));
    } else if (option == "--default-output-tokens") {
      result.config.token_estimator.default_output_tokens =
          parse_integer<std::uint64_t>(
              option, require_value(argc, argv, index, option));
    } else if (option == "--maximum-output-tokens") {
      result.config.token_estimator.maximum_output_tokens =
          parse_integer<std::uint64_t>(
              option, require_value(argc, argv, index, option));
    } else {
      throw std::invalid_argument("unknown option: " + std::string(option));
    }
  }
  if (result.config.io_threads == 0 ||
      result.config.primary.connection_pool_size == 0 ||
      result.config.fallback.connection_pool_size == 0) {
    throw std::invalid_argument("thread and pool sizes must be positive");
  }
  if (result.config.default_request_timeout <=
          std::chrono::milliseconds::zero() ||
      result.config.health_interval <= std::chrono::milliseconds::zero()) {
    throw std::invalid_argument(
        "timeouts and health intervals must be positive");
  }
  return result;
}

std::string command_line_help() {
  return R"(frontier-forge C++20 LLM-aware gateway

Usage: forge_gateway [options]

  --listen-host HOST                 default 127.0.0.1
  --listen-port PORT                 default 9000
  --io-threads N                     default 2
  --primary-host HOST                default 127.0.0.1
  --primary-port PORT                default 8080
  --primary-model MODEL              optional request model override
  --primary-health-path PATH         default /health
  --fallback-host HOST               enables fallback route
  --fallback-port PORT               enables fallback route
  --fallback-model MODEL             enables fallback and rewrites model
  --fallback-health-path PATH        default /health
  --disable-fallback
  --pool-size N                      connections per route
  --primary-token-capacity TOKENS
  --fallback-token-capacity TOKENS
  --max-queue-requests N
  --max-queue-tokens TOKENS
  --degrade-utilization FRACTION
  --request-timeout-ms MS
  --health-interval-ms MS
  --rate-requests-per-second N
  --rate-request-burst N
  --rate-tokens-per-second N
  --rate-token-burst N
  --circuit-failures N
  --circuit-recovery-ms MS
  --default-output-tokens N
  --maximum-output-tokens N
)";
}

} // namespace frontier_forge
