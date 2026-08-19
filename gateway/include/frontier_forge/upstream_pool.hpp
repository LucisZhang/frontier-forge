#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

#include <boost/asio/any_io_executor.hpp>
#include <boost/asio/awaitable.hpp>
#include <boost/beast/core/flat_buffer.hpp>
#include <boost/beast/core/tcp_stream.hpp>

#include "frontier_forge/admission.hpp"
#include "frontier_forge/circuit_breaker.hpp"
#include "frontier_forge/config.hpp"
#include "frontier_forge/metrics.hpp"

namespace frontier_forge {

struct UpstreamConnection {
  explicit UpstreamConnection(boost::asio::any_io_executor executor)
      : stream(std::move(executor)) {}

  boost::beast::tcp_stream stream;
  boost::beast::flat_buffer buffer;
  bool in_use{};
};

class UpstreamPool : public std::enable_shared_from_this<UpstreamPool> {
public:
  UpstreamPool(boost::asio::any_io_executor executor, UpstreamConfig config,
               UpstreamRoute route, CircuitBreakerConfig circuit_config,
               std::chrono::milliseconds connect_timeout,
               std::chrono::milliseconds health_interval,
               std::chrono::milliseconds health_timeout,
               GatewayMetrics &metrics);

  [[nodiscard]] boost::asio::awaitable<std::shared_ptr<UpstreamConnection>>
  acquire(SteadyClock::time_point deadline);
  boost::asio::awaitable<void>
  ensure_connected(const std::shared_ptr<UpstreamConnection> &connection,
                   SteadyClock::time_point deadline);
  void release(const std::shared_ptr<UpstreamConnection> &connection,
               bool reusable);
  void cancel(const std::shared_ptr<UpstreamConnection> &connection);

  void start_health_checks();
  void stop();

  [[nodiscard]] bool
  available(SteadyClock::time_point now = SteadyClock::now()) const;
  [[nodiscard]] bool
  allow_attempt(SteadyClock::time_point now = SteadyClock::now());
  void record_success();
  void record_failure(SteadyClock::time_point now = SteadyClock::now());

  [[nodiscard]] bool healthy() const noexcept {
    return healthy_.load(std::memory_order_relaxed);
  }
  [[nodiscard]] CircuitSnapshot circuit_snapshot() const {
    return circuit_.snapshot();
  }
  [[nodiscard]] const UpstreamConfig &config() const noexcept {
    return config_;
  }
  [[nodiscard]] UpstreamRoute route() const noexcept { return route_; }
  [[nodiscard]] std::size_t in_use_count() const;

  void set_healthy_for_test(bool healthy);

private:
  boost::asio::awaitable<void> health_loop();
  boost::asio::awaitable<bool> check_health_once();
  void update_health_metric(bool healthy);
  void update_circuit_metric();

  boost::asio::any_io_executor executor_;
  UpstreamConfig config_;
  UpstreamRoute route_;
  CircuitBreaker circuit_;
  std::chrono::milliseconds connect_timeout_;
  std::chrono::milliseconds health_interval_;
  std::chrono::milliseconds health_timeout_;
  GatewayMetrics &metrics_;
  std::atomic<bool> healthy_{true};
  std::atomic<bool> stopping_{};
  mutable std::mutex mutex_;
  std::vector<std::shared_ptr<UpstreamConnection>> connections_;
};

} // namespace frontier_forge
