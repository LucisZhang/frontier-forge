#include "frontier_forge/upstream_pool.hpp"

#include <algorithm>
#include <cerrno>
#include <exception>
#include <string>

#include <sys/socket.h>

#include <boost/asio/co_spawn.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/detached.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/redirect_error.hpp>
#include <boost/asio/steady_timer.hpp>
#include <boost/asio/use_awaitable.hpp>
#include <boost/beast/http.hpp>

namespace frontier_forge {
namespace net = boost::asio;
namespace beast = boost::beast;
namespace http = beast::http;
using tcp = net::ip::tcp;

namespace {

void close_connection_noexcept(
    const std::shared_ptr<UpstreamConnection> &connection) noexcept {
  boost::system::error_code ignored;
  connection->stream.socket().shutdown(tcp::socket::shutdown_both, ignored);
  connection->stream.socket().close(ignored);
  connection->buffer.consume(connection->buffer.size());
}

bool stale_idle_socket(tcp::socket &socket) noexcept {
  if (!socket.is_open()) {
    return false;
  }

  char byte{};
  while (true) {
    errno = 0;
    const auto received =
        ::recv(socket.native_handle(), &byte, sizeof(byte),
               MSG_PEEK | MSG_DONTWAIT);
    if (received >= 0) {
      // An orderly FIN returns zero. Any byte on an otherwise idle HTTP/1.1
      // connection is also unsafe to reuse because the previous response was
      // already parsed to completion.
      return true;
    }
    if (errno == EINTR) {
      continue;
    }
    return errno != EAGAIN && errno != EWOULDBLOCK;
  }
}

void cancel_timer_noexcept(
    const std::shared_ptr<net::steady_timer> &timer) noexcept {
  try {
    static_cast<void>(timer->cancel());
  } catch (const std::exception &) {
  }
}

net::awaitable<tcp::resolver::results_type>
resolve_with_deadline(net::any_io_executor executor, std::string host,
                      std::uint16_t port, SteadyClock::time_point deadline) {
  auto resolver = std::make_shared<tcp::resolver>(executor);
  auto timer = std::make_shared<net::steady_timer>(executor);
  timer->expires_at(deadline);
  net::co_spawn(
      executor,
      [resolver, timer]() -> net::awaitable<void> {
        boost::system::error_code error;
        co_await timer->async_wait(
            net::redirect_error(net::use_awaitable, error));
        if (!error) {
          resolver->cancel();
        }
      },
      net::detached);
  try {
    auto endpoints = co_await resolver->async_resolve(
        std::move(host), std::to_string(port), net::use_awaitable);
    cancel_timer_noexcept(timer);
    co_return endpoints;
  } catch (const boost::system::system_error &) {
    cancel_timer_noexcept(timer);
    if (SteadyClock::now() >= deadline) {
      throw boost::system::system_error(net::error::timed_out);
    }
    throw;
  } catch (...) {
    cancel_timer_noexcept(timer);
    throw;
  }
}

} // namespace

UpstreamPool::UpstreamPool(net::any_io_executor executor, UpstreamConfig config,
                           UpstreamRoute route,
                           CircuitBreakerConfig circuit_config,
                           std::chrono::milliseconds connect_timeout,
                           std::chrono::milliseconds health_interval,
                           std::chrono::milliseconds health_timeout,
                           GatewayMetrics &metrics)
    : executor_(std::move(executor)), config_(std::move(config)), route_(route),
      circuit_(circuit_config), connect_timeout_(connect_timeout),
      health_interval_(health_interval), health_timeout_(health_timeout),
      metrics_(metrics) {
  connections_.reserve(config_.connection_pool_size);
  for (std::size_t index = 0; index < config_.connection_pool_size; ++index) {
    connections_.push_back(std::make_shared<UpstreamConnection>(executor_));
  }
  update_health_metric(true);
  update_circuit_metric();
}

net::awaitable<std::shared_ptr<UpstreamConnection>>
UpstreamPool::acquire(SteadyClock::time_point deadline) {
  net::steady_timer timer(executor_);
  while (!stopping_.load(std::memory_order_relaxed)) {
    {
      std::scoped_lock lock(mutex_);
      const auto found = std::find_if(
          connections_.begin(), connections_.end(),
          [](const auto &connection) { return !connection->in_use; });
      if (found != connections_.end()) {
        // Point-in-time only: a FIN can still arrive after this probe and before
        // async_write. This narrows stale reuse; it does not eliminate the
        // transport-failure class.
        if (stale_idle_socket((*found)->stream.socket())) {
          close_connection_noexcept(*found);
        }
        (*found)->in_use = true;
        co_return *found;
      }
    }
    const auto now = SteadyClock::now();
    if (now >= deadline) {
      throw boost::system::system_error(net::error::timed_out);
    }
    timer.expires_at(std::min(deadline, now + std::chrono::milliseconds(2)));
    boost::system::error_code error;
    co_await timer.async_wait(net::redirect_error(net::use_awaitable, error));
    if (error && error != net::error::operation_aborted) {
      throw boost::system::system_error(error);
    }
  }
  throw boost::system::system_error(net::error::operation_aborted);
}

net::awaitable<void> UpstreamPool::ensure_connected(
    const std::shared_ptr<UpstreamConnection> &connection,
    SteadyClock::time_point deadline) {
  if (connection->stream.socket().is_open()) {
    co_return;
  }
  const auto effective_deadline =
      std::min(deadline, SteadyClock::now() + connect_timeout_);
  connection->stream.expires_at(effective_deadline);
  auto endpoints = co_await resolve_with_deadline(
      executor_, config_.host, config_.port, effective_deadline);
  co_await connection->stream.async_connect(endpoints, net::use_awaitable);
}

void UpstreamPool::release(
    const std::shared_ptr<UpstreamConnection> &connection, bool reusable) {
  if (!reusable) {
    close_connection_noexcept(connection);
  }
  std::scoped_lock lock(mutex_);
  connection->in_use = false;
}

void UpstreamPool::cancel(
    const std::shared_ptr<UpstreamConnection> &connection) {
  net::dispatch(executor_, [connection] {
    boost::system::error_code ignored;
    connection->stream.socket().cancel(ignored);
    connection->stream.socket().close(ignored);
  });
}

void UpstreamPool::start_health_checks() {
  auto self = shared_from_this();
  net::co_spawn(
      executor_,
      [self]() -> net::awaitable<void> { co_await self->health_loop(); },
      net::detached);
}

void UpstreamPool::stop() {
  stopping_.store(true, std::memory_order_relaxed);
  std::vector<std::shared_ptr<UpstreamConnection>> connections;
  {
    std::scoped_lock lock(mutex_);
    connections = connections_;
  }
  for (const auto &connection : connections) {
    cancel(connection);
  }
}

bool UpstreamPool::available(SteadyClock::time_point now) const {
  return healthy() && circuit_.peek_available(now);
}

bool UpstreamPool::allow_attempt(SteadyClock::time_point now) {
  const bool allowed = healthy() && circuit_.allow(now);
  update_circuit_metric();
  return allowed;
}

void UpstreamPool::record_success() {
  circuit_.record_success();
  update_circuit_metric();
}

void UpstreamPool::record_failure(SteadyClock::time_point now) {
  circuit_.record_failure(now);
  metrics_.record_upstream_failure(route_);
  update_circuit_metric();
}

std::size_t UpstreamPool::in_use_count() const {
  std::scoped_lock lock(mutex_);
  return static_cast<std::size_t>(
      std::count_if(connections_.begin(), connections_.end(),
                    [](const auto &connection) { return connection->in_use; }));
}

void UpstreamPool::set_healthy_for_test(bool healthy) {
  healthy_.store(healthy, std::memory_order_relaxed);
  update_health_metric(healthy);
}

net::awaitable<void> UpstreamPool::health_loop() {
  net::steady_timer timer(executor_);
  while (!stopping_.load(std::memory_order_relaxed)) {
    bool healthy = false;
    try {
      healthy = co_await check_health_once();
    } catch (const std::exception &) {
      healthy = false;
    }
    healthy_.store(healthy, std::memory_order_relaxed);
    update_health_metric(healthy);
    timer.expires_after(health_interval_);
    boost::system::error_code error;
    co_await timer.async_wait(net::redirect_error(net::use_awaitable, error));
    if (error == net::error::operation_aborted) {
      co_return;
    }
  }
}

net::awaitable<bool> UpstreamPool::check_health_once() {
  beast::tcp_stream stream(executor_);
  const auto deadline = SteadyClock::now() + health_timeout_;
  stream.expires_at(deadline);
  auto endpoints = co_await resolve_with_deadline(executor_, config_.host,
                                                  config_.port, deadline);
  co_await stream.async_connect(endpoints, net::use_awaitable);

  http::request<http::empty_body> request{http::verb::get,
                                          config_.health_target, 11};
  request.set(http::field::host,
              config_.host + ":" + std::to_string(config_.port));
  request.set(http::field::user_agent, "frontier-forge-health/0.5.0");
  request.keep_alive(false);
  co_await http::async_write(stream, request, net::use_awaitable);
  beast::flat_buffer buffer;
  http::response<http::string_body> response;
  co_await http::async_read(stream, buffer, response, net::use_awaitable);
  boost::system::error_code ignored;
  stream.socket().close(ignored);
  co_return response.result_int() >= 200 && response.result_int() < 300;
}

void UpstreamPool::update_health_metric(bool healthy) {
  if (route_ == UpstreamRoute::primary) {
    metrics_.set_primary_healthy(healthy);
  } else {
    metrics_.set_fallback_healthy(healthy);
  }
}

void UpstreamPool::update_circuit_metric() {
  const auto state = circuit_.snapshot().state;
  if (route_ == UpstreamRoute::primary) {
    metrics_.set_primary_circuit(state);
  } else {
    metrics_.set_fallback_circuit(state);
  }
}

} // namespace frontier_forge
