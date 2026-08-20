#include "frontier_forge/gateway_server.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include <boost/asio/co_spawn.hpp>
#include <boost/asio/detached.hpp>
#include <boost/asio/dispatch.hpp>
#include <boost/asio/redirect_error.hpp>
#include <boost/asio/steady_timer.hpp>
#include <boost/asio/strand.hpp>
#include <boost/asio/use_awaitable.hpp>
#include <boost/beast/http/chunk_encode.hpp>
#include <boost/json.hpp>

namespace frontier_forge {
namespace net = boost::asio;
namespace beast = boost::beast;
namespace http = beast::http;
namespace json = boost::json;
using tcp = net::ip::tcp;
using namespace std::chrono_literals;

namespace {

class LatencyGuard {
public:
  explicit LatencyGuard(GatewayMetrics &metrics)
      : metrics_(metrics), started_at_(SteadyClock::now()) {}
  ~LatencyGuard() {
    metrics_.observe_latency(
        std::chrono::duration_cast<std::chrono::microseconds>(
            SteadyClock::now() - started_at_));
  }

private:
  GatewayMetrics &metrics_;
  SteadyClock::time_point started_at_;
};

class AdmissionGuard {
public:
  AdmissionGuard(AdmissionController &admission, AdmissionLease lease)
      : admission_(admission), lease_(lease), started_at_(SteadyClock::now()) {}
  ~AdmissionGuard() {
    admission_.complete(lease_,
                        std::chrono::duration_cast<std::chrono::milliseconds>(
                            SteadyClock::now() - started_at_));
  }

private:
  AdmissionController &admission_;
  AdmissionLease lease_;
  SteadyClock::time_point started_at_;
};

struct ProxyFailure : std::runtime_error {
  ProxyFailure(http::status response_status, bool sent_headers,
               std::string message)
      : std::runtime_error(std::move(message)), status(response_status),
        headers_sent(sent_headers) {}
  http::status status;
  bool headers_sent;
};

bool is_hop_by_hop(http::field field) {
  switch (field) {
  case http::field::connection:
  case http::field::keep_alive:
  case http::field::proxy_authenticate:
  case http::field::proxy_authorization:
  case http::field::te:
  case http::field::trailer:
  case http::field::transfer_encoding:
  case http::field::upgrade:
    return true;
  default:
    return false;
  }
}

std::chrono::milliseconds parse_timeout(const GatewayConfig &config,
                                        const GatewayServer::Request &request) {
  const auto header = request["X-Request-Timeout-Ms"];
  if (header.empty()) {
    return config.default_request_timeout;
  }
  std::int64_t milliseconds{};
  const std::string_view value(header.data(), header.size());
  const auto [end, error] =
      std::from_chars(value.data(), value.data() + value.size(), milliseconds);
  if (error != std::errc{} || end != value.data() + value.size() ||
      milliseconds <= 0) {
    return config.default_request_timeout;
  }
  return std::clamp(std::chrono::milliseconds(milliseconds), 1ms,
                    config.maximum_request_timeout);
}

std::string client_id(const GatewayServer::Request &request,
                      const beast::tcp_stream &downstream) {
  const auto header = request["X-Client-ID"];
  if (!header.empty()) {
    return std::string(header.data(), header.size());
  }
  boost::system::error_code error;
  const auto endpoint = downstream.socket().remote_endpoint(error);
  return error ? "anonymous" : endpoint.address().to_string();
}

bool allow_degrade(const GatewayServer::Request &request) {
  const auto header = request["X-Allow-Degrade"];
  return header != "0" && header != "false" && header != "False";
}

unsigned retry_after_seconds(std::chrono::milliseconds retry_after) {
  if (retry_after <= std::chrono::milliseconds::zero()) {
    return 0;
  }
  return static_cast<unsigned>(
      std::max<std::int64_t>(1, (retry_after.count() + 999) / 1000));
}

bool is_timeout_error(const boost::system::error_code &error) {
  return error == beast::error::timeout || error == net::error::timed_out;
}

void close_stream_noexcept(
    const std::shared_ptr<beast::tcp_stream> &stream) noexcept {
  boost::system::error_code ignored;
  stream->socket().cancel(ignored);
  stream->socket().shutdown(tcp::socket::shutdown_both, ignored);
  stream->socket().close(ignored);
}

} // namespace

struct GatewayServer::CancellationState {
  std::atomic<bool> client_gone{};
  std::atomic<bool> finished{};
  std::mutex mutex;
  std::weak_ptr<UpstreamConnection> connection;
  std::weak_ptr<UpstreamPool> pool;

  void attach(const std::shared_ptr<UpstreamPool> &attached_pool,
              const std::shared_ptr<UpstreamConnection> &attached_connection) {
    std::scoped_lock lock(mutex);
    pool = attached_pool;
    connection = attached_connection;
  }

  void cancel_upstream() {
    std::shared_ptr<UpstreamPool> attached_pool;
    std::shared_ptr<UpstreamConnection> attached_connection;
    {
      std::scoped_lock lock(mutex);
      attached_pool = pool.lock();
      attached_connection = connection.lock();
    }
    if (attached_pool && attached_connection) {
      attached_pool->cancel(attached_connection);
    }
  }
};

struct GatewayServer::ProxyResult {
  unsigned status{};
  bool headers_sent{};
};

GatewayServer::GatewayServer(net::any_io_executor executor,
                             GatewayConfig config)
    : executor_(net::make_strand(std::move(executor))),
      config_(std::move(config)), acceptor_(executor_),
      admission_(config_.admission), rate_limiter_(config_.rate_limit),
      token_estimator_(config_.token_estimator),
      primary_(std::make_shared<UpstreamPool>(
          executor_, config_.primary, UpstreamRoute::primary,
          config_.circuit_breaker, config_.connect_timeout,
          config_.health_interval, config_.health_timeout, metrics_)) {
  if (config_.fallback_enabled) {
    fallback_ = std::make_shared<UpstreamPool>(
        executor_, config_.fallback, UpstreamRoute::fallback,
        config_.circuit_breaker, config_.connect_timeout,
        config_.health_interval, config_.health_timeout, metrics_);
  }

  const auto address = net::ip::make_address(config_.listen_host);
  tcp::endpoint endpoint(address, config_.listen_port);
  acceptor_.open(endpoint.protocol());
  acceptor_.set_option(net::socket_base::reuse_address(true));
  acceptor_.bind(endpoint);
  acceptor_.listen(net::socket_base::max_listen_connections);
  listen_port_ = acceptor_.local_endpoint().port();
}

void GatewayServer::start() {
  primary_->start_health_checks();
  if (fallback_) {
    fallback_->start_health_checks();
  }
  auto self = shared_from_this();
  net::co_spawn(
      executor_,
      [self]() -> net::awaitable<void> { co_await self->accept_loop(); },
      net::detached);
}

void GatewayServer::stop() {
  if (stopping_.exchange(true, std::memory_order_relaxed)) {
    return;
  }
  auto self = shared_from_this();
  net::dispatch(executor_, [self] {
    boost::system::error_code ignored;
    self->acceptor_.cancel(ignored);
    self->acceptor_.close(ignored);
    self->primary_->stop();
    if (self->fallback_) {
      self->fallback_->stop();
    }
  });
}

std::uint16_t GatewayServer::port() const { return listen_port_; }

net::awaitable<void> GatewayServer::accept_loop() {
  while (!stopping_.load(std::memory_order_relaxed)) {
    boost::system::error_code error;
    auto socket = co_await acceptor_.async_accept(
        net::redirect_error(net::use_awaitable, error));
    if (error) {
      if (stopping_.load(std::memory_order_relaxed) ||
          error == net::error::operation_aborted) {
        co_return;
      }
      continue;
    }
    auto self = shared_from_this();
    net::co_spawn(
        executor_,
        [self, socket = std::move(socket)]() mutable -> net::awaitable<void> {
          co_await self->handle_session(std::move(socket));
        },
        net::detached);
  }
}

net::awaitable<void> GatewayServer::handle_session(tcp::socket socket) {
  auto downstream = std::make_shared<beast::tcp_stream>(std::move(socket));
  LatencyGuard latency_guard(metrics_);
  beast::flat_buffer buffer;
  http::request_parser<http::string_body> request_parser;
  request_parser.body_limit(config_.maximum_request_body_bytes);
  downstream->expires_after(config_.default_request_timeout);
  boost::system::error_code read_error;
  co_await http::async_read(
      *downstream, buffer, request_parser,
      net::redirect_error(net::use_awaitable, read_error));
  if (read_error) {
    if (read_error == http::error::body_limit) {
      metrics_.record_decision(MetricDecision::reject_bad_request);
      metrics_.record_response(413);
      co_await send_json_error(downstream, http::status::payload_too_large,
                               "request_body_too_large",
                               "request body exceeds configured byte limit");
    }
    co_return;
  }
  auto request = request_parser.release();

  const auto target =
      std::string_view(request.target().data(), request.target().size());
  if (request.method() == http::verb::get && target == "/metrics") {
    co_await send_text(downstream, http::status::ok,
                       metrics_.render(admission_.snapshot()),
                       "text/plain; version=0.0.4; charset=utf-8");
    co_return;
  }
  if (request.method() == http::verb::get &&
      (target == "/healthz" || target == "/readyz")) {
    const bool ready = primary_->available();
    co_await send_text(downstream,
                       ready ? http::status::ok
                             : http::status::service_unavailable,
                       ready ? "ok\n" : "primary upstream unavailable\n",
                       "text/plain; charset=utf-8");
    co_return;
  }
  if (request.method() != http::verb::post || !target.starts_with("/v1/")) {
    metrics_.record_decision(MetricDecision::reject_bad_request);
    metrics_.record_response(404);
    co_await send_json_error(
        downstream, http::status::not_found, "not_found",
        "only POST /v1/*, GET /healthz, and GET /metrics are supported");
    co_return;
  }

  const auto estimate = token_estimator_.estimate(request.body());
  if (!estimate.valid_json) {
    metrics_.record_decision(MetricDecision::reject_bad_request);
    metrics_.record_response(400);
    co_await send_json_error(downstream, http::status::bad_request,
                             "invalid_json", estimate.error);
    co_return;
  }

  const auto rate = rate_limiter_.allow(client_id(request, *downstream),
                                        estimate.cost.total_tokens());
  if (!rate.allowed) {
    metrics_.record_decision(MetricDecision::reject_rate_limit);
    metrics_.record_response(429);
    co_await send_json_error(
        downstream, http::status::too_many_requests, "rate_limited",
        "per-client request or token budget exhausted", rate.retry_after);
    co_return;
  }

  const auto timeout = parse_timeout(config_, request);
  const auto deadline = SteadyClock::now() + timeout;
  auto admission = admission_.admit(estimate.cost, deadline,
                                    fallback_ && allow_degrade(request),
                                    route_availability(SteadyClock::now()));
  std::optional<std::uint64_t> queued_ticket;
  if (admission.kind == AdmissionKind::queued) {
    metrics_.record_decision(MetricDecision::queued);
    queued_ticket = admission.ticket;
    auto cancellation = std::make_shared<CancellationState>();
    auto self = shared_from_this();
    net::co_spawn(
        executor_,
        [self, downstream, cancellation]() -> net::awaitable<void> {
          co_await self->watch_client_disconnect(downstream, cancellation);
        },
        net::detached);
    net::steady_timer timer(executor_);
    auto poll_interval = config_.queue_poll_interval;
    const auto maximum_poll_interval =
        std::max(config_.queue_poll_interval, 20ms);
    while (admission.kind == AdmissionKind::queued &&
           !cancellation->client_gone.load(std::memory_order_relaxed)) {
      timer.expires_after(poll_interval);
      boost::system::error_code wait_error;
      co_await timer.async_wait(
          net::redirect_error(net::use_awaitable, wait_error));
      if (wait_error == net::error::operation_aborted) {
        break;
      }
      admission = admission_.poll(*queued_ticket,
                                  route_availability(SteadyClock::now()));
      if (admission.kind == AdmissionKind::queued &&
          poll_interval < maximum_poll_interval) {
        poll_interval = std::min(maximum_poll_interval, poll_interval * 2);
      }
    }
    if (cancellation->client_gone.load(std::memory_order_relaxed)) {
      const bool cancelled = admission_.cancel(*queued_ticket);
      (void)cancelled;
      co_return;
    }
    cancellation->finished.store(true, std::memory_order_relaxed);
  }

  if (!admission.lease.has_value()) {
    http::status status = http::status::too_many_requests;
    MetricDecision metric = MetricDecision::reject_overload;
    std::string code = "overloaded";
    if (admission.kind == AdmissionKind::rejected_deadline) {
      status = http::status::gateway_timeout;
      metric = MetricDecision::reject_deadline;
      code = "deadline_exceeded";
    } else if (admission.kind == AdmissionKind::rejected_oversize) {
      status = http::status::payload_too_large;
      code = "token_budget_too_large";
    } else if (admission.kind == AdmissionKind::rejected_unavailable) {
      metric = MetricDecision::reject_unavailable;
      code = "upstream_unavailable";
    }
    metrics_.record_decision(metric);
    metrics_.record_response(static_cast<unsigned>(status));
    co_await send_json_error(downstream, status, std::move(code),
                             admission.reason, admission.retry_after);
    close_stream_noexcept(downstream);
    co_return;
  }

  const auto lease = *admission.lease;
  AdmissionGuard admission_guard(admission_, lease);
  const auto &pool =
      lease.route == UpstreamRoute::primary ? primary_ : fallback_;
  if (!pool) {
    metrics_.record_decision(MetricDecision::reject_unavailable);
    metrics_.record_response(503);
    co_await send_json_error(downstream, http::status::service_unavailable,
                             "upstream_unavailable",
                             "selected route is not configured");
    co_return;
  }
  metrics_.record_decision(lease.route == UpstreamRoute::primary
                               ? MetricDecision::primary
                               : MetricDecision::fallback);

  if (!pool->config().model.empty() &&
      (lease.route == UpstreamRoute::fallback || estimate.model.empty())) {
    request.body() =
        token_estimator_.rewrite_model(request.body(), pool->config().model);
    request.prepare_payload();
  }

  auto cancellation = std::make_shared<CancellationState>();
  auto self = shared_from_this();
  net::co_spawn(
      executor_,
      [self, downstream, cancellation]() -> net::awaitable<void> {
        co_await self->watch_client_disconnect(downstream, cancellation);
      },
      net::detached);
  std::optional<http::status> proxy_error_status;
  std::string proxy_error_message;
  bool proxy_headers_sent = false;
  try {
    const auto result = co_await proxy_request(
        downstream, std::move(request), pool, lease, deadline, cancellation);
    metrics_.record_response(result.status);
  } catch (const ProxyFailure &failure) {
    proxy_error_status = failure.status;
    proxy_error_message = failure.what();
    proxy_headers_sent = failure.headers_sent;
  }
  if (proxy_error_status.has_value() &&
      !cancellation->client_gone.load(std::memory_order_relaxed) &&
      !proxy_headers_sent) {
    metrics_.record_response(static_cast<unsigned>(*proxy_error_status));
    co_await send_json_error(downstream, *proxy_error_status, "upstream_error",
                             std::move(proxy_error_message), 100ms);
  }
  cancellation->finished.store(true, std::memory_order_relaxed);
  close_stream_noexcept(downstream);
}

net::awaitable<void> GatewayServer::watch_client_disconnect(
    std::shared_ptr<beast::tcp_stream> downstream,
    std::shared_ptr<CancellationState> cancellation) {
  boost::system::error_code wait_error;
  co_await downstream->socket().async_wait(
      tcp::socket::wait_read,
      net::redirect_error(net::use_awaitable, wait_error));
  if (wait_error || cancellation->finished.load(std::memory_order_relaxed)) {
    co_return;
  }
  std::array<char, 1> probe{};
  boost::system::error_code receive_error;
  const auto received = downstream->socket().receive(
      net::buffer(probe), tcp::socket::message_peek, receive_error);
  if (received == 0 || receive_error == net::error::eof ||
      receive_error == net::error::connection_reset) {
    cancellation->client_gone.store(true, std::memory_order_relaxed);
    cancellation->cancel_upstream();
  }
}

net::awaitable<GatewayServer::ProxyResult> GatewayServer::proxy_request(
    std::shared_ptr<beast::tcp_stream> downstream, Request request,
    const std::shared_ptr<UpstreamPool> &pool, const AdmissionLease &lease,
    SteadyClock::time_point deadline,
    const std::shared_ptr<CancellationState> &cancellation) {
  if (!pool->allow_attempt()) {
    throw ProxyFailure(http::status::service_unavailable, false,
                       "upstream circuit is open");
  }

  std::shared_ptr<UpstreamConnection> connection;
  bool reusable = false;
  bool headers_sent = false;
  try {
    connection = co_await pool->acquire(deadline);
    cancellation->attach(pool, connection);
    co_await pool->ensure_connected(connection, deadline);
    connection->stream.expires_at(deadline);

    request.set(http::field::host, pool->config().host + ":" +
                                       std::to_string(pool->config().port));
    request.set(http::field::user_agent, "frontier-forge-gateway/0.5.0");
    request.set("X-Forge-Route", to_string(lease.route));
    const auto remaining = std::max<std::int64_t>(
        1, std::chrono::duration_cast<std::chrono::milliseconds>(
               deadline - SteadyClock::now())
               .count());
    request.set("X-Request-Timeout-Ms", std::to_string(remaining));
    request.erase(http::field::connection);
    request.keep_alive(true);
    co_await http::async_write(connection->stream, request, net::use_awaitable);

    http::response_parser<http::buffer_body> parser;
    parser.body_limit(config_.maximum_response_body_bytes);
    co_await http::async_read_header(connection->stream, connection->buffer,
                                     parser, net::use_awaitable);
    const auto upstream_status = parser.get().result_int();
    if (upstream_status >= 500) {
      pool->record_failure();
    } else {
      pool->record_success();
    }

    http::response<http::empty_body> response{parser.get().result(), 11};
    for (const auto &field : parser.get().base()) {
      if (!is_hop_by_hop(field.name()) &&
          field.name() != http::field::content_length) {
        response.set(field.name_string(), field.value());
      }
    }
    response.set(http::field::server, "frontier-forge-gateway/0.5.0");
    response.set("X-Forge-Route", to_string(lease.route));
    response.erase(http::field::content_length);
    response.chunked(true);
    response.keep_alive(false);
    http::response_serializer<http::empty_body> serializer(response);
    downstream->expires_at(deadline);
    boost::system::error_code downstream_error;
    co_await http::async_write_header(
        *downstream, serializer,
        net::redirect_error(net::use_awaitable, downstream_error));
    if (downstream_error) {
      cancellation->client_gone.store(true, std::memory_order_relaxed);
      throw ProxyFailure(http::status::bad_gateway, false,
                         "downstream disconnected before response headers");
    }
    headers_sent = true;

    std::array<char, 16 * 1024> body_buffer{};
    while (!parser.is_done()) {
      parser.get().body().data = body_buffer.data();
      parser.get().body().size = body_buffer.size();
      boost::system::error_code upstream_error;
      co_await http::async_read_some(
          connection->stream, connection->buffer, parser,
          net::redirect_error(net::use_awaitable, upstream_error));
      const auto produced = body_buffer.size() - parser.get().body().size;
      if (upstream_error && upstream_error != http::error::need_buffer) {
        throw boost::system::system_error(upstream_error);
      }
      if (produced > 0) {
        co_await net::async_write(
            *downstream,
            http::make_chunk(net::buffer(body_buffer.data(), produced)),
            net::redirect_error(net::use_awaitable, downstream_error));
        if (downstream_error) {
          cancellation->client_gone.store(true, std::memory_order_relaxed);
          cancellation->cancel_upstream();
          throw ProxyFailure(http::status::bad_gateway, true,
                             "downstream disconnected during response body");
        }
      }
    }
    co_await net::async_write(
        *downstream, http::make_chunk_last(),
        net::redirect_error(net::use_awaitable, downstream_error));
    if (downstream_error) {
      cancellation->client_gone.store(true, std::memory_order_relaxed);
    }
    reusable = parser.get().keep_alive() && !downstream_error;
    pool->release(connection, reusable);
    co_return ProxyResult{.status = upstream_status,
                          .headers_sent = headers_sent};
  } catch (const ProxyFailure &) {
    if (connection) {
      pool->release(connection, false);
    }
    throw;
  } catch (const boost::system::system_error &error) {
    if (connection) {
      pool->release(connection, false);
    }
    if (connection &&
        !cancellation->client_gone.load(std::memory_order_relaxed)) {
      pool->record_failure();
    }
    throw ProxyFailure(is_timeout_error(error.code())
                           ? http::status::gateway_timeout
                           : http::status::bad_gateway,
                       headers_sent, error.what());
  } catch (const std::exception &error) {
    if (connection) {
      pool->release(connection, false);
    }
    if (connection &&
        !cancellation->client_gone.load(std::memory_order_relaxed)) {
      pool->record_failure();
    }
    throw ProxyFailure(http::status::bad_gateway, headers_sent, error.what());
  }
}

net::awaitable<void> GatewayServer::send_json_error(
    const std::shared_ptr<beast::tcp_stream> &downstream, http::status status,
    std::string code, std::string message,
    std::chrono::milliseconds retry_after) {
  json::object error{{"error", json::object{{"code", std::move(code)},
                                            {"message", std::move(message)},
                                            {"type", "gateway_error"}}}};
  http::response<http::string_body> response{status, 11};
  response.set(http::field::content_type, "application/json");
  response.set(http::field::server, "frontier-forge-gateway/0.5.0");
  if (const auto seconds = retry_after_seconds(retry_after); seconds > 0) {
    response.set(http::field::retry_after, std::to_string(seconds));
  }
  response.keep_alive(false);
  response.body() = json::serialize(error);
  response.prepare_payload();
  boost::system::error_code ignored;
  co_await http::async_write(*downstream, response,
                             net::redirect_error(net::use_awaitable, ignored));
}

net::awaitable<void>
GatewayServer::send_text(const std::shared_ptr<beast::tcp_stream> &downstream,
                         http::status status, std::string body,
                         std::string content_type) {
  http::response<http::string_body> response{status, 11};
  response.set(http::field::content_type, std::move(content_type));
  response.set(http::field::server, "frontier-forge-gateway/0.5.0");
  response.keep_alive(false);
  response.body() = std::move(body);
  response.prepare_payload();
  boost::system::error_code ignored;
  co_await http::async_write(*downstream, response,
                             net::redirect_error(net::use_awaitable, ignored));
}

RouteAvailability
GatewayServer::route_availability(SteadyClock::time_point now) const {
  return {.primary = primary_->available(now),
          .fallback = fallback_ && fallback_->available(now)};
}

} // namespace frontier_forge
