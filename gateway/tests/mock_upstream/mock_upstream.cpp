#include "mock_upstream/mock_upstream.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <memory>
#include <string>
#include <string_view>

#include <boost/asio/co_spawn.hpp>
#include <boost/asio/detached.hpp>
#include <boost/asio/redirect_error.hpp>
#include <boost/asio/steady_timer.hpp>
#include <boost/asio/use_awaitable.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/http/chunk_encode.hpp>
#include <boost/json.hpp>

namespace frontier_forge::test {
namespace net = boost::asio;
namespace beast = boost::beast;
namespace http = beast::http;
namespace json = boost::json;
using tcp = net::ip::tcp;
using namespace std::chrono_literals;

namespace {

std::int64_t header_integer(const http::request<http::string_body> &request,
                            std::string_view name, std::int64_t fallback = 0) {
  const auto header = request[name];
  if (header.empty()) {
    return fallback;
  }
  std::int64_t result{};
  const std::string_view text(header.data(), header.size());
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), result);
  return error == std::errc{} && end == text.data() + text.size() ? result
                                                                  : fallback;
}

bool streaming_request(std::string_view body) {
  boost::system::error_code error;
  const auto document = json::parse(body, error);
  if (error || !document.is_object()) {
    return false;
  }
  const auto *stream = document.as_object().if_contains("stream");
  return stream && stream->is_bool() && stream->as_bool();
}

std::string requested_model(std::string_view body) {
  boost::system::error_code error;
  const auto document = json::parse(body, error);
  if (error || !document.is_object()) {
    return "unknown";
  }
  const auto *model = document.as_object().if_contains("model");
  return model && model->is_string() ? std::string(model->as_string().c_str())
                                     : "unknown";
}

class ActiveGuard {
public:
  ActiveGuard(std::atomic<std::size_t> &active,
              std::atomic<std::size_t> &maximum)
      : active_(active) {
    const auto current = active_.fetch_add(1, std::memory_order_relaxed) + 1;
    auto observed = maximum.load(std::memory_order_relaxed);
    while (current > observed &&
           !maximum.compare_exchange_weak(observed, current,
                                          std::memory_order_relaxed)) {
    }
  }
  ~ActiveGuard() { active_.fetch_sub(1, std::memory_order_relaxed); }

private:
  std::atomic<std::size_t> &active_;
};

} // namespace

MockUpstream::MockUpstream(net::any_io_executor executor, std::uint16_t port)
    : executor_(std::move(executor)), acceptor_(executor_) {
  tcp::endpoint endpoint(net::ip::make_address("127.0.0.1"), port);
  acceptor_.open(endpoint.protocol());
  acceptor_.set_option(net::socket_base::reuse_address(true));
  acceptor_.bind(endpoint);
  acceptor_.listen(net::socket_base::max_listen_connections);
}

void MockUpstream::start() {
  auto self = shared_from_this();
  net::co_spawn(
      executor_,
      [self]() -> net::awaitable<void> { co_await self->accept_loop(); },
      net::detached);
}

void MockUpstream::stop() {
  stopping_.store(true, std::memory_order_relaxed);
  boost::system::error_code ignored;
  acceptor_.cancel(ignored);
  acceptor_.close(ignored);
}

std::uint16_t MockUpstream::port() const {
  boost::system::error_code error;
  const auto endpoint = acceptor_.local_endpoint(error);
  return error ? 0 : endpoint.port();
}

net::awaitable<void> MockUpstream::accept_loop() {
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
    accepted_connections_.fetch_add(1, std::memory_order_relaxed);
    auto self = shared_from_this();
    net::co_spawn(
        executor_,
        [self, socket = std::move(socket)]() mutable -> net::awaitable<void> {
          co_await self->session(std::move(socket));
        },
        net::detached);
  }
}

net::awaitable<void> MockUpstream::session(tcp::socket socket) {
  beast::tcp_stream stream(std::move(socket));
  beast::flat_buffer buffer;
  while (!stopping_.load(std::memory_order_relaxed)) {
    http::request<http::string_body> request;
    boost::system::error_code read_error;
    co_await http::async_read(
        stream, buffer, request,
        net::redirect_error(net::use_awaitable, read_error));
    if (read_error) {
      co_return;
    }

    const auto target =
        std::string_view(request.target().data(), request.target().size());
    if (request.method() == http::verb::get && target == "/health") {
      const bool healthy = healthy_.load(std::memory_order_relaxed);
      http::response<http::string_body> response{
          healthy ? http::status::ok : http::status::service_unavailable, 11};
      response.set(http::field::content_type, "application/json");
      response.keep_alive(request.keep_alive());
      response.body() =
          healthy ? R"({"status":"ok"})" : R"({"status":"unhealthy"})";
      response.prepare_payload();
      boost::system::error_code write_error;
      co_await http::async_write(
          stream, response,
          net::redirect_error(net::use_awaitable, write_error));
      if (write_error || !response.keep_alive()) {
        co_return;
      }
      continue;
    }

    ActiveGuard active_guard(active_requests_, maximum_active_requests_);
    const auto latency = std::chrono::milliseconds(std::max<std::int64_t>(
        0, header_integer(request, "X-Mock-Latency-Ms")));
    if (latency > 0ms) {
      net::steady_timer timer(executor_);
      timer.expires_after(latency);
      boost::system::error_code timer_error;
      co_await timer.async_wait(
          net::redirect_error(net::use_awaitable, timer_error));
      if (timer_error) {
        co_return;
      }
    }

    const auto disconnect = request["X-Mock-Disconnect"];
    if (disconnect == "before_headers") {
      boost::system::error_code ignored;
      stream.socket().close(ignored);
      co_return;
    }

    const auto injected_status = std::clamp<std::int64_t>(
        header_integer(request, "X-Mock-Status", 200), 100, 599);
    const auto status = static_cast<http::status>(injected_status);
    const auto model = requested_model(request.body());
    if (!streaming_request(request.body())) {
      json::object body{
          {"id", "mock-completion"},
          {"object", "chat.completion"},
          {"model", model},
          {"choices", json::array{json::object{
                          {"index", 0},
                          {"message", json::object{{"role", "assistant"},
                                                   {"content", "mock-ok"}}},
                          {"finish_reason", "stop"}}}}};
      http::response<http::string_body> response{status, 11};
      response.set(http::field::content_type, "application/json");
      response.set("X-Mock-Observed-Route", request["X-Forge-Route"]);
      response.keep_alive(request.keep_alive());
      response.body() = json::serialize(body);
      response.prepare_payload();
      boost::system::error_code write_error;
      co_await http::async_write(
          stream, response,
          net::redirect_error(net::use_awaitable, write_error));
      if (write_error) {
        disconnected_writes_.fetch_add(1, std::memory_order_relaxed);
        co_return;
      }
      if (!response.keep_alive()) {
        co_return;
      }
      continue;
    }

    http::response<http::empty_body> response{status, 11};
    response.set(http::field::content_type, "text/event-stream");
    response.set(http::field::cache_control, "no-cache");
    response.set("X-Mock-Observed-Route", request["X-Forge-Route"]);
    response.chunked(true);
    response.keep_alive(request.keep_alive());
    http::response_serializer<http::empty_body> serializer(response);
    boost::system::error_code write_error;
    co_await http::async_write_header(
        stream, serializer,
        net::redirect_error(net::use_awaitable, write_error));
    if (write_error) {
      disconnected_writes_.fetch_add(1, std::memory_order_relaxed);
      co_return;
    }

    const std::array<std::string_view, 3> chunks{
        "data: "
        "{\"id\":\"mock\",\"choices\":[{\"delta\":{\"content\":\"A\"}}]}\n\n",
        "data: "
        "{\"id\":\"mock\",\"choices\":[{\"delta\":{\"content\":\"B\"}}]}\n\n",
        "data: [DONE]\n\n"};
    for (std::size_t index = 0; index < chunks.size(); ++index) {
      co_await net::async_write(
          stream, http::make_chunk(net::buffer(chunks[index])),
          net::redirect_error(net::use_awaitable, write_error));
      if (write_error) {
        disconnected_writes_.fetch_add(1, std::memory_order_relaxed);
        co_return;
      }
      if (index == 0 && disconnect == "mid_body") {
        boost::system::error_code ignored;
        stream.socket().close(ignored);
        co_return;
      }
      if (index + 1 < chunks.size()) {
        net::steady_timer timer(executor_);
        timer.expires_after(25ms);
        co_await timer.async_wait(net::use_awaitable);
      }
    }
    co_await net::async_write(
        stream, http::make_chunk_last(),
        net::redirect_error(net::use_awaitable, write_error));
    if (write_error) {
      disconnected_writes_.fetch_add(1, std::memory_order_relaxed);
      co_return;
    }
    if (!response.keep_alive()) {
      co_return;
    }
  }
}

} // namespace frontier_forge::test
