#pragma once

#include <atomic>
#include <cstdint>
#include <memory>

#include <boost/asio/any_io_executor.hpp>
#include <boost/asio/awaitable.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/beast/core/tcp_stream.hpp>
#include <boost/beast/http.hpp>

#include "frontier_forge/admission.hpp"
#include "frontier_forge/config.hpp"
#include "frontier_forge/metrics.hpp"
#include "frontier_forge/rate_limiter.hpp"
#include "frontier_forge/token_estimator.hpp"
#include "frontier_forge/upstream_pool.hpp"

namespace frontier_forge {

class GatewayServer : public std::enable_shared_from_this<GatewayServer> {
public:
  using Request = boost::beast::http::request<boost::beast::http::string_body>;

  GatewayServer(boost::asio::any_io_executor executor, GatewayConfig config);

  void start();
  void stop();

  [[nodiscard]] std::uint16_t port() const;
  [[nodiscard]] AdmissionSnapshot admission_snapshot() const {
    return admission_.snapshot();
  }
  [[nodiscard]] const GatewayConfig &config() const noexcept { return config_; }

private:
  struct CancellationState;
  struct ProxyResult;

  boost::asio::awaitable<void> accept_loop();
  boost::asio::awaitable<void>
  handle_session(boost::asio::ip::tcp::socket socket);
  boost::asio::awaitable<void>
  watch_client_disconnect(std::shared_ptr<boost::beast::tcp_stream> downstream,
                          std::shared_ptr<CancellationState> cancellation);
  boost::asio::awaitable<ProxyResult>
  proxy_request(std::shared_ptr<boost::beast::tcp_stream> downstream,
                Request request, const std::shared_ptr<UpstreamPool> &pool,
                const AdmissionLease &lease, SteadyClock::time_point deadline,
                const std::shared_ptr<CancellationState> &cancellation);

  boost::asio::awaitable<void>
  send_json_error(const std::shared_ptr<boost::beast::tcp_stream> &downstream,
                  boost::beast::http::status status, std::string code,
                  std::string message,
                  std::chrono::milliseconds retry_after = {});
  boost::asio::awaitable<void>
  send_text(const std::shared_ptr<boost::beast::tcp_stream> &downstream,
            boost::beast::http::status status, std::string body,
            std::string content_type);

  [[nodiscard]] RouteAvailability
  route_availability(SteadyClock::time_point now) const;

  boost::asio::any_io_executor executor_;
  GatewayConfig config_;
  boost::asio::ip::tcp::acceptor acceptor_;
  std::uint16_t listen_port_{};
  GatewayMetrics metrics_;
  AdmissionController admission_;
  PerClientRateLimiter rate_limiter_;
  TokenEstimator token_estimator_;
  std::shared_ptr<UpstreamPool> primary_;
  std::shared_ptr<UpstreamPool> fallback_;
  std::atomic<bool> stopping_{};
};

} // namespace frontier_forge
