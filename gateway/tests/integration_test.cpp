#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <boost/asio/connect.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/json.hpp>

#include "frontier_forge/gateway_server.hpp"
#include "mock_upstream/mock_upstream.hpp"

namespace frontier_forge {
namespace {

namespace net = boost::asio;
namespace beast = boost::beast;
namespace http = beast::http;
namespace json = boost::json;
using tcp = net::ip::tcp;
using namespace std::chrono_literals;

struct ClientResponse {
  unsigned status{};
  std::string body;
  std::map<std::string, std::string> headers;
};

ClientResponse request(std::uint16_t port, http::verb method,
                       std::string target, std::string body = {},
                       const std::map<std::string, std::string> &headers = {}) {
  net::io_context io_context;
  tcp::resolver resolver(io_context);
  beast::tcp_stream stream(io_context);
  const auto endpoints = resolver.resolve("127.0.0.1", std::to_string(port));
  stream.connect(endpoints);
  http::request<http::string_body> outgoing{method, std::move(target), 11};
  outgoing.set(http::field::host, "127.0.0.1");
  outgoing.set(http::field::user_agent, "gateway-integration-test");
  for (const auto &[name, value] : headers) {
    outgoing.set(name, value);
  }
  outgoing.body() = std::move(body);
  outgoing.prepare_payload();
  http::write(stream, outgoing);
  beast::flat_buffer buffer;
  http::response<http::string_body> incoming;
  http::read(stream, buffer, incoming);
  ClientResponse result{.status = incoming.result_int(),
                        .body = std::move(incoming.body())};
  for (const auto &field : incoming.base()) {
    result.headers.emplace(std::string(field.name_string()),
                           std::string(field.value()));
  }
  return result;
}

std::string chat_body(bool stream = false, int max_tokens = 16) {
  json::object body{
      {"model", "caller-model"},
      {"messages", json::array{json::object{{"role", "user"},
                                            {"content", "hello gateway"}}}},
      {"max_tokens", max_tokens},
      {"stream", stream}};
  return json::serialize(body);
}

bool wait_until(const std::function<bool()> &condition,
                std::chrono::milliseconds timeout = 2s) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (condition()) {
      return true;
    }
    std::this_thread::sleep_for(5ms);
  }
  return condition();
}

struct StreamTiming {
  unsigned status{};
  std::string body;
  std::chrono::milliseconds first_body_byte{};
  std::chrono::milliseconds complete{};
};

StreamTiming streaming_request(std::uint16_t port) {
  net::io_context io_context;
  tcp::resolver resolver(io_context);
  beast::tcp_stream stream(io_context);
  stream.connect(resolver.resolve("127.0.0.1", std::to_string(port)));
  http::request<http::string_body> outgoing{http::verb::post,
                                            "/v1/chat/completions", 11};
  outgoing.set(http::field::host, "127.0.0.1");
  outgoing.set("X-Client-ID", "stream-client");
  outgoing.body() = chat_body(true);
  outgoing.prepare_payload();
  const auto started_at = std::chrono::steady_clock::now();
  http::write(stream, outgoing);

  beast::flat_buffer buffer;
  http::response_parser<http::buffer_body> parser;
  http::read_header(stream, buffer, parser);
  std::array<char, 1024> storage{};
  StreamTiming result{.status = parser.get().result_int()};
  bool saw_body = false;
  while (!parser.is_done()) {
    parser.get().body().data = storage.data();
    parser.get().body().size = storage.size();
    boost::system::error_code error;
    http::read_some(stream, buffer, parser, error);
    const auto produced = storage.size() - parser.get().body().size;
    if (produced > 0) {
      if (!saw_body) {
        saw_body = true;
        result.first_body_byte =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - started_at);
      }
      result.body.append(storage.data(), produced);
    }
    if (error && error != http::error::need_buffer) {
      throw boost::system::system_error(error);
    }
  }
  result.complete = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - started_at);
  return result;
}

class GatewayIntegrationTest : public ::testing::Test {
protected:
  void SetUp() override {
    primary = std::make_shared<test::MockUpstream>(io_context.get_executor());
    fallback = std::make_shared<test::MockUpstream>(io_context.get_executor());
    primary->start();
    fallback->start();
    for (int index = 0; index < 4; ++index) {
      workers.emplace_back([this] { io_context.run(); });
    }
  }

  void TearDown() override {
    if (gateway) {
      gateway->stop();
    }
    if (primary) {
      primary->stop();
    }
    if (fallback) {
      fallback->stop();
    }
    io_context.stop();
    for (auto &worker : workers) {
      worker.join();
    }
  }

  GatewayConfig default_config() const {
    GatewayConfig config;
    config.listen_port = 0;
    config.primary.port = primary->port();
    config.primary.model = "primary-model";
    config.primary.connection_pool_size = 4;
    config.fallback_enabled = true;
    config.fallback.port = fallback->port();
    config.fallback.model = "fallback-model";
    config.fallback.connection_pool_size = 4;
    config.admission.primary_token_capacity = 1000;
    config.admission.fallback_token_capacity = 1000;
    config.admission.max_queue_requests = 2;
    config.admission.max_queue_tokens = 1000;
    config.admission.degrade_utilization = 0.8;
    config.admission.initial_service_time = 100ms;
    config.admission.minimum_execution_budget = 10ms;
    config.rate_limit.requests_per_second = 10000.0;
    config.rate_limit.request_burst = 10000.0;
    config.rate_limit.tokens_per_second = 1e9;
    config.rate_limit.token_burst = 1e9;
    config.circuit_breaker.failure_threshold = 1;
    config.circuit_breaker.recovery_timeout = 1s;
    config.default_request_timeout = 2s;
    config.maximum_request_timeout = 5s;
    config.health_interval = 100ms;
    config.health_timeout = 100ms;
    return config;
  }

  void start_gateway(GatewayConfig config) {
    gateway = std::make_shared<GatewayServer>(io_context.get_executor(),
                                              std::move(config));
    gateway->start();
    ASSERT_TRUE(wait_until([this] {
      try {
        return request(gateway->port(), http::verb::get, "/healthz").status ==
               200;
      } catch (const std::exception &) {
        return false;
      }
    }));
  }

  net::io_context io_context{4};
  std::shared_ptr<test::MockUpstream> primary;
  std::shared_ptr<test::MockUpstream> fallback;
  std::shared_ptr<GatewayServer> gateway;
  std::vector<std::thread> workers;
};

TEST_F(GatewayIntegrationTest, PassesThroughOpenAIJsonAndExposesMetrics) {
  start_gateway(default_config());
  const auto response =
      request(gateway->port(), http::verb::post, "/v1/chat/completions",
              chat_body(), {{"X-Client-ID", "alice"}});
  ASSERT_EQ(response.status, 200U);
  const auto parsed = json::parse(response.body).as_object();
  EXPECT_EQ(parsed.at("model").as_string(), "caller-model");
  EXPECT_EQ(response.headers.at("X-Forge-Route"), "primary");
  EXPECT_EQ(response.headers.at("X-Mock-Observed-Route"), "primary");

  const auto metrics = request(gateway->port(), http::verb::get, "/metrics");
  EXPECT_EQ(metrics.status, 200U);
  EXPECT_NE(metrics.body.find("decision=\"primary\"} 1"), std::string::npos);
  EXPECT_NE(metrics.body.find("forge_gateway_active_tokens"),
            std::string::npos);
}

TEST_F(GatewayIntegrationTest, ReusesKeepAliveConnectionsFromTheBoundedPool) {
  auto config = default_config();
  config.health_interval = 10s;
  start_gateway(config);
  ASSERT_TRUE(
      wait_until([this] { return primary->accepted_connections() >= 1; }));
  const auto baseline = primary->accepted_connections();
  for (int index = 0; index < 3; ++index) {
    const auto response =
        request(gateway->port(), http::verb::post, "/v1/chat/completions",
                chat_body(), {{"X-Client-ID", "pool-client"}});
    ASSERT_EQ(response.status, 200U);
  }
  EXPECT_LE(primary->accepted_connections() - baseline, 1U);
}

TEST_F(GatewayIntegrationTest,
       ReconnectsBeforeReusingPeerClosedKeepAliveSocket) {
  auto config = default_config();
  config.fallback_enabled = false;
  config.primary.connection_pool_size = 1;
  config.health_interval = 10s;
  start_gateway(config);

  const auto first = request(
      gateway->port(), http::verb::post, "/v1/chat/completions", chat_body(),
      {{"X-Client-ID", "idle-close-first"},
       {"X-Mock-Idle-Close-Ms", "20"}});
  ASSERT_EQ(first.status, 200U);
  ASSERT_TRUE(
      wait_until([this] { return primary->idle_disconnects() == 1; }));
  const auto connections_before_retry = primary->accepted_connections();

  const auto second =
      request(gateway->port(), http::verb::post, "/v1/chat/completions",
              chat_body(), {{"X-Client-ID", "idle-close-second"}});
  EXPECT_EQ(second.status, 200U);
  EXPECT_EQ(primary->accepted_connections(), connections_before_retry + 1);
}

TEST_F(GatewayIntegrationTest, CircuitBreakerDegradesToFallbackModel) {
  start_gateway(default_config());
  const auto failed = request(
      gateway->port(), http::verb::post, "/v1/chat/completions", chat_body(),
      {{"X-Client-ID", "circuit-client"}, {"X-Mock-Status", "500"}});
  EXPECT_EQ(failed.status, 500U);

  const auto degraded =
      request(gateway->port(), http::verb::post, "/v1/chat/completions",
              chat_body(), {{"X-Client-ID", "circuit-client"}});
  ASSERT_EQ(degraded.status, 200U);
  EXPECT_EQ(degraded.headers.at("X-Forge-Route"), "fallback");
  const auto parsed = json::parse(degraded.body).as_object();
  EXPECT_EQ(parsed.at("model").as_string(), "fallback-model");
}

TEST_F(GatewayIntegrationTest, StreamsSseWithoutBufferingWholeResponse) {
  start_gateway(default_config());
  const auto response = streaming_request(gateway->port());
  EXPECT_EQ(response.status, 200U);
  EXPECT_NE(response.body.find("\"content\":\"A\""), std::string::npos);
  EXPECT_NE(response.body.find("\"content\":\"B\""), std::string::npos);
  EXPECT_NE(response.body.find("data: [DONE]"), std::string::npos);
  EXPECT_GE(response.complete - response.first_body_byte, 20ms);
}

TEST_F(GatewayIntegrationTest, ConvertsInjectedLatencyToDeadlineFailure) {
  start_gateway(default_config());
  const auto response = request(gateway->port(), http::verb::post,
                                "/v1/chat/completions", chat_body(),
                                {{"X-Client-ID", "timeout-client"},
                                 {"X-Mock-Latency-Ms", "300"},
                                 {"X-Request-Timeout-Ms", "50"}});
  EXPECT_EQ(response.status, 504U);
  EXPECT_NE(response.body.find("upstream_error"), std::string::npos);
}

TEST_F(GatewayIntegrationTest, ConvertsUpstreamDisconnectToBadGateway) {
  start_gateway(default_config());
  const auto response = request(gateway->port(), http::verb::post,
                                "/v1/chat/completions", chat_body(),
                                {{"X-Client-ID", "disconnect-client"},
                                 {"X-Mock-Disconnect", "before_headers"}});
  EXPECT_EQ(response.status, 502U);
  EXPECT_NE(response.body.find("upstream_error"), std::string::npos);
}

TEST_F(GatewayIntegrationTest, MidStreamDisconnectTripsCircuitForNextRequest) {
  start_gateway(default_config());
  EXPECT_THROW(request(gateway->port(), http::verb::post,
                       "/v1/chat/completions", chat_body(true),
                       {{"X-Client-ID", "midstream-client"},
                        {"X-Mock-Disconnect", "mid_body"}}),
               boost::system::system_error);
  const auto recovered =
      request(gateway->port(), http::verb::post, "/v1/chat/completions",
              chat_body(), {{"X-Client-ID", "midstream-client"}});
  EXPECT_EQ(recovered.status, 200U);
  EXPECT_EQ(recovered.headers.at("X-Forge-Route"), "fallback");
}

TEST_F(GatewayIntegrationTest, BoundsQueueAndFastRejectsExcessLoad) {
  auto config = default_config();
  config.fallback_enabled = false;
  config.admission.primary_token_capacity = 100;
  config.admission.max_queue_requests = 1;
  config.admission.max_queue_tokens = 100;
  config.admission.degrade_utilization = 1.0;
  start_gateway(config);

  ClientResponse first;
  ClientResponse second;
  const auto body = chat_body(false, 80);
  std::thread first_client([&] {
    first =
        request(gateway->port(), http::verb::post, "/v1/chat/completions", body,
                {{"X-Client-ID", "load-1"}, {"X-Mock-Latency-Ms", "250"}});
  });
  ASSERT_TRUE(wait_until([this] { return primary->active_requests() == 1; }));
  std::thread second_client([&] {
    second =
        request(gateway->port(), http::verb::post, "/v1/chat/completions", body,
                {{"X-Client-ID", "load-2"}, {"X-Mock-Latency-Ms", "250"}});
  });
  ASSERT_TRUE(wait_until(
      [this] { return gateway->admission_snapshot().queue_requests == 1; }));

  const auto started_at = std::chrono::steady_clock::now();
  const auto rejected =
      request(gateway->port(), http::verb::post, "/v1/chat/completions", body,
              {{"X-Client-ID", "load-3"}, {"X-Mock-Latency-Ms", "250"}});
  const auto reject_latency = std::chrono::steady_clock::now() - started_at;
  EXPECT_EQ(rejected.status, 429U);
  EXPECT_LT(reject_latency, 100ms);
  EXPECT_TRUE(rejected.headers.contains("Retry-After"));
  EXPECT_NE(rejected.body.find("\"code\":\"overloaded\""),
            std::string::npos);
  const auto metrics = request(gateway->port(), http::verb::get, "/metrics");
  EXPECT_NE(metrics.body.find("decision=\"reject_overload\"} 1"),
            std::string::npos);
  EXPECT_NE(metrics.body.find("class=\"4xx\"} 1"), std::string::npos);
  EXPECT_NE(metrics.body.find("class=\"5xx\"} 0"), std::string::npos);

  first_client.join();
  second_client.join();
  EXPECT_EQ(first.status, 200U);
  EXPECT_EQ(second.status, 200U);
  const auto snapshot = gateway->admission_snapshot();
  EXPECT_EQ(snapshot.queue_requests, 0U);
  EXPECT_EQ(snapshot.max_observed_queue_requests, 1U);
}

TEST_F(GatewayIntegrationTest, EnforcesPerClientRateLimit) {
  auto config = default_config();
  config.rate_limit.requests_per_second = 0.1;
  config.rate_limit.request_burst = 1.0;
  start_gateway(config);
  const auto first =
      request(gateway->port(), http::verb::post, "/v1/chat/completions",
              chat_body(), {{"X-Client-ID", "rate-client"}});
  const auto second =
      request(gateway->port(), http::verb::post, "/v1/chat/completions",
              chat_body(), {{"X-Client-ID", "rate-client"}});
  EXPECT_EQ(first.status, 200U);
  EXPECT_EQ(second.status, 429U);
  EXPECT_TRUE(second.headers.contains("Retry-After"));
}

TEST_F(GatewayIntegrationTest, RejectsRequestBodyAboveConfiguredBound) {
  auto config = default_config();
  config.maximum_request_body_bytes = 128;
  start_gateway(config);
  const auto response =
      request(gateway->port(), http::verb::post, "/v1/chat/completions",
              std::string(512, 'x'), {{"X-Client-ID", "large-body-client"}});
  EXPECT_EQ(response.status, 413U);
  EXPECT_NE(response.body.find("request_body_too_large"), std::string::npos);
}

TEST_F(GatewayIntegrationTest, HealthChecksChangeReadiness) {
  auto config = default_config();
  config.health_interval = 20ms;
  start_gateway(config);
  primary->set_healthy(false);
  ASSERT_TRUE(wait_until([this] {
    return request(gateway->port(), http::verb::get, "/healthz").status == 503;
  }));
  primary->set_healthy(true);
  EXPECT_TRUE(wait_until([this] {
    return request(gateway->port(), http::verb::get, "/healthz").status == 200;
  }));
}

TEST_F(GatewayIntegrationTest, PreservesUnavailableAsServiceUnavailable) {
  auto config = default_config();
  config.fallback_enabled = false;
  config.health_interval = 20ms;
  start_gateway(config);
  primary->set_healthy(false);
  ASSERT_TRUE(wait_until([this] {
    return request(gateway->port(), http::verb::get, "/healthz").status == 503;
  }));

  const auto response =
      request(gateway->port(), http::verb::post, "/v1/chat/completions",
              chat_body(), {{"X-Client-ID", "unavailable-client"}});
  EXPECT_EQ(response.status, 503U);
  EXPECT_TRUE(response.headers.contains("Retry-After"));
  EXPECT_NE(response.body.find("\"code\":\"upstream_unavailable\""),
            std::string::npos);
}

TEST_F(GatewayIntegrationTest, DownstreamCancellationReleasesAdmissionLease) {
  start_gateway(default_config());
  net::io_context client_context;
  beast::tcp_stream client(client_context);
  client.connect(tcp::resolver(client_context)
                     .resolve("127.0.0.1", std::to_string(gateway->port())));
  http::request<http::string_body> outgoing{http::verb::post,
                                            "/v1/chat/completions", 11};
  outgoing.set(http::field::host, "127.0.0.1");
  outgoing.set("X-Client-ID", "cancel-client");
  outgoing.set("X-Mock-Latency-Ms", "500");
  outgoing.body() = chat_body();
  outgoing.prepare_payload();
  http::write(client, outgoing);
  ASSERT_TRUE(wait_until([this] { return primary->active_requests() == 1; }));
  boost::system::error_code ignored;
  client.socket().shutdown(tcp::socket::shutdown_both, ignored);
  client.socket().close(ignored);
  EXPECT_TRUE(wait_until(
      [this] { return gateway->admission_snapshot().active_requests == 0; },
      250ms));
}

} // namespace
} // namespace frontier_forge
