#include <algorithm>
#include <charconv>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <thread>

#include <boost/asio/io_context.hpp>
#include <boost/asio/signal_set.hpp>

#include "mock_upstream/mock_upstream.hpp"

int main(int argc, char **argv) {
  try {
    std::uint16_t port = 18080;
    if (argc == 3 && std::string_view(argv[1]) == "--port") {
      const std::string_view value(argv[2]);
      const auto [end, error] =
          std::from_chars(value.data(), value.data() + value.size(), port);
      if (error != std::errc{} || end != value.data() + value.size()) {
        throw std::invalid_argument("--port expects an integer");
      }
    } else if (argc != 1) {
      throw std::invalid_argument("usage: mock_upstream [--port PORT]");
    }
    boost::asio::io_context io_context(2);
    auto server = std::make_shared<frontier_forge::test::MockUpstream>(
        io_context.get_executor(), port);
    server->start();
    boost::asio::signal_set signals(io_context, SIGINT, SIGTERM);
    signals.async_wait(
        [server, &io_context](const boost::system::error_code &, int) {
          server->stop();
          io_context.stop();
        });
    std::cout << "mock upstream listening on 127.0.0.1:" << server->port()
              << '\n';
    std::thread worker([&io_context] { io_context.run(); });
    io_context.run();
    worker.join();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "mock upstream failed: " << error.what() << '\n';
    return 2;
  }
}
