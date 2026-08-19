#include <algorithm>
#include <exception>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#include <boost/asio/io_context.hpp>
#include <boost/asio/signal_set.hpp>

#include "frontier_forge/config.hpp"
#include "frontier_forge/gateway_server.hpp"

int main(int argc, char **argv) {
  try {
    auto command_line = frontier_forge::parse_command_line(argc, argv);
    if (command_line.show_help) {
      std::cout << frontier_forge::command_line_help();
      return 0;
    }

    boost::asio::io_context io_context(
        static_cast<int>(command_line.config.io_threads));
    auto server = std::make_shared<frontier_forge::GatewayServer>(
        io_context.get_executor(), command_line.config);
    server->start();

    boost::asio::signal_set signals(io_context, SIGINT, SIGTERM);
    signals.async_wait(
        [server](const boost::system::error_code &, int) { server->stop(); });

    std::cout << "frontier-forge gateway listening on "
              << command_line.config.listen_host << ':' << server->port()
              << " (primary " << command_line.config.primary.host << ':'
              << command_line.config.primary.port << ")\n";

    std::vector<std::thread> threads;
    const auto worker_count =
        std::max<std::size_t>(1, command_line.config.io_threads);
    threads.reserve(worker_count - 1);
    for (std::size_t index = 1; index < worker_count; ++index) {
      threads.emplace_back([&io_context] { io_context.run(); });
    }
    io_context.run();
    for (auto &thread : threads) {
      thread.join();
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "gateway startup failed: " << error.what() << '\n';
    return 2;
  }
}
