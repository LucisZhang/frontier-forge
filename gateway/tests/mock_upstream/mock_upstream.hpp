#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>

#include <boost/asio/any_io_executor.hpp>
#include <boost/asio/awaitable.hpp>
#include <boost/asio/ip/tcp.hpp>

namespace frontier_forge::test {

class MockUpstream : public std::enable_shared_from_this<MockUpstream> {
public:
  explicit MockUpstream(boost::asio::any_io_executor executor,
                        std::uint16_t port = 0);

  void start();
  void stop();
  void set_healthy(bool healthy) noexcept {
    healthy_.store(healthy, std::memory_order_relaxed);
  }

  [[nodiscard]] std::uint16_t port() const;
  [[nodiscard]] std::size_t active_requests() const noexcept {
    return active_requests_.load(std::memory_order_relaxed);
  }
  [[nodiscard]] std::size_t maximum_active_requests() const noexcept {
    return maximum_active_requests_.load(std::memory_order_relaxed);
  }
  [[nodiscard]] std::size_t disconnected_writes() const noexcept {
    return disconnected_writes_.load(std::memory_order_relaxed);
  }
  [[nodiscard]] std::size_t accepted_connections() const noexcept {
    return accepted_connections_.load(std::memory_order_relaxed);
  }

private:
  boost::asio::awaitable<void> accept_loop();
  boost::asio::awaitable<void> session(boost::asio::ip::tcp::socket socket);

  boost::asio::any_io_executor executor_;
  boost::asio::ip::tcp::acceptor acceptor_;
  std::atomic<bool> stopping_{};
  std::atomic<bool> healthy_{true};
  std::atomic<std::size_t> active_requests_{};
  std::atomic<std::size_t> maximum_active_requests_{};
  std::atomic<std::size_t> disconnected_writes_{};
  std::atomic<std::size_t> accepted_connections_{};
};

} // namespace frontier_forge::test
