#include <gtest/gtest.h>

#include <array>
#include <stdexcept>

#include "frontier_forge/config.hpp"

namespace frontier_forge {
namespace {

TEST(GatewayConfigTest, ParsesRouteAndAdmissionFlags) {
  std::array arguments{"forge_gateway",
                       "--listen-port",
                       "9100",
                       "--primary-port",
                       "8100",
                       "--fallback-port",
                       "8101",
                       "--fallback-model",
                       "small-model",
                       "--primary-token-capacity",
                       "4096",
                       "--max-queue-requests",
                       "7"};
  std::array<char *, arguments.size()> argv{};
  for (std::size_t index = 0; index < arguments.size(); ++index) {
    argv[index] = const_cast<char *>(arguments[index]);
  }
  const auto parsed =
      parse_command_line(static_cast<int>(argv.size()), argv.data());
  EXPECT_EQ(parsed.config.listen_port, 9100);
  EXPECT_EQ(parsed.config.primary.port, 8100);
  EXPECT_TRUE(parsed.config.fallback_enabled);
  EXPECT_EQ(parsed.config.fallback.port, 8101);
  EXPECT_EQ(parsed.config.fallback.model, "small-model");
  EXPECT_EQ(parsed.config.admission.primary_token_capacity, 4096U);
  EXPECT_EQ(parsed.config.admission.max_queue_requests, 7U);
}

TEST(GatewayConfigTest, RejectsUnknownAndInvalidFlags) {
  char program[] = "forge_gateway";
  char unknown[] = "--unknown";
  char *unknown_argv[]{program, unknown};
  EXPECT_THROW(static_cast<void>(parse_command_line(2, unknown_argv)),
               std::invalid_argument);

  char timeout_option[] = "--request-timeout-ms";
  char invalid_timeout[] = "-1";
  char *timeout_argv[]{program, timeout_option, invalid_timeout};
  EXPECT_THROW(static_cast<void>(parse_command_line(3, timeout_argv)),
               std::invalid_argument);
}

} // namespace
} // namespace frontier_forge
