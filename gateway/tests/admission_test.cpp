#include <gtest/gtest.h>

#include <chrono>
#include <thread>
#include <vector>

#include "frontier_forge/admission.hpp"

namespace frontier_forge {
namespace {

using namespace std::chrono_literals;

AdmissionConfig small_config() {
  return {.primary_token_capacity = 100,
          .fallback_token_capacity = 80,
          .max_queue_requests = 1,
          .max_queue_tokens = 50,
          .degrade_utilization = 0.8,
          .initial_service_time = 100ms,
          .minimum_execution_budget = 10ms,
          .minimum_retry_after = 20ms};
}

TEST(AdmissionControllerTest, AdmitsWithinPrimaryTokenBudget) {
  AdmissionController controller(small_config());
  const auto now = SteadyClock::time_point{};
  const auto result = controller.admit(
      {30, 20}, now + 1s, true, {.primary = true, .fallback = true}, now);
  ASSERT_EQ(result.kind, AdmissionKind::admitted);
  ASSERT_TRUE(result.lease.has_value());
  EXPECT_EQ(result.lease->route, UpstreamRoute::primary);
  EXPECT_EQ(controller.snapshot().active_primary_tokens, 50U);

  controller.complete(*result.lease, 50ms);
  EXPECT_EQ(controller.snapshot().active_requests, 0U);
  EXPECT_EQ(controller.snapshot().service_time_ewma, 90ms);
}

TEST(AdmissionControllerTest, DegradesWhenPrimaryUtilizationCrossesThreshold) {
  AdmissionController controller(small_config());
  const auto now = SteadyClock::time_point{};
  const auto first = controller.admit({40, 40}, now + 1s, true,
                                      {.primary = true, .fallback = true}, now);
  ASSERT_TRUE(first.lease.has_value());
  EXPECT_EQ(first.lease->route, UpstreamRoute::primary);

  const auto second = controller.admit(
      {10, 10}, now + 1s, true, {.primary = true, .fallback = true}, now);
  ASSERT_TRUE(second.lease.has_value());
  EXPECT_EQ(second.lease->route, UpstreamRoute::fallback);
  EXPECT_EQ(controller.snapshot().active_fallback_tokens, 20U);
}

TEST(AdmissionControllerTest, BoundsQueueAndPromotesAfterCompletion) {
  AdmissionController controller(small_config());
  const auto now = SteadyClock::time_point{};
  const auto active =
      controller.admit({50, 50}, now + 2s, false, {.primary = true}, now);
  ASSERT_TRUE(active.lease.has_value());

  const auto queued =
      controller.admit({10, 10}, now + 2s, false, {.primary = true}, now);
  ASSERT_EQ(queued.kind, AdmissionKind::queued);
  ASSERT_TRUE(queued.ticket.has_value());
  EXPECT_EQ(controller.snapshot().queue_requests, 1U);

  const auto rejected =
      controller.admit({5, 5}, now + 2s, false, {.primary = true}, now);
  EXPECT_EQ(rejected.kind, AdmissionKind::rejected_overload);
  EXPECT_GE(rejected.retry_after, 20ms);
  EXPECT_EQ(controller.snapshot().max_observed_queue_requests, 1U);

  controller.complete(*active.lease, 100ms);
  const auto promoted = controller.poll(*queued.ticket, {.primary = true}, now);
  ASSERT_EQ(promoted.kind, AdmissionKind::admitted);
  ASSERT_TRUE(promoted.lease.has_value());
  EXPECT_EQ(promoted.lease->route, UpstreamRoute::primary);
  EXPECT_EQ(controller.snapshot().queue_requests, 0U);
  controller.complete(*promoted.lease);
}

TEST(AdmissionControllerTest, RejectsDeadlineThatCannotSurviveQueue) {
  AdmissionController controller(small_config());
  const auto now = SteadyClock::time_point{};
  const auto active =
      controller.admit({50, 50}, now + 1s, false, {.primary = true}, now);
  ASSERT_TRUE(active.lease.has_value());
  const auto rejected =
      controller.admit({10, 10}, now + 50ms, false, {.primary = true}, now);
  EXPECT_EQ(rejected.kind, AdmissionKind::rejected_deadline);
}

TEST(AdmissionControllerTest, RejectsOversizeAndUnavailableRequests) {
  AdmissionController controller(small_config());
  const auto now = SteadyClock::time_point{};
  EXPECT_EQ(
      controller.admit({100, 1}, now + 1s, false, {.primary = true}, now).kind,
      AdmissionKind::rejected_oversize);
  EXPECT_EQ(controller
                .admit({5, 5}, now + 1s, false,
                       {.primary = false, .fallback = false}, now)
                .kind,
            AdmissionKind::rejected_unavailable);
}

TEST(AdmissionControllerTest,
     CancelsQueuedAndReadyTicketsWithoutLeakingTokens) {
  AdmissionController controller(small_config());
  const auto now = SteadyClock::time_point{};
  const auto active =
      controller.admit({50, 50}, now + 2s, false, {.primary = true}, now);
  const auto queued =
      controller.admit({10, 10}, now + 2s, false, {.primary = true}, now);
  ASSERT_TRUE(active.lease.has_value());
  ASSERT_TRUE(queued.ticket.has_value());
  EXPECT_TRUE(controller.cancel(*queued.ticket));
  EXPECT_EQ(controller.snapshot().queue_tokens, 0U);
  controller.complete(*active.lease);
}

TEST(AdmissionControllerTest, ExpiresTicketsAtTheirDeadline) {
  AdmissionController controller(small_config());
  const auto now = SteadyClock::time_point{};
  const auto active =
      controller.admit({50, 50}, now + 2s, false, {.primary = true}, now);
  const auto queued =
      controller.admit({10, 10}, now + 250ms, false, {.primary = true}, now);
  ASSERT_TRUE(active.lease.has_value());
  ASSERT_TRUE(queued.ticket.has_value());
  const auto expired =
      controller.poll(*queued.ticket, {.primary = true}, now + 245ms);
  EXPECT_EQ(expired.kind, AdmissionKind::rejected_deadline);
  EXPECT_EQ(controller.snapshot().queue_requests, 0U);
  controller.complete(*active.lease);
}

TEST(AdmissionControllerTest, ConcurrentAdmissionsNeverExceedCapacity) {
  AdmissionConfig config = small_config();
  config.primary_token_capacity = 1000;
  config.max_queue_requests = 64;
  config.max_queue_tokens = 6400;
  config.degrade_utilization = 1.0;
  AdmissionController controller(config);
  std::vector<std::thread> workers;
  for (int worker = 0; worker < 8; ++worker) {
    workers.emplace_back([&controller] {
      for (int iteration = 0; iteration < 200; ++iteration) {
        const auto result = controller.admit({1, 1}, SteadyClock::now() + 5s,
                                             false, {.primary = true});
        if (result.lease) {
          controller.complete(*result.lease);
        } else if (result.ticket) {
          EXPECT_TRUE(controller.cancel(*result.ticket));
        }
      }
    });
  }
  for (auto &worker : workers) {
    worker.join();
  }
  const auto snapshot = controller.snapshot();
  EXPECT_EQ(snapshot.active_requests, 0U);
  EXPECT_EQ(snapshot.queue_requests, 0U);
  EXPECT_LE(snapshot.max_observed_queue_requests, config.max_queue_requests);
}

} // namespace
} // namespace frontier_forge
