#include "frontier_forge/admission.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace frontier_forge {

namespace {

std::uint64_t saturating_add(std::uint64_t lhs, std::uint64_t rhs) {
  if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
    return std::numeric_limits<std::uint64_t>::max();
  }
  return lhs + rhs;
}

} // namespace

const char *to_string(UpstreamRoute route) noexcept {
  switch (route) {
  case UpstreamRoute::primary:
    return "primary";
  case UpstreamRoute::fallback:
    return "fallback";
  }
  return "unknown";
}

std::uint64_t RequestCost::total_tokens() const noexcept {
  return saturating_add(prompt_tokens, output_tokens);
}

const char *to_string(AdmissionKind kind) noexcept {
  switch (kind) {
  case AdmissionKind::admitted:
    return "admitted";
  case AdmissionKind::queued:
    return "queued";
  case AdmissionKind::rejected_overload:
    return "rejected_overload";
  case AdmissionKind::rejected_deadline:
    return "rejected_deadline";
  case AdmissionKind::rejected_oversize:
    return "rejected_oversize";
  case AdmissionKind::rejected_unavailable:
    return "rejected_unavailable";
  case AdmissionKind::cancelled:
    return "cancelled";
  }
  return "unknown";
}

AdmissionController::AdmissionController(AdmissionConfig config)
    : config_(std::move(config)), service_time_ewma_ms_(static_cast<double>(
                                      config_.initial_service_time.count())) {
  if (config_.primary_token_capacity == 0 ||
      config_.fallback_token_capacity == 0 || config_.max_queue_requests == 0 ||
      config_.max_queue_tokens == 0) {
    throw std::invalid_argument("admission capacities must be positive");
  }
  if (!(config_.degrade_utilization >= 0.0 &&
        config_.degrade_utilization <= 1.0)) {
    throw std::invalid_argument("degrade_utilization must be in [0, 1]");
  }
  if (config_.initial_service_time <= std::chrono::milliseconds::zero()) {
    throw std::invalid_argument("initial_service_time must be positive");
  }
}

AdmissionResult AdmissionController::admit(RequestCost cost,
                                           SteadyClock::time_point deadline,
                                           bool allow_degrade,
                                           RouteAvailability availability,
                                           SteadyClock::time_point now) {
  const auto tokens = cost.total_tokens();
  std::scoped_lock lock(mutex_);
  expire_locked(now);
  promote_locked(availability, now);

  if (tokens == 0 ||
      (tokens > config_.primary_token_capacity &&
       (!allow_degrade || tokens > config_.fallback_token_capacity))) {
    return {.kind = AdmissionKind::rejected_oversize,
            .retry_after = retry_after_locked(),
            .reason = "request token estimate exceeds every eligible route"};
  }
  if (deadline <= now + config_.minimum_execution_budget) {
    return {.kind = AdmissionKind::rejected_deadline,
            .reason = "request deadline leaves no execution budget"};
  }

  const double primary_load =
      static_cast<double>(
          saturating_add(active_primary_tokens_, queue_tokens_)) /
      static_cast<double>(config_.primary_token_capacity);
  const bool primary_fits =
      availability.primary && route_fits_locked(UpstreamRoute::primary, tokens);
  const bool should_degrade =
      !primary_fits || primary_load >= config_.degrade_utilization;

  if (allow_degrade && should_degrade && availability.fallback &&
      route_fits_locked(UpstreamRoute::fallback, tokens)) {
    return {.kind = AdmissionKind::admitted,
            .lease = make_lease_locked(UpstreamRoute::fallback, tokens),
            .reason = "degraded to fallback due to primary load"};
  }
  if (primary_fits) {
    return {.kind = AdmissionKind::admitted,
            .lease = make_lease_locked(UpstreamRoute::primary, tokens),
            .reason = "admitted to primary"};
  }

  if (!availability.primary) {
    return {.kind = AdmissionKind::rejected_unavailable,
            .retry_after = retry_after_locked(),
            .reason = "no healthy upstream route is available"};
  }

  const bool queue_full = queue_.size() >= config_.max_queue_requests ||
                          tokens > config_.max_queue_tokens - queue_tokens_;
  if (queue_full) {
    return {.kind = AdmissionKind::rejected_overload,
            .retry_after = retry_after_locked(),
            .reason = "bounded admission queue is full"};
  }

  const auto predicted_wait = predicted_wait_locked(tokens);
  if (deadline <= now + predicted_wait + config_.minimum_execution_budget) {
    return {.kind = AdmissionKind::rejected_deadline,
            .retry_after = retry_after_locked(),
            .reason = "predicted queue wait exceeds request deadline"};
  }

  const auto ticket = next_id_++;
  queue_.push_back({.id = ticket,
                    .tokens = tokens,
                    .deadline = deadline,
                    .allow_degrade = allow_degrade});
  queue_tokens_ += tokens;
  max_observed_queue_requests_ =
      std::max(max_observed_queue_requests_, queue_.size());
  max_observed_queue_tokens_ =
      std::max(max_observed_queue_tokens_, queue_tokens_);
  return {.kind = AdmissionKind::queued,
          .ticket = ticket,
          .retry_after = predicted_wait,
          .reason = "queued behind active token budget"};
}

AdmissionResult AdmissionController::poll(std::uint64_t ticket,
                                          RouteAvailability availability,
                                          SteadyClock::time_point now) {
  std::scoped_lock lock(mutex_);
  expire_locked(now);
  promote_locked(availability, now);

  if (auto found = ready_.find(ticket); found != ready_.end()) {
    auto lease = found->second;
    ready_.erase(found);
    return {.kind = AdmissionKind::admitted,
            .lease = lease,
            .reason = "queued request admitted"};
  }
  if (auto found = terminal_.find(ticket); found != terminal_.end()) {
    auto result = std::move(found->second);
    terminal_.erase(found);
    return result;
  }
  if (std::any_of(
          queue_.begin(), queue_.end(),
          [ticket](const QueueEntry &entry) { return entry.id == ticket; })) {
    return {.kind = AdmissionKind::queued,
            .ticket = ticket,
            .retry_after = retry_after_locked(),
            .reason = "request remains queued"};
  }
  return {.kind = AdmissionKind::cancelled,
          .reason = "ticket is no longer active"};
}

bool AdmissionController::cancel(std::uint64_t ticket) {
  std::scoped_lock lock(mutex_);
  for (auto it = queue_.begin(); it != queue_.end(); ++it) {
    if (it->id == ticket) {
      queue_tokens_ -= it->tokens;
      queue_.erase(it);
      terminal_.erase(ticket);
      return true;
    }
  }
  if (auto ready = ready_.find(ticket); ready != ready_.end()) {
    const auto lease = ready->second;
    if (lease.route == UpstreamRoute::primary) {
      active_primary_tokens_ -= lease.tokens;
    } else {
      active_fallback_tokens_ -= lease.tokens;
    }
    active_.erase(lease.id);
    ready_.erase(ready);
    return true;
  }
  return terminal_.erase(ticket) > 0;
}

void AdmissionController::complete(const AdmissionLease &lease,
                                   std::chrono::milliseconds service_time) {
  std::scoped_lock lock(mutex_);
  const auto found = active_.find(lease.id);
  if (found == active_.end()) {
    return;
  }
  const auto recorded = found->second;
  if (recorded.route == UpstreamRoute::primary) {
    active_primary_tokens_ -= recorded.tokens;
  } else {
    active_fallback_tokens_ -= recorded.tokens;
  }
  active_.erase(found);

  if (service_time > std::chrono::milliseconds::zero()) {
    constexpr double alpha = 0.20;
    service_time_ewma_ms_ = (1.0 - alpha) * service_time_ewma_ms_ +
                            alpha * static_cast<double>(service_time.count());
  }
}

AdmissionSnapshot AdmissionController::snapshot() const {
  std::scoped_lock lock(mutex_);
  return {.active_requests = active_.size(),
          .active_primary_tokens = active_primary_tokens_,
          .active_fallback_tokens = active_fallback_tokens_,
          .queue_requests = queue_.size(),
          .queue_tokens = queue_tokens_,
          .max_observed_queue_requests = max_observed_queue_requests_,
          .max_observed_queue_tokens = max_observed_queue_tokens_,
          .service_time_ewma = std::chrono::milliseconds(
              static_cast<std::int64_t>(std::llround(service_time_ewma_ms_)))};
}

AdmissionLease AdmissionController::make_lease_locked(UpstreamRoute route,
                                                      std::uint64_t tokens) {
  AdmissionLease lease{.id = next_id_++, .route = route, .tokens = tokens};
  if (route == UpstreamRoute::primary) {
    active_primary_tokens_ += tokens;
  } else {
    active_fallback_tokens_ += tokens;
  }
  active_.emplace(lease.id, lease);
  return lease;
}

void AdmissionController::promote_locked(RouteAvailability availability,
                                         SteadyClock::time_point now) {
  expire_locked(now);
  while (!queue_.empty()) {
    const auto &entry = queue_.front();
    std::optional<UpstreamRoute> route;
    if (availability.primary &&
        route_fits_locked(UpstreamRoute::primary, entry.tokens)) {
      route = UpstreamRoute::primary;
    } else if (entry.allow_degrade && availability.fallback &&
               route_fits_locked(UpstreamRoute::fallback, entry.tokens)) {
      route = UpstreamRoute::fallback;
    }
    if (!route.has_value()) {
      break;
    }
    const auto id = entry.id;
    const auto tokens = entry.tokens;
    queue_tokens_ -= tokens;
    queue_.pop_front();
    ready_.emplace(id, make_lease_locked(*route, tokens));
  }
}

void AdmissionController::expire_locked(SteadyClock::time_point now) {
  for (auto it = queue_.begin(); it != queue_.end();) {
    if (it->deadline > now + config_.minimum_execution_budget) {
      ++it;
      continue;
    }
    queue_tokens_ -= it->tokens;
    terminal_.emplace(
        it->id, AdmissionResult{.kind = AdmissionKind::rejected_deadline,
                                .reason = "request deadline expired in queue"});
    it = queue_.erase(it);
  }
}

std::chrono::milliseconds AdmissionController::predicted_wait_locked(
    std::uint64_t additional_tokens) const {
  const auto work = saturating_add(
      saturating_add(active_primary_tokens_, queue_tokens_), additional_tokens);
  const auto batches = std::max<std::uint64_t>(
      1, work / config_.primary_token_capacity +
             (work % config_.primary_token_capacity == 0 ? 0 : 1));
  const auto wait_ms = service_time_ewma_ms_ * static_cast<double>(batches);
  const auto bounded = std::min(
      wait_ms, static_cast<double>(std::numeric_limits<std::int64_t>::max()));
  return std::chrono::milliseconds(
      static_cast<std::int64_t>(std::ceil(bounded)));
}

std::chrono::milliseconds AdmissionController::retry_after_locked() const {
  const auto estimated = std::chrono::milliseconds(
      static_cast<std::int64_t>(std::ceil(service_time_ewma_ms_)));
  return std::max(config_.minimum_retry_after, estimated);
}

bool AdmissionController::route_fits_locked(UpstreamRoute route,
                                            std::uint64_t tokens) const {
  const auto active = route == UpstreamRoute::primary ? active_primary_tokens_
                                                      : active_fallback_tokens_;
  const auto capacity = route == UpstreamRoute::primary
                            ? config_.primary_token_capacity
                            : config_.fallback_token_capacity;
  return tokens <= capacity && active <= capacity - tokens;
}

} // namespace frontier_forge
