#include "frontier_forge/token_estimator.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

#include <boost/json.hpp>

namespace frontier_forge {
namespace json = boost::json;

namespace {

std::uint64_t saturating_add(std::uint64_t lhs, std::uint64_t rhs) {
  if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
    return std::numeric_limits<std::uint64_t>::max();
  }
  return lhs + rhs;
}

std::uint64_t estimate_json_strings(const json::value &value) {
  if (value.is_string()) {
    return TokenEstimator::estimate_text_tokens(value.as_string());
  }
  std::uint64_t tokens = 0;
  if (value.is_array()) {
    for (const auto &item : value.as_array()) {
      tokens = saturating_add(tokens, estimate_json_strings(item));
    }
  } else if (value.is_object()) {
    for (const auto &item : value.as_object()) {
      tokens = saturating_add(tokens, estimate_json_strings(item.value()));
    }
  }
  return tokens;
}

std::uint64_t unsigned_integer(const json::value &value,
                               std::uint64_t fallback) {
  if (value.is_uint64()) {
    return value.as_uint64();
  }
  if (value.is_int64() && value.as_int64() >= 0) {
    return static_cast<std::uint64_t>(value.as_int64());
  }
  return fallback;
}

} // namespace

TokenEstimator::TokenEstimator(TokenEstimatorConfig config) : config_(config) {
  if (config_.default_output_tokens == 0 ||
      config_.maximum_output_tokens == 0 ||
      config_.maximum_prompt_tokens == 0 ||
      config_.default_output_tokens > config_.maximum_output_tokens) {
    throw std::invalid_argument("invalid token-estimator limits");
  }
}

OpenAIRequestEstimate
TokenEstimator::estimate(std::string_view request_body) const {
  boost::system::error_code error;
  auto document = json::parse(request_body, error);
  if (error || !document.is_object()) {
    return {.valid_json = false,
            .error =
                error ? error.message() : "request body must be an object"};
  }
  const auto &object = document.as_object();

  std::uint64_t prompt_tokens = 3;
  if (const auto *messages = object.if_contains("messages")) {
    prompt_tokens =
        saturating_add(prompt_tokens, estimate_json_strings(*messages));
    if (messages->is_array()) {
      prompt_tokens = saturating_add(
          prompt_tokens,
          static_cast<std::uint64_t>(messages->as_array().size()) * 4);
    }
  }
  if (const auto *prompt = object.if_contains("prompt")) {
    prompt_tokens =
        saturating_add(prompt_tokens, estimate_json_strings(*prompt));
  }
  if (const auto *input = object.if_contains("input")) {
    prompt_tokens =
        saturating_add(prompt_tokens, estimate_json_strings(*input));
  }
  if (const auto *tools = object.if_contains("tools")) {
    prompt_tokens =
        saturating_add(prompt_tokens, estimate_json_strings(*tools));
  }
  prompt_tokens = std::min(prompt_tokens, config_.maximum_prompt_tokens);

  std::uint64_t output_tokens = config_.default_output_tokens;
  if (const auto *maximum = object.if_contains("max_completion_tokens")) {
    output_tokens = unsigned_integer(*maximum, output_tokens);
  } else if (const auto *maximum = object.if_contains("max_tokens")) {
    output_tokens = unsigned_integer(*maximum, output_tokens);
  }
  output_tokens = std::clamp<std::uint64_t>(output_tokens, 1,
                                            config_.maximum_output_tokens);

  bool stream = false;
  if (const auto *stream_value = object.if_contains("stream");
      stream_value && stream_value->is_bool()) {
    stream = stream_value->as_bool();
  }
  std::string model;
  if (const auto *model_value = object.if_contains("model");
      model_value && model_value->is_string()) {
    model = model_value->as_string().c_str();
  }
  return {
      .valid_json = true,
      .stream = stream,
      .cost = {.prompt_tokens = prompt_tokens, .output_tokens = output_tokens},
      .model = std::move(model)};
}

std::string TokenEstimator::rewrite_model(std::string_view request_body,
                                          std::string_view model) const {
  boost::system::error_code error;
  auto document = json::parse(request_body, error);
  if (error || !document.is_object()) {
    throw std::invalid_argument("cannot rewrite model in invalid JSON request");
  }
  document.as_object()["model"] = model;
  return json::serialize(document);
}

std::uint64_t
TokenEstimator::estimate_text_tokens(std::string_view text) noexcept {
  std::uint64_t codepoints = 0;
  for (const char character : text) {
    const auto byte = static_cast<unsigned char>(character);
    if ((byte & 0xC0U) != 0x80U) {
      ++codepoints;
    }
  }
  return std::max<std::uint64_t>(1, (codepoints + 3) / 4);
}

} // namespace frontier_forge
