#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "frontier_forge/admission.hpp"

namespace frontier_forge {

struct TokenEstimatorConfig {
  std::uint64_t default_output_tokens{256};
  std::uint64_t maximum_output_tokens{8192};
  std::uint64_t maximum_prompt_tokens{131072};
};

struct OpenAIRequestEstimate {
  bool valid_json{};
  bool stream{};
  RequestCost cost;
  std::string model;
  std::string error;
};

class TokenEstimator {
public:
  explicit TokenEstimator(TokenEstimatorConfig config = {});

  [[nodiscard]] OpenAIRequestEstimate
  estimate(std::string_view request_body) const;
  [[nodiscard]] std::string rewrite_model(std::string_view request_body,
                                          std::string_view model) const;

  [[nodiscard]] static std::uint64_t
  estimate_text_tokens(std::string_view text) noexcept;

private:
  TokenEstimatorConfig config_;
};

} // namespace frontier_forge
