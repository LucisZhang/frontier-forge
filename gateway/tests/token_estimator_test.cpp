#include <gtest/gtest.h>

#include <boost/json.hpp>

#include "frontier_forge/token_estimator.hpp"

namespace frontier_forge {
namespace {

TEST(TokenEstimatorTest, EstimatesOpenAIChatRequestWithoutTokenizer) {
  TokenEstimator estimator;
  const auto result = estimator.estimate(R"({
    "model":"primary-model",
    "messages":[{"role":"user","content":"abcdefgh"}],
    "max_completion_tokens":32,
    "stream":true
  })");
  ASSERT_TRUE(result.valid_json) << result.error;
  EXPECT_TRUE(result.stream);
  EXPECT_EQ(result.model, "primary-model");
  EXPECT_GE(result.cost.prompt_tokens, 9U);
  EXPECT_EQ(result.cost.output_tokens, 32U);
}

TEST(TokenEstimatorTest, HandlesUnicodeAndClampsOutputBudget) {
  TokenEstimator estimator({.default_output_tokens = 16,
                            .maximum_output_tokens = 64,
                            .maximum_prompt_tokens = 1024});
  const auto result = estimator.estimate(
      R"({"messages":[{"role":"user","content":"你好世界"}],"max_tokens":999})");
  ASSERT_TRUE(result.valid_json);
  EXPECT_EQ(result.cost.output_tokens, 64U);
  EXPECT_EQ(TokenEstimator::estimate_text_tokens("你好世界"), 1U);
}

TEST(TokenEstimatorTest, RejectsMalformedOrNonObjectJson) {
  TokenEstimator estimator;
  EXPECT_FALSE(estimator.estimate("{").valid_json);
  EXPECT_FALSE(estimator.estimate("[]").valid_json);
}

TEST(TokenEstimatorTest, RewritesOnlyTheModelForFallbackRoute) {
  TokenEstimator estimator;
  const auto rewritten = estimator.rewrite_model(
      R"({"model":"large","messages":[],"stream":false})", "small");
  const auto parsed = boost::json::parse(rewritten).as_object();
  EXPECT_EQ(parsed.at("model").as_string(), "small");
  EXPECT_FALSE(parsed.at("stream").as_bool());
}

} // namespace
} // namespace frontier_forge
