#include <gtest/gtest.h>

#include "frontier_forge/greeting.hpp"

TEST(GreetingTest, ReturnsProjectName) {
  EXPECT_EQ(frontier_forge::greeting(), "frontier-forge gateway");
}
