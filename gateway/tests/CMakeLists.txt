add_library(mock_upstream_support mock_upstream/mock_upstream.cpp)
target_include_directories(mock_upstream_support PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mock_upstream_support PUBLIC forge_gateway_core)
target_compile_options(mock_upstream_support PRIVATE -Wall -Wextra -Wpedantic -Wconversion)

add_executable(mock_upstream mock_upstream/main.cpp)
target_link_libraries(mock_upstream PRIVATE mock_upstream_support)
target_compile_options(mock_upstream PRIVATE -Wall -Wextra -Wpedantic -Wconversion)

add_executable(
  gateway_tests
  admission_test.cpp
  circuit_breaker_test.cpp
  config_test.cpp
  integration_test.cpp
  metrics_test.cpp
  rate_limiter_test.cpp
  token_estimator_test.cpp)
target_link_libraries(
  gateway_tests
  PRIVATE mock_upstream_support GTest::gtest_main)
target_compile_options(gateway_tests PRIVATE -Wall -Wextra -Wpedantic -Wconversion)

include(GoogleTest)
gtest_discover_tests(gateway_tests)
