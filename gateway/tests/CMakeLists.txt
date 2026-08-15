add_executable(gateway_tests gateway_test.cpp)
target_link_libraries(gateway_tests PRIVATE forge_gateway_core GTest::gtest_main)
target_compile_options(gateway_tests PRIVATE -Wall -Wextra -Wpedantic)

include(GoogleTest)
gtest_discover_tests(gateway_tests)
