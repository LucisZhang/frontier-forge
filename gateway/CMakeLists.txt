cmake_minimum_required(VERSION 3.25)
project(frontier_forge_gateway VERSION 0.5.0 LANGUAGES CXX)

if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Boost 1.83 REQUIRED COMPONENTS json system)
find_package(Threads REQUIRED)

add_library(
  forge_gateway_core
  src/admission.cpp
  src/circuit_breaker.cpp
  src/config.cpp
  src/gateway_server.cpp
  src/metrics.cpp
  src/rate_limiter.cpp
  src/token_estimator.cpp
  src/upstream_pool.cpp)
target_include_directories(forge_gateway_core PUBLIC include)
target_compile_options(forge_gateway_core PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
target_compile_definitions(
  forge_gateway_core
  PUBLIC BOOST_ASIO_NO_DEPRECATED BOOST_BEAST_USE_STD_STRING_VIEW)
target_link_libraries(
  forge_gateway_core
  PUBLIC Boost::json Boost::system Threads::Threads)

add_executable(forge_gateway src/main.cpp)
target_link_libraries(forge_gateway PRIVATE forge_gateway_core)
target_compile_options(forge_gateway PRIVATE -Wall -Wextra -Wpedantic -Wconversion)

include(CTest)
if(BUILD_TESTING)
  include(FetchContent)
  find_package(GTest CONFIG QUIET)
  if(NOT GTest_FOUND)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG v1.15.2
      GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(googletest)
  endif()
  add_subdirectory(tests)
endif()
