cmake_minimum_required(VERSION 3.25)
project(frontier_forge_gateway VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(forge_gateway_core src/greeting.cpp)
target_include_directories(forge_gateway_core PUBLIC include)
target_compile_options(forge_gateway_core PRIVATE -Wall -Wextra -Wpedantic)

add_executable(forge_gateway src/main.cpp)
target_link_libraries(forge_gateway PRIVATE forge_gateway_core)

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
