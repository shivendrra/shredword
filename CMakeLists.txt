cmake_minimum_required(VERSION 3.18)

# Force Ninja generator on Windows if using MinGW
if(WIN32 AND NOT DEFINED CMAKE_GENERATOR)
  set(CMAKE_GENERATOR "Ninja")
endif()

project(shredword-trainer LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# For MinGW on Windows
if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

file(GLOB_RECURSE CSRC_FILES "shredword/csrc/*.c" "shredword/csrc/*.cpp")
file(GLOB_RECURSE INC_FILES "shredword/inc/*.h" "shredword/inc/*.hpp")

message(STATUS "Found source files: ${CSRC_FILES}")
message(STATUS "Found header files: ${INC_FILES}")

if(NOT CSRC_FILES)
  message(FATAL_ERROR "No source files found in shredword/csrc/")
endif()

add_library(trainer SHARED ${CSRC_FILES})
target_link_libraries(trainer PRIVATE Python::Module)

if(WIN32)
  set_target_properties(trainer PROPERTIES SUFFIX ".pyd")
else()
  set_target_properties(trainer PROPERTIES PREFIX "lib")
endif()

install(TARGETS trainer DESTINATION shredword COMPONENT python_modules)
install(DIRECTORY shredword/ DESTINATION shredword COMPONENT python_modules FILES_MATCHING PATTERN "*.py")