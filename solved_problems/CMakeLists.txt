project(solved_problem CXX)
FILE(GLOB_RECURSE TARGET_SRC "*.cpp" "*.hpp")
add_custom_target(${PROJECT_NAME} SOURCES ${TARGET_SRC})
