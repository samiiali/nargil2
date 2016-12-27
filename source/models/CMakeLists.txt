project(models CXX)
FILE(GLOB_RECURSE TARGET_SRC "*.cpp")
add_custom_target(${PROJECT_NAME} SOURCES ${TARGET_SRC})

