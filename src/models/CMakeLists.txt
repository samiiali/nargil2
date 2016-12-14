project(models CXX)
add_custom_target(${PROJECT_NAME} SOURCES
                  generic_model.hpp
                  generic_model.cpp
                  hybridized_models/implicit_numbering.hpp
                  hybridized_models/implicit_numbering.cpp)

