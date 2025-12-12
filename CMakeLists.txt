project(gpt_inference_engine VERSION 1.0 LANGUAGES CXX)

cmake_minimum_required(VERSION 3.15)

#setting compiler options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -02")

include_directories(include)

set(
    SOURCES
    main.cpp
    src/model.cpp
    src/bpe.cpp
    src/load_gpt2.cpp
    src/load_gpt2_quant.cpp
    src/ops.cpp
    src/model_quant.cpp
)

add_executable(gpt_inference_engine ${SOURCES})