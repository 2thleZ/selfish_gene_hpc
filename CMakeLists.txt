cmake_minimum_required(VERSION 3.18)
project(SelfishGeneHpc LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# GTX 1650 is Compute 7.5
set(CMAKE_CUDA_ARCHITECTURES 75)

find_package(CUDAToolkit REQUIRED)

include_directories(src)

add_executable(selfish_gene
    src/main.cpp
    src/gene_sim.cu
)

target_link_libraries(selfish_gene PRIVATE CUDA::cudart CUDA::curand)

if(UNIX)
    target_compile_options(selfish_gene PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
endif()
