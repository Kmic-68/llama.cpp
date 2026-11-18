# Builds the all-cards split engine (engine/, Strata's src/split ported to the Tesla P100) on its own, without the
# rest of the Strata tree: one executable, strata-split-2, for two sm_60 cards. Use build.sh, which fetches the pinned
# llama.cpp headers and passes the values below; the same flags built the production binary (see README, "Build").
#
#   -DSTRATA_GGML_DIR=<llama.cpp checkout at 3cf03257f219afbe7334045ff7c6a06ac68c627d>   (required: ggml-cuda headers)
#   -DCMAKE_CUDA_ARCHITECTURES=60 -DSTRATA_SPLIT_ARCH=60 -DSTRATA_SPLIT_CARDS=2 -DSTRATA_SPLIT_PF_JOB=64   (the defaults here)
#   -DSTRATA_SPLIT_PF_JOB=96 (what build.sh passes; PF_JOB=64 bash build.sh passes 64): our production for all three models since
#   17 Nov 2026, long prompts read faster (README)
cmake_minimum_required(VERSION 3.22)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 60)
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "build type" FORCE)
endif()
project(p100_flash_next_engine LANGUAGES CXX CUDA)

# what Strata's top-level CMakeLists sets before it adds src/split (CUDA only: the HIP branch needs the full Strata tree)
set(STRATA_ENABLE_CUDA ON)
set(STRATA_ENABLE_HIP OFF)
set(STRATA_BUILD_SPLIT ON)
option(STRATA_BUILD_TESTS "also build sFNX's CPU unit tests and test drivers (not run for this release)" OFF)
set(STRATA_SPLIT_ARCH "60" CACHE STRING "compute capability the engine is built for (Tesla P100: 60)")
set(STRATA_SPLIT_CARDS "2" CACHE STRING "card counts to build (only 2 has been run on P100)")
set(STRATA_SPLIT_PF_JOB "64" CACHE STRING "rows of one expert per prompt-path job (build.sh passes 96, our production for all three models)")
set(STRATA_GGML_DIR "" CACHE PATH "llama.cpp checkout at the pinned commit (its ggml-cuda headers)")

if(NOT STRATA_GGML_DIR OR NOT EXISTS "${STRATA_GGML_DIR}/ggml/src/ggml-cuda/vecdotq.cuh")
  message(FATAL_ERROR "-DSTRATA_GGML_DIR=<llama.cpp checkout at 3cf03257f219afbe7334045ff7c6a06ac68c627d> is required "
                      "(build.sh fetches it); got '${STRATA_GGML_DIR}'")
endif()
if(NOT STRATA_SPLIT_ARCH STREQUAL "60")
  message(WARNING "STRATA_SPLIT_ARCH=${STRATA_SPLIT_ARCH}: this release has only been built and run for sm_60 (Tesla P100)")
endif()

if(STRATA_BUILD_TESTS)
  enable_testing()
endif()
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/engine split)

# p100_compat.h is force-included in every CUDA source (sm_60 has no dp4a or nanosleep), as in the production build
foreach(_t strata-split-2 strata-split-3 strata-split-4 split_fnx_run split_moe_tile_test)
  if(TARGET ${_t})
    target_compile_options(${_t} PRIVATE
                           "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-include ${CMAKE_CURRENT_SOURCE_DIR}/engine/p100_compat.h>")
  endif()
endforeach()
