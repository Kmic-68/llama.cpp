set(TARGET llama-vision-bench)
add_executable(${TARGET} vision-bench.cpp)
target_link_libraries(${TARGET} PRIVATE mtmd llama ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
