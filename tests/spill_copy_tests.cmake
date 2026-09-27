# Link-injected CPU tests: compile real plane implementations, discard unrelated
# CUDA functions, and never link the CUDA runtime or require a GPU. fork checks
# intentional exit(70) in a child. Keep assertions enabled in Release builds.
if(UNIX AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  function(ninfer_add_spill_copy_test name)
    add_executable(${name} ${ARGN})
    target_compile_features(${name} PRIVATE cxx_std_20)
    target_include_directories(${name} PRIVATE
      ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src
      ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/third_party
      ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export
      ${CUDAToolkit_INCLUDE_DIRS})
    target_compile_options(${name} PRIVATE -UNDEBUG -ffunction-sections -fdata-sections)
    target_link_options(${name} PRIVATE -Wl,--gc-sections)
    add_test(NAME ${name} COMMAND ${name})
  endfunction()
  set(spill_core ${PROJECT_SOURCE_DIR}/src/core)
  ninfer_add_spill_copy_test(ninfer_spill_copy_test
    ${CMAKE_CURRENT_LIST_DIR}/spill_copy_test.cpp)
  ninfer_add_spill_copy_test(ninfer_spill_copy_planes_test
    ${CMAKE_CURRENT_LIST_DIR}/spill_copy_planes_test.cpp
    ${spill_core}/paged_kv_cache.cpp ${spill_core}/layout.cpp
    ${spill_core}/tensor.cpp ${spill_core}/dtype.cpp)
  ninfer_add_spill_copy_test(ninfer_spill_copy_state_planes_test
    ${CMAKE_CURRENT_LIST_DIR}/spill_copy_state_planes_test.cpp
    ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/impl/state/state_image.cpp
    ${spill_core}/linear_attention_state.cpp ${spill_core}/cyclic_kv_cache.cpp
    ${spill_core}/layout.cpp ${spill_core}/tensor.cpp ${spill_core}/dtype.cpp)
  ninfer_add_spill_copy_test(ninfer_spill_state_scratch_test
    ${CMAKE_CURRENT_LIST_DIR}/spill_state_scratch_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/spill_state_scratch_arena.cpp
    ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/impl/state/state_image.cpp
    ${spill_core}/linear_attention_state.cpp ${spill_core}/cyclic_kv_cache.cpp
    ${spill_core}/layout.cpp ${spill_core}/tensor.cpp ${spill_core}/dtype.cpp
    ${spill_core}/disk_kv_bridge.cpp ${spill_core}/disk_kv_store.cpp)
  target_link_libraries(ninfer_spill_state_scratch_test PRIVATE Threads::Threads)
  ninfer_add_spill_copy_test(ninfer_proactive_closure_test
    ${CMAKE_CURRENT_LIST_DIR}/proactive_closure_test.cpp
    ${spill_core}/disk_kv_bridge.cpp ${spill_core}/disk_kv_store.cpp)
  target_link_libraries(ninfer_proactive_closure_test PRIVATE Threads::Threads)
endif()
