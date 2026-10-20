# Opt-in resident 1017 backend. No change to upstream targets when disabled.
if(NOT STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP OR STRATA_HIP_GFX906)
  message(FATAL_ERROR "The resident 1017 backend requires NVIDIA CUDA")
endif()
find_package(Threads REQUIRED)
add_library(strata_ds41_engine STATIC
  src/ds41/pack.cpp src/ds41/stage_transfer.cpp src/ds41/ops.cu
  src/ds41/exl3_linear.cu src/ds41/prefill_ops.cu src/ds41/doorbell.cu
  src/ds41/expert_staging.cu src/ds41/expert_prefetch.cu src/ds41/vram_experts.cu
  src/ds41/engram_rows.cpp src/ds41/host_experts.cpp src/ds41/lookahead.cpp
  src/ds41/expert_stream.cpp src/ds41/wo_a_fp8.cu src/ds41/dspark_ops.cu
  src/ds41/engine.cu src/ds41/resident_cpu_guard.cpp src/platform/direct_file.cpp
  src/ds41/kernels/fp8_gemv.cu src/ds41/kernels/k2_fp8_gemm.cu
  src/ds41/kernels/k3_sparse_attn.cu src/ds41/kernels/k5_indexer.cu
  src/ds41/kernels/k7_hc.cu src/ds41/kernels/k8_router.cu
  src/ds41/kernels/k10_exl3_moe.cu src/ds41/kernels/k12_exl3_moe_prefill.cu
  src/ds41/kernels/k13_sparse_attn_prefill.cu src/ds41/kernels/k14_indexer_prefill.cu
  src/ds41/kernels/k15_hc_prefill.cu)
target_include_directories(strata_ds41_engine PUBLIC include PRIVATE third_party/exllamav3_moe ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_compile_features(strata_ds41_engine PUBLIC cxx_std_17)
target_compile_definitions(strata_ds41_engine PRIVATE STRATA_DS41_1017=1)
target_compile_options(strata_ds41_engine PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:--ftz=false;--prec-div=true;--prec-sqrt=true>")
target_link_libraries(strata_ds41_engine PUBLIC CUDA::cudart CUDA::cublas Threads::Threads)

add_library(strata_ds41_suffix STATIC src/ds41/suffix_drafter.cpp)
target_include_directories(strata_ds41_suffix PUBLIC include)
target_compile_features(strata_ds41_suffix PUBLIC cxx_std_17)
add_library(ds41_serve_sampler STATIC src/kernels/cuda/sampler.cu)
target_include_directories(ds41_serve_sampler PRIVATE include)
target_link_libraries(ds41_serve_sampler PUBLIC CUDA::cudart)
foreach(program generate serve)
  add_executable(ds41_${program} src/ds41/ds41_${program}.cpp)
  set_target_properties(ds41_${program} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
  target_link_libraries(ds41_${program} PRIVATE strata_ds41_engine strata_ds41_suffix ds41_serve_sampler)
endforeach()

option(STRATA_DS41_STAGE_TESTS "Build resident 1017 component/model tests (never run automatically)" OFF)
if(STRATA_DS41_STAGE_TESTS)
  enable_testing()
  foreach(test stage_plan suffix_drafter serve_request)
    add_executable(ds41_${test}_test src/ds41/tests/${test}_test.cpp)
    target_link_libraries(ds41_${test}_test PRIVATE strata_ds41_suffix)
    add_test(NAME ds41_${test} COMMAND ds41_${test}_test)
  endforeach()
  add_executable(ds41_resident_policy_test src/ds41/tests/resident_policy_test.cpp)
  target_link_libraries(ds41_resident_policy_test PRIVATE strata_ds41_engine)
  add_test(NAME ds41_resident_policy COMMAND ds41_resident_policy_test)
  foreach(test stage_transfer stage_upload mcg_decode mcg_pack exl3_linear)
    add_executable(ds41_${test}_test src/ds41/tests/${test}_test.cu)
    target_link_libraries(ds41_${test}_test PRIVATE strata_ds41_engine)
    add_test(NAME ds41_${test} COMMAND ds41_${test}_test)
    set_tests_properties(ds41_${test} PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180)
  endforeach()
  add_executable(ds41_stage_expert_test src/ds41/tests/hybrid_decode_test.cu)
  target_link_libraries(ds41_stage_expert_test PRIVATE strata_ds41_engine)
  add_test(NAME ds41_stage_expert COMMAND ds41_stage_expert_test)
  set_tests_properties(ds41_stage_expert PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180)
  add_executable(ds41_exl3_prefill_experts_test src/ds41/tests/k12_exl3_moe_prefill_test.cu)
  target_link_libraries(ds41_exl3_prefill_experts_test PRIVATE strata_ds41_engine)
  add_test(NAME ds41_exl3_prefill_experts COMMAND ds41_exl3_prefill_experts_test --component-only)
  set_tests_properties(ds41_exl3_prefill_experts PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180)
  foreach(test stage_model_smoke stage_prefill_model stage_serving_model)
    add_executable(ds41_${test}_test src/ds41/tests/${test}_test.cpp)
    target_link_libraries(ds41_${test}_test PRIVATE strata_ds41_engine)
  endforeach()
endif()
