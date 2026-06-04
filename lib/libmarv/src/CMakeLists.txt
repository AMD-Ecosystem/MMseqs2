cmake_minimum_required(VERSION 3.18)

if(USE_HIP)
    project(CUDA_INCLUDE_DIRS LANGUAGES CXX)
else()
    project(CUDA_INCLUDE_DIRS LANGUAGES CXX CUDA)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

set(LIBRARY_ONLY 0 CACHE BOOL "Do not build executables")

if(USE_HIP)
    enable_language(HIP)
    if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES gfx90a)
    endif()
    find_package(hip REQUIRED)
    find_package(rocthrust REQUIRED)
endif()

set(NVCC_FLAGS)
list(APPEND NVCC_FLAGS --extended-lambda)
list(APPEND NVCC_FLAGS --expt-relaxed-constexpr)
list(APPEND NVCC_FLAGS -rdc=true)
string(TOLOWER "${CMAKE_BUILD_TYPE}" BUILD_TYPE_LOWER)
if(BUILD_TYPE_LOWER STREQUAL "debug" OR BUILD_TYPE_LOWER STREQUAL "relwithdebinfo")
    list(APPEND NVCC_FLAGS -lineinfo)
endif()

set(NVCC_FLAGS_OMP ${NVCC_FLAGS})
list(APPEND NVCC_FLAGS_OMP -Xcompiler -fopenmp)

set(ALIGN_SOURCES
    kernels.cuh
    blosum.cu
    pssm.cuh
    pssmkernels_gapless.cuh
    pssmkernels_gapless_instantiation_dpx.cu
    pssmkernels_gapless_instantiation_half2.cu
    pssmkernels_gapless_instantiation_dpx_kernelparamzero.cu
    pssmkernels_gapless_instantiation_half2_kernelparamzero.cu
    pssmkernels_smithwaterman.cuh
    pssmkernels_smithwaterman_instantiation_float.cu
    pssmkernels_smithwaterman_instantiation_dpx.cu
    dbdata.cpp
)

set(INT8_GAPLESS_OBJECTS "")
set(INT8_GAPLESS_DEFINITION "")
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "13.2")
    add_library(pssmkernels_gapless_int8_obj OBJECT pssmkernels_gapless_instantiation_int8.cu)
    set_target_properties(pssmkernels_gapless_int8_obj PROPERTIES
        CUDA_ARCHITECTURES "120f-real;121f-real"
    )
    target_compile_options(pssmkernels_gapless_int8_obj PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${NVCC_FLAGS}>
    )
    target_compile_definitions(pssmkernels_gapless_int8_obj PRIVATE CUDASW_INT8_GAPLESS)
    set(INT8_GAPLESS_OBJECTS $<TARGET_OBJECTS:pssmkernels_gapless_int8_obj>)
    set(INT8_GAPLESS_DEFINITION CUDASW_INT8_GAPLESS)
endif()

if(USE_HIP)
    # The aligner sources keep their CUDA spelling; cuda_to_hip.h (force
    # included) maps the runtime API and the SIMD intrinsics to ROCm/HIP, and
    # hip_compat/ redirects the <cuda_fp16.h> / <cooperative_groups*> includes.
    # Only the .cu / .cpp translation units are compiled; the .cuh / .h headers
    # in ALIGN_SOURCES must not become their own TUs (that would duplicate the
    # __constant__ definitions in kernels.cuh). Compile the .cu as HIP.
    set(MARV_HIP_SOURCES "")
    foreach(src ${ALIGN_SOURCES} marv.cu)
        get_filename_component(ext ${src} EXT)
        if(ext STREQUAL ".cu" OR ext STREQUAL ".cpp")
            set_source_files_properties(${src} PROPERTIES LANGUAGE HIP)
            list(APPEND MARV_HIP_SOURCES ${src})
        endif()
    endforeach()

    # Build as a shared library so the relocatable-device-code link happens here
    # (HIP_RESOLVE_DEVICE_SYMBOLS) and the consumer (mmseqs, a plain C++ link)
    # sees only host symbols and never needs the HIP device linker.
    add_library(marv SHARED ${MARV_HIP_SOURCES})
    set_target_properties(marv PROPERTIES
        HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
        HIP_SEPARABLE_COMPILATION ON
        HIP_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON
    )
    target_include_directories(marv PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/hip_compat
    )
    target_compile_options(marv PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:-fgpu-rdc;--offload-compress;-include;${CMAKE_CURRENT_SOURCE_DIR}/cuda_to_hip.h>
    )
    target_link_options(marv PRIVATE -fgpu-rdc --hip-link)
    target_compile_definitions(marv PRIVATE USE_HIP NO_NVTOOLSEXT)
    # Link the ROCm libraries PRIVATE so their HIP-language compile options
    # (--offload-arch) do not leak to the plain C++ consumers (mmseqs-framework).
    # marv resolves its device symbols internally (HIP_RESOLVE_DEVICE_SYMBOLS).
    target_link_libraries(marv PRIVATE roc::rocthrust hip::host)
else()
add_library(marv ${ALIGN_SOURCES} marv.cu marv.h ${INT8_GAPLESS_OBJECTS})
target_compile_options(marv PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${NVCC_FLAGS}>)
set_target_properties(marv
    PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_compile_definitions(marv PRIVATE NO_NVTOOLSEXT ${INT8_GAPLESS_DEFINITION})
endif()

if (NOT LIBRARY_ONLY)
    find_package(CUDAToolkit REQUIRED)
    find_package(ZLIB REQUIRED)
    find_package(OpenMP REQUIRED)

    add_executable(align ${ALIGN_SOURCES} main.cu options.cpp sequence_io.cpp ${INT8_GAPLESS_OBJECTS})
    target_compile_options(align PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${NVCC_FLAGS_OMP}>)
    target_compile_definitions(align PRIVATE ${INT8_GAPLESS_DEFINITION})
    target_link_libraries(align ZLIB::ZLIB CUDA::nvToolsExt OpenMP::OpenMP_CXX)
    set_target_properties(align PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    
    add_executable(makedb makedb.cpp sequence_io.cpp dbdata.cpp)
    target_compile_options(makedb PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${NVCC_FLAGS_OMP}>)
    target_link_libraries(makedb ZLIB::ZLIB OpenMP::OpenMP_CXX)
    set_target_properties(makedb PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
