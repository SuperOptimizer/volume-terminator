cmake_minimum_required(VERSION 3.22 FATAL_ERROR)


list(PREPEND CMAKE_PREFIX_PATH "$ENV{HOME}/vc-dependencies")

project(VT VERSION 1.0)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

option(VC_USE_OPENMP "Enable OpenMP support" OFF)
set(CHUNK_CACHE_SIZE_GB "10" CACHE STRING "size of the chunk cache in gigabytes")
add_definitions(-DCHUNK_CACHE_SIZE_GB=${CHUNK_CACHE_SIZE_GB})

set(CMAKE_CXX_FLAGS " -Wall -Wextra -Weverything -march=native -std=c++23 -DWITH_BLOSC=1 -DWITH_ZLIB=1 -g3 -gdwarf-5 -funwind-tables -fasynchronous-unwind-tables -ferror-limit=1000 " )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++98-compat -Wno-shadow -Wno-exit-time-destructors -Wno-global-constructors -Wno-c++98-compat-pedantic -Wno-unused-parameter -Wno-padded -Wno-unused-const-variable -Wno-shorten-64-to-32 -Wno-float-conversion -Wno-implicit-int-float-conversion -Wno-c++11-narrowing -Wno-unsafe-buffer-usage -Wno-double-promotion -Wno-old-style-cast -Wno-sign-compare -Wno-implicit-float-conversion -Wno-missing-prototypes -Wno-sign-conversion -Wno-float-equal -Wno-unused-variable -Wno-misleading-indentation -Wno-source-uses-openmp -Wno-missing-noreturn -Wno-extra-semi -Wno-reserved-identifier -Wno-inconsistent-missing-destructor-override -Wno-newline-eof -Wno-implicit-int-conversion -Wno-reorder-ctor -Wno-unused-but-set-variable -Wno-implicit-int-conversion -Wno-suggest-destructor-override -Wno-logical-op-parentheses -Wno-missing-variable-declarations -Wno-extra-semi-stmt -Wno-c++20-compat -Wno-switch-default -Wno-suggest-override -Wno-deprecated-declarations -Wno-unused-value -Wno-weak-vtables -Wno-inconsistent-missing-override -Wno-zero-as-null-pointer-constant -Wno-covered-switch-default -Wno-switch-enum -Wno-tautological-unsigned-zero-compare -Wno-unused-exception-parameter -Wno-unused-macros -Wno-unused-template")

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}  -g3 -gdwarf-5 -rdynamic")

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -O0")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3  -flto")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto ")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -flto")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto ")
elseif(CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Os  -flto")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto ")
endif()


option(VC_ENABLE_ASAN "Enable AddressSanitizer for memory error detection" OFF)
option(VC_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
option(VC_ENABLE_TSAN "Enable ThreadSanitizer for data race detection" OFF)
option(VC_ENABLE_LSAN "Enable LeakSanitizer for memory leak detection (standalone)" OFF)

if(VC_ENABLE_ASAN AND VC_ENABLE_TSAN)
    message(FATAL_ERROR "AddressSanitizer and ThreadSanitizer cannot be used together")
endif()

if(VC_ENABLE_LSAN AND VC_ENABLE_ASAN)
    message(WARNING "LeakSanitizer is already included with AddressSanitizer. Disabling standalone LeakSanitizer.")
    set(VC_ENABLE_LSAN OFF)
endif()

if(VC_ENABLE_LSAN AND VC_ENABLE_TSAN)
    message(FATAL_ERROR "LeakSanitizer and ThreadSanitizer cannot be used together")
endif()

if(VC_ENABLE_ASAN)
    message(STATUS "AddressSanitizer enabled")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=address")

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DADDRESS_SANITIZER")
endif()

if(VC_ENABLE_UBSAN)
    message(STATUS "UndefinedBehaviorSanitizer enabled")
    set(UBSAN_FLAGS "-fsanitize=undefined -fno-sanitize-recover=all")

    set(UBSAN_FLAGS "${UBSAN_FLAGS} -fsanitize=float-divide-by-zero")
    set(UBSAN_FLAGS "${UBSAN_FLAGS} -fsanitize=float-cast-overflow")
    set(UBSAN_FLAGS "${UBSAN_FLAGS} -fsanitize=nullability")

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${UBSAN_FLAGS} -fno-omit-frame-pointer")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${UBSAN_FLAGS} -fno-omit-frame-pointer")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${UBSAN_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${UBSAN_FLAGS}")
endif()

if(VC_ENABLE_TSAN)
    message(STATUS "ThreadSanitizer enabled")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread -fno-omit-frame-pointer")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=thread -fno-omit-frame-pointer")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=thread")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=thread")

    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

if(VC_ENABLE_LSAN)
    message(STATUS "LeakSanitizer enabled (standalone)")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=leak -fno-omit-frame-pointer")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=leak -fno-omit-frame-pointer")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=leak")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=leak")
endif()

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)
set(XTENSOR_USE_XSIMD 1)

option(VC_WITH_CUDA_SPARSE "use cudss" ON)
if (VC_WITH_CUDA_SPARSE)
    add_definitions(-DVC_USE_CUDA_SPARSE=1)
endif()

if(VC_USE_OPENMP)
    message(STATUS "OpenMP support enabled")

    find_package(OpenMP REQUIRED)
    set(XTENSOR_USE_OPENMP 1)
else()
    message(STATUS "OpenMP support disabled")

    set(XTENSOR_USE_OPENMP 0)
    include_directories(${CMAKE_SOURCE_DIR}/core/openmp_stub)
    add_library(openmp_stub INTERFACE)
    add_library(OpenMP::OpenMP_CXX ALIAS openmp_stub)
    add_library(OpenMP::OpenMP_C ALIAS openmp_stub)
endif()

include_directories(core/src)

find_package(Qt6  REQUIRED COMPONENTS Widgets Gui Core Network)
find_package(xtensor REQUIRED)
find_package(spdlog  REQUIRED)
find_package(CURL REQUIRED)
find_package(Boost REQUIRED COMPONENTS system program_options)
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui videoio calib3d)
find_package(z5 REQUIRED)

add_library(vc_core
        core/src/Volume.cpp
        core/src/Segmentation.cpp
        core/src/VolumePkg.cpp
        core/src/Logging.cpp
        core/src/Slicing.cpp
        core/src/Surface.cpp
        core/src/SurfaceHelpers.cpp
        core/src/SurfaceVoxelizer.cpp
)

target_link_libraries(vc_core
        ${OpenCV_LIBS}
        spdlog::spdlog
        nlohmann_json::nlohmann_json
        z5
        CURL::libcurl
        OpenMP::OpenMP_CXX
        Eigen3::Eigen
        Ceres::ceres
)

add_library(vcui
        core/src/surface_metrics.cpp
        core/src/VCCollection.cpp
)

target_link_libraries(vcui vc_core Qt6::Core Qt6::Gui Qt6::Widgets)

add_executable(VT
        apps/VT/VCAppMain.cpp
        apps/VT/CWindow.cpp
        apps/VT/CVolumeViewer.cpp
        apps/VT/CVolumeViewerView.cpp
        apps/VT/SettingsDialog.cpp
        apps/VT/CSurfaceCollection.cpp
        apps/VT/OpChain.cpp
        apps/VT/OpsList.cpp
        apps/VT/OpsSettings.cpp
        apps/VT/formsetsrc.cpp
        apps/VT/SurfaceTreeWidget.hpp
        apps/VT/SurfaceTreeWidget.cpp
        apps/VT/SeedingWidget.cpp
        apps/VT/SeedingWidget.hpp
        apps/VT/DrawingWidget.cpp
        apps/VT/DrawingWidget.hpp
        apps/VT/PathData.cpp
        apps/VT/PathData.hpp
        apps/VT/CSegmentationEditorWindow.cpp
        apps/VT/CSegmentationEditorWindow.hpp
        apps/VT/SegmentRenderThread.cpp
        apps/VT/SegmentRenderThread.hpp
        apps/VT/ProgressUtil.cpp
        apps/VT/ProgressUtil.hpp
        apps/VT/CommandLineToolRunner.cpp
        apps/VT/CommandLineToolRunner.hpp
        apps/VT/ConsoleOutputWidget.cpp
        apps/VT/ConsoleOutputWidget.hpp
        apps/VT/CPointCollectionWidget.cpp
        apps/VT/COutlinedTextItem.hpp
        apps/VT/COutlinedTextItem.cpp)

target_include_directories(VT PUBLIC apps/VT)

target_link_libraries(VT
        vc_core
        vcui
        Qt6::Core
        Qt6::Gui
        Qt6::Widgets
        z5
        OpenMP::OpenMP_CXX
        Ceres::ceres
)


add_executable(vc_render_video apps/vc_render_video.cpp)
target_link_libraries(vc_render_video vc_core)

add_executable(vc_render_tifxyz apps/vc_render_tifxyz.cpp)
target_link_libraries(vc_render_tifxyz vc_core Boost::program_options)

add_executable(vc_objrefine apps/vc_objrefine.cpp)
target_link_libraries(vc_objrefine vc_core)

add_executable(vc_grow_seg_from_seed apps/vc_grow_seg_from_seed.cpp)
target_link_libraries(vc_grow_seg_from_seed vc_core)

add_executable(vc_seg_add_overlap apps/vc_seg_add_overlap.cpp)
target_link_libraries(vc_seg_add_overlap vc_core)

add_executable(vc_grow_seg_from_segments apps/vc_grow_seg_from_segments.cpp)
target_link_libraries(vc_grow_seg_from_segments vc_core)

add_executable(vc_tifxyz2obj apps/vc_tifxyz2obj.cpp)
target_link_libraries(vc_tifxyz2obj vc_core)

add_executable(vc_obj2tifxyz apps/vc_obj2tifxyz.cpp)
target_link_libraries(vc_obj2tifxyz vc_core)

add_executable(vc_tifxyz_inp_mask apps/vc_tifxyz_inp_mask.cpp)
target_link_libraries(vc_tifxyz_inp_mask vc_core)

add_executable(vc_tifxyz_winding apps/vc_tifxyz_winding.cpp)
target_link_libraries(vc_tifxyz_winding vc_core)

add_executable(vc_fill_quadmesh apps/vc_fill_quadmesh.cpp)
target_link_libraries(vc_fill_quadmesh vc_core)

add_executable(vc_tiffxyz_upscale_grounding apps/vc_tiffxyz_upscale_grounding.cpp)
target_link_libraries(vc_tiffxyz_upscale_grounding vc_core)

add_executable(vc_calc_surface_metrics apps/vc_calc_surface_metrics.cpp)
target_link_libraries(vc_calc_surface_metrics vc_core vcui Boost::program_options)
