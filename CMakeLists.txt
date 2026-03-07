cmake_minimum_required(VERSION 3.18)
project(geoxgb LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Compiler optimisation flags ───────────────────────────────────────────────
# Release mode gets -O2 by default; add architecture-specific flags, LTO,
# and fast-math (safe here: no NaN-dependent logic, finite-only inputs).
if(MSVC)
    # /fp:precise keeps IEEE compliance (no reordering); /fp:fast can flip
    # marginal coefficient signs in interpretability tests.
    # /Ob3: aggressive inlining.  /GL + /LTCG: whole-program LTO.
    # /arch:AVX2: 256-bit SIMD.
    add_compile_options(/O2 /Ob3 /GL /fp:precise /arch:AVX2 /DNDEBUG)
    add_link_options(/LTCG)
else()
    # -ffast-math safe on GCC/Clang: our code doesn't depend on NaN semantics
    # or strict associativity. Use -ffp-contract=fast for FMA without full
    # -ffast-math if tests prove sensitive on Linux CI.
    add_compile_options(-O3 -march=native -ffast-math -flto -DNDEBUG)
    add_link_options(-flto)
endif()

# ── OpenMP ────────────────────────────────────────────────────────────────────
find_package(OpenMP)

# ── Eigen ─────────────────────────────────────────────────────────────────────
find_package(Eigen3 3.4 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        Eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        3.4.0
        GIT_SHALLOW    TRUE
        EXCLUDE_FROM_ALL
    )
    set(EIGEN_BUILD_DOC   OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING     OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(Eigen)
endif()

# ── pybind11 ──────────────────────────────────────────────────────────────────
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v3.0.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# ── HVRT core (static) ────────────────────────────────────────────────────────
add_library(hvrt_core STATIC
    cpp/hvrt/src/whitener.cpp
    cpp/hvrt/src/binner.cpp
    cpp/hvrt/src/target.cpp
    cpp/hvrt/src/tree.cpp
    cpp/hvrt/src/reduce.cpp
    cpp/hvrt/src/expand.cpp
    cpp/hvrt/src/hvrt.cpp
)

target_include_directories(hvrt_core PUBLIC
    cpp/hvrt/include
    cpp/hvrt/third_party/pcg
)

target_link_libraries(hvrt_core PUBLIC Eigen3::Eigen)
if(OpenMP_CXX_FOUND)
    target_link_libraries(hvrt_core PUBLIC OpenMP::OpenMP_CXX)
endif()
set_target_properties(hvrt_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Suppress warnings from vendored HVRT source (not our code to fix)
if(MSVC)
    target_compile_options(hvrt_core PRIVATE /W0)
else()
    target_compile_options(hvrt_core PRIVATE -w)
endif()

# ── GeoXGB core (static) ──────────────────────────────────────────────────────
add_library(geoxgb_core STATIC
    cpp/src/noise.cpp
    cpp/src/geoxgb_base.cpp
    cpp/src/geoxgb_multiclass.cpp
)

target_include_directories(geoxgb_core PUBLIC cpp/include)
target_link_libraries(geoxgb_core PUBLIC hvrt_core)
set_target_properties(geoxgb_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ── Python extension ──────────────────────────────────────────────────────────
pybind11_add_module(_geoxgb_cpp cpp/python/bindings.cpp)
target_link_libraries(_geoxgb_cpp PRIVATE geoxgb_core)

# Install the extension alongside the Python source package
install(TARGETS _geoxgb_cpp DESTINATION geoxgb)
