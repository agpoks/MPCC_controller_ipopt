cmake_minimum_required(VERSION 3.14)
project(MPCC_controller_cpp)

set(CMAKE_C_COMPILER   "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -Wall -Wextra -pthread")
set(CMAKE_CXX_FLAGS_DEBUG   "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# ── Optional HSL runtime path for direct MA57 execution ─────────────────────
# Priority:
#   1) env MPC_HSL_LIB_DIR
#   2) env HSL_LIB_DIR
#   3) $HOME/ThirdParty-HSL/.libs (common local setup)
set(HSL_RUNTIME_DIR "")
if(DEFINED ENV{MPC_HSL_LIB_DIR} AND NOT "$ENV{MPC_HSL_LIB_DIR}" STREQUAL "")
    set(HSL_RUNTIME_DIR "$ENV{MPC_HSL_LIB_DIR}")
elseif(DEFINED ENV{HSL_LIB_DIR} AND NOT "$ENV{HSL_LIB_DIR}" STREQUAL "")
    set(HSL_RUNTIME_DIR "$ENV{HSL_LIB_DIR}")
elseif(DEFINED ENV{HOME} AND EXISTS "$ENV{HOME}/ThirdParty-HSL/.libs")
    set(HSL_RUNTIME_DIR "$ENV{HOME}/ThirdParty-HSL/.libs")
endif()

# ── CasADi ────────────────────────────────────────────────────────────────────
set(CASADI_INCLUDE_DIR /usr/local/include/casadi)
find_library(CASADI_LIBRARY
    NAMES casadi
    HINTS /usr/local/lib $ENV{CASADI_PREFIX}/lib)
if(CASADI_LIBRARY)
    set(CASADI_LIBRARIES ${CASADI_LIBRARY})
else()
    message(FATAL_ERROR "CasADi library not found. Install it or set CASADI_PREFIX.")
endif()

# ── IPOPT (used by CasADi plugin; also linked directly for safety) ─────────
if(UNIX AND NOT APPLE)
    set(IPOPT_INCLUDE_DIR /usr/include/coin)
    find_library(IPOPT_LIBRARY NAMES ipopt HINTS /usr/local/lib /usr/lib)
elseif(APPLE)
    set(IPOPT_INCLUDE_DIR /usr/local/include/coin-or)
    find_library(IPOPT_LIBRARY NAMES ipopt HINTS /usr/local/lib)
endif()

# ── Threads ────────────────────────────────────────────────────────────────
find_package(Threads REQUIRED)

# ── Python + NumPy (for matplotlibcpp) ────────────────────────────────────
# Prefer system Python (/usr) to avoid broken virtual-env prefixes.
set(Python_ROOT_DIR /usr CACHE PATH "Force system Python")
find_package(Python COMPONENTS Interpreter Development NumPy REQUIRED)

# ── Runtime library search paths ───────────────────────────────────────────
link_directories(/usr/local/lib)

# ── Executable ────────────────────────────────────────────────────────────
add_executable(MPCC_controller_cpp
    main.cpp
    MPCC_controller.cpp
    utils.cpp
    rl_manager.cpp
    plotter.cpp
    # Headers listed for IDE awareness (not compiled)
    include/MPCC_controller.h
    include/MPCC_planner.h
    include/utils.h
    include/pid.h
    include/main.h
    include/rl_manager.h
    include/plotter.h
)

target_include_directories(MPCC_controller_cpp PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CASADI_INCLUDE_DIR}
    ${IPOPT_INCLUDE_DIR}
    ${Python_INCLUDE_DIRS}
    ${Python_NumPy_INCLUDE_DIRS}
)

target_link_libraries(MPCC_controller_cpp PRIVATE
    ${CASADI_LIBRARIES}
    $<$<BOOL:${IPOPT_LIBRARY}>:${IPOPT_LIBRARY}>
    Threads::Threads
    Python::Python
    Python::NumPy
)

if(HSL_RUNTIME_DIR AND EXISTS "${HSL_RUNTIME_DIR}")
    message(STATUS "HSL runtime directory detected: ${HSL_RUNTIME_DIR}")
    target_link_directories(MPCC_controller_cpp PRIVATE "${HSL_RUNTIME_DIR}")
    set_property(TARGET MPCC_controller_cpp APPEND PROPERTY BUILD_RPATH "${HSL_RUNTIME_DIR}")
    set_property(TARGET MPCC_controller_cpp APPEND PROPERTY INSTALL_RPATH "${HSL_RUNTIME_DIR}")
endif()
