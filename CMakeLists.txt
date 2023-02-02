cmake_minimum_required(VERSION 3.16.4)

# 1. Add project (=solution in Visual Studio)
project("RelaxationLabelingSolver" VERSION 1.0.0)

# 2. Set project's defintions

#If compiler is GNU (=g++), is should be g++-9 due to std::execution support issue with prior versions
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU") 
	set(CMAKE_CXX_COMPILER "g++-9")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "My multi config types" FORCE)
if (MSVC)
    # warning level 4 and all warnings as errors
    add_compile_options(/W4 /WX)
else()
    # lots of warnings and all warnings as errors
    add_compile_options(-Wall -Wextra -pedantic -Werror)
endif()

#_SILENCE_CXX17_ADAPTOR_TYPEDEFS_DEPRECATION_WARNING is defined to silence an eigen warning
add_definitions(-D_SILENCE_CXX17_ADAPTOR_TYPEDEFS_DEPRECATION_WARNING)

#Win32 includes different conan files than other platforms due to Eigen include path issue
#Moreover, in Win32 'NOMINMAX' defintion is needed to avoid errors with 'std::min' and 'std::max'
if (WIN32)
    include(conanbuildinfo_multi.cmake)
    add_definitions(-DNOMINMAX)
else()
    include(conanbuildinfo.cmake)
endif()

conan_basic_setup()

#3. Set variables
set(RelaxationLabelingSolverBaseDir ${CMAKE_SOURCE_DIR}/RelaxationLabelingSolver/RelaxationLabelingSolverBase)
set(PuzzleRL_SolverDir ${CMAKE_SOURCE_DIR}/RelaxationLabelingSolver/PuzzleRL_Solver)

#4. Add subdirectories
add_subdirectory(${RelaxationLabelingSolverBaseDir})
add_subdirectory(${PuzzleRL_SolverDir})