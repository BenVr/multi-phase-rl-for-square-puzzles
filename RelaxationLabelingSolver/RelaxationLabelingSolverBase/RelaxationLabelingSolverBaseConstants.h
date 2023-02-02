#pragma once

#include <filesystem>
#include <execution>

//SAVE_ALL_PAIRS_SUPPORT_COMPUTATION_METHOD is the faster method, but it's more memory heavy.
//In this method, for each (object i, label lambda) pair, we save all pairs that take part in the computation of its support value
//(take part means that corresponding compatibility value is not 0), meaning all the (object, label) pairs
//that take part in the computation of Si(lambda).
//We save here a lot of data - but it allows us to perform minimal number of iterations in each support value computation
#define SAVE_ALL_PAIRS_SUPPORT_COMPUTATION_METHOD 0

//STRAIGHTFORWARD_SUPPORT_COMPUTATION_METHOD is the simple inefficient straightforward way to compute support.
//This method should be used for debugging.
#define STRAIGHTFORWARD_SUPPORT_COMPUTATION_METHOD 1

const std::filesystem::path OUTPUT_DIRECTORY_PATH = "./Output/";

//For parallel run, define 'IS_PARALLEL_EXECUTION_POLICY' to true
//For sequential run, define 'IS_PARALLEL_EXECUTION_POLICY' to false
#define IS_PARALLEL_EXECUTION_POLICY true

//if OUTPUT_ONLY_NEW_CURRENT_SOLUTIONS == true -> iteration current solution are outputted only if current iteration solution is different 
//than the solution of the previous iteration
//if OUTPUT_ONLY_NEW_CURRENT_SOLUTIONS == false -> current solution is outputted for all iterations
constexpr bool OUTPUT_ONLY_NEW_CURRENT_SOLUTIONS = true;

//SUPPORT_COMPUTATION_METHOD determines the support calculation method.
//Although accompanied with cumbersome code, the SUPPORT_COMPUTATION_METHOD define allows us to alternate the support computation method
//as the two current methods are far from perfect 
#define SUPPORT_COMPUTATION_METHOD \
    SAVE_ALL_PAIRS_SUPPORT_COMPUTATION_METHOD

// If DO_NOT_COMPUTE_UNNECESSARY_SUPPORT_VALUES is true, support is computed more efficiently
constexpr bool DO_NOT_COMPUTE_UNNECESSARY_SUPPORT_VALUES = true;

namespace RelaxationLabelingSolverBaseConstants
{
    static constexpr double epsilon = 1.0e-4;
}

#if IS_PARALLEL_EXECUTION_POLICY
constexpr auto executionPolicy = std::execution::par_unseq;
#else
constexpr auto executionPolicy = std::execution::seq;
#endif

#define IS_SAVE_ALL_PAIRS_SUPPORT_COMPUTATION_METHOD() \
    (SUPPORT_COMPUTATION_METHOD == SAVE_ALL_PAIRS_SUPPORT_COMPUTATION_METHOD)

#define IS_STRAIGHTFORWARD_SUPPORT_COMPUTATION_METHOD() \
    (SUPPORT_COMPUTATION_METHOD == STRAIGHTFORWARD_SUPPORT_COMPUTATION_METHOD)