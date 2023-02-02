cmake_minimum_required(VERSION 3.16.4)

#1. Set variables
Set(PuzzleRLSolverSources
	PuzzleRL_Solver.cpp
	PuzzleRL_SolverOutputManager.cpp
	ImageUtils.cpp
	PuzzlePieceObject.cpp
	PuzzleImage.cpp
	LocationRotationLabel.cpp
	PuzzleRL_SolverConfiguration.cpp
	PuzzleRL_SolverUtils.cpp
	PiecesBlockTranslationManager.cpp
	PuzzleRL_SolverInputInfo.cpp)

Set(PuzzleRLSolverHeaders
	PuzzleRL_Solver.h
	PuzzleRL_SolverOutputManager.h
	PuzzleRL_SolverConstants.h
	ImageUtils.h
	PuzzlePieceObject.h
	PuzzleImage.h
	LocationRotationLabel.h
	PuzzleRL_SolverConfiguration.h
	PuzzleRL_SolverUtils.h
	PiecesBlockTranslationManager.h
	PuzzleRL_SolverInputInfo.h)

#2. Add macro (to be used in runtime)
add_definitions(-DPUZZLE_RL_SOLVER_DIR="${PuzzleRL_SolverDir}")

#3. Add and set properties for PuzzleRL_Solver executable
add_executable(PuzzleRL_Solver 
	main.cpp
	${PuzzleRLSolverSources} 
	${PuzzleRLSolverHeaders}
	Resources/PuzzleRL_SolverConfigurations.xml)

conan_target_link_libraries(PuzzleRL_Solver)

target_link_libraries(PuzzleRL_Solver
	RelaxationLabelingSolverBase)

target_include_directories(PuzzleRL_Solver PUBLIC ${RelaxationLabelingSolverBaseDir})