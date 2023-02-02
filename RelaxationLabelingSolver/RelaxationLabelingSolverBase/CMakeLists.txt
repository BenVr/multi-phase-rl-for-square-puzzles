cmake_minimum_required(VERSION 3.16.4)

#1. Set variables
Set(RelaxationLabelingSolverBaseSources
	RelaxationLabelingSolverBase.cpp
	Label.cpp
	Object.cpp
	Labeling.cpp
	FileSystemUtils.cpp
	Utilities.cpp
	XML_Configuration.cpp
	RelaxationLabelingSolverBaseUtils.cpp)

Set(RelaxationLabelingSolverBaseHeaders
	RelaxationLabelingSolverBase.h
	Label.h
	Object.h
	Labeling.h
	FileSystemUtils.h
	Utilities.h
	XML_Configuration.h
	RelaxationLabelingSolverBaseConstants.h
	RelaxationLabelingSolverBaseUtils.h)

#2. Add macro (to be used in runtime)
add_definitions(-DRELAXATION_LABELING_SOLVER_BASE_DIR="${RelaxationLabelingSolverBaseDir}")

#3. Add and set properties for RelaxationLabelingSolverBase library
add_library(RelaxationLabelingSolverBase 
	${RelaxationLabelingSolverBaseSources}
	${RelaxationLabelingSolverBaseHeaders})

conan_target_link_libraries(RelaxationLabelingSolverBase)