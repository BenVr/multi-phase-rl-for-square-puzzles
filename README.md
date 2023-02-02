# Multi-Phase Relaxation Labeling for Square Jigsaw Puzzle Solving
## Table of Contents

  * [Introduction](#introduction)
  * [Installation](#installation)
    + [General](#general)
    + [Installation Steps](#installation-steps)
  * [Running the Solver](#running-the-solver)
    + [Program Output](#program-output)
    + [Changing Program Parameters](#changing-program-parameters)
    
## Introduction
This repository contains the C++ implmentation of the square jigsaw puzzle solver suggested in the "Multi-Phase Relaxation Labeling for Square Jigsaw Puzzle Solving" paper. Please see [the project webpage](https://icvl.cs.bgu.ac.il/multi-phase-rl-for-jigsaw-puzzle-solving/) for more details.

## Installation

### General
Since package support is not built-in into C++, and since the code uses some external libraries (like OpenCV and Eigen), there is a need to use C++ package manager. More specifically, we use *Conan* as a C++ package manager. To allow convienent use of *Conan*, *CMake* is also in use. Although *Conan* and *CMake* allow building on various platforms and IDEs, we note here that the code was developed and tested on Windows and Visual Studio IDE only.<br/>

### Installation Steps

**1.** Clone the repository into local machine.<br/>
**2.** [Download *Conan*](https://conan.io/).<br/>
**3.** [Download *Cmake*](https://cmake.org/).<br/>
**4.** Open command line and cd into the main directory of the cloned repository.<br/>
**5.** In this step, we wish to install the dependent packages with *Conan*. Installing is done separtely for release and debug versions of the packages (please note that packages may take few GBs of memory).
   - The command for release packages installation is:

    conan install . --build=missing -g cmake_multi -s build_type=Release
            
   - The command for debug packages installation is:

    conan install . --build=missing -g cmake_multi -s build_type=Debug
            
**6.** After all the packages were installed by *Conan*, we are ready to use *Cmake*.<br/>
   - Still in the repository's main directory, type:
   
    mkdir build && cd build
            
   - To produce *CMake* products, type:
     
    cmake ..

**7.** Now 'build' directory contains all the files required for building the program exectubale.<br/>
On Windows, the 'build' directory will contain a Visual Studio solution named 'RelaxationLabelingSolver.sln' - and now you can open this solution, set the 'PuzzleRL_Solver' project as startup project, and building the solver.

## Running the Solver

### Program Output
The solver can be run for a single image, for a whole folder, or for a single image/whole folder more than once.

Running the solver for a single image once will output some logs to screen, and will output few files located in 'RelaxationLabelingSolver\PuzzleRL_Solver\Output' directory, incuding:
1. 'iterations' directory - contains visualizations for each relaxation labeling iteration 'current solution'. Iterations' solutions after translating the pieces block will be in separate folders (e.g., 'right dilemma - right translation')
2. 'originalImage.png' - the original image
3. 'puzzleImage.png' - the puzzle what was created from the original image
4. 'solutionImage.png' - the final solution
5. 'screenOutput.txt' - all screen output in txt file

The screen logs include information about each iteration, and in the end of the run - a short summary with run stats is printed to screen.

For running on a whole folder or running for an image/folder more than one time, separate folders will be created for each puzzle, each with the information detailed above for a single puzzle.

### Changing Program Parameters
Commonly-changed program parameters are read from the following XML file: 'RelaxationLabelingSolver\PuzzleRL_Solver\Resources\PuzzleRL_SolverConfigurations.xml'.
Therefore, changing parameters such as piece size, does not involve recompilinig the executable, but just changing the XML resource.
Below is a documentation of the parameters in XML file:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<Configurations>
  <RunParameters>
    <ImageInput>
      <!--1. PieceSize: puzzle pieces size-->
      <PieceSize>28</PieceSize>

      <!--2. One of 'SourceImageName' and 'RunAllFolderName' must be defined-->
      <!--SourceImageName: source image path (should be relative to 'RelaxationLabelingSolver\PuzzleRL_Solver\Resources\data\')
      In this case, the algorithm will solve a puzzle made from a single image ('MIT_dataset/5.png' in this example)-->
      <!--<SourceImageName>MIT_dataset/5.png</SourceImageName>-->

      <!--RunAllFolderName: run folder path (relative to 'RelaxationLabelingSolver\PuzzleRL_Solver\Resources\data\')
      In this case, the algorithm will solve puzzles made from all images in the folder-->
      <RunAllFolderName>MIT_dataset</RunAllFolderName>
    </ImageInput>

    <!--3. PuzzleType: puzzle type (1 or 2)-->
    <PuzzleType>1</PuzzleType>

    <!--4. NumOfRuns: the number of runs for each puzzle (can be more than 1)--> 
    <NumOfRuns>1</NumOfRuns>
  
    <SolverParameters>
      <!--5. MinThresholdForPieceCompatibility: a compatibility value under which all values are set to zero-->  
      <MinThresholdForPieceCompatibility Value="1e-8"/>
    </SolverParameters>
  </RunParameters>

  <TechnicalParameters>
    <!--6. IterationsFrequencyOfImagePrintsDuringAlg: how often to print iterations' solutions visualizations--> 
    <IterationsFrequencyOfImagePrintsDuringAlg>200</IterationsFrequencyOfImagePrintsDuringAlg>
  </TechnicalParameters>
</Configurations>
```
