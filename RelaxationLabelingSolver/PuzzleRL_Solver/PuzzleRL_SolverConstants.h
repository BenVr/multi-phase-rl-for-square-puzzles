#pragma once

#include <filesystem>
#include "PuzzleRL_SolverUtils.h"

/****************************************** Algorithm Parameters (Non-Technical) ******************************************/
constexpr DissimilarityType DISSIMILARITY_TYPE = DissimilarityType::eImprovedMGC;

/****************************************** Technical Parameters That May Affect Results ******************************************/
//if PRINT_PIECE_NUMBERS_IN_IMAGE true, label numbers will be printed on each piece
constexpr bool PRINT_PIECE_NUMBERS_IN_IMAGE = false;

/****************************************** Technical Parameters ******************************************/
constexpr bool OUTPUT_ALL_PIECES_SEPARATELY = false;

static constexpr bool SHOULD_LOG_SCREEN_OUTPUT_TO_FILE = true;

const std::filesystem::path CONFIG_XML_PATH = "./Resources/PuzzleRL_SolverConfigurations.xml";
const std::filesystem::path DATA_FOLDER_PATH = "./Resources/data/";

const std::filesystem::path SCREEN_OUTPUT_FILE = "screenOutput.txt";
const std::filesystem::path MULTIPLE_RUNS_TEXT_SUMMARY_FILE = "resultsSummary.txt";
const std::filesystem::path PRINT_ALL_PIECES_PATH = "pieces/";
const std::filesystem::path ITERATIONS_FOLDER_NAME = "iterations/";
const std::filesystem::path ORIGINAL_IMAGE_FILE_NAME = "originalImage.png";
const std::filesystem::path PUZZLE_IMAGE_FILE_NAME = "puzzleImage.png";
const std::filesystem::path SOLUTION_IMAGE_FILE_NAME = "solutionImage.png";