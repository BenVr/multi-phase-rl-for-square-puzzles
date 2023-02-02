#pragma once

//Disable all warnings coming from Eigen Tensor module
#pragma warning(push, 0)
#include <unsupported/Eigen/CXX11/Tensor>
#pragma warning(pop)

#include <string>
#include <chrono>
#include <Eigen/Core>
#include "Utilities.h"
#include "RelaxationLabelingSolverBaseUtils.h"

using PiecesTensorType = Eigen::Tensor<double, 3>;

enum class PuzzleType
{
    eType1_NoRotation = 1,
    eType2_UknownRotation = 2
};

enum class DissimilarityType
{
    eInvalidDissimilarityType = -1,
    eImprovedMGC, //Taken from "Solving Square Jigsaw Puzzle by Hierarchical Loop Constraints" (Son et. al)
    eNumOfDissimilarityTypes
};

//All these degrees represent clockwise rotation
//The enums must be arranged in the following order: 0, 90, 180, 270
enum ImageRotation
{
    eInvalidRotation = -1,
    e_0_degrees = 0,
    e_90_degrees,
    e_180_degrees,
    e_270_degrees,
    eNumOfImageRotations
};

using RotationsVec = std::vector<ImageRotation>;

enum class TranslationMode
{
    eInvalidTranslationMode = -1,
    eDidNoTranslation = 0,
    eDidTranslation,
    eVerticalTranslationUpDilemma,
    eVerticalTranslationDownDilemma,
    eHorizontalTranslationLeftDilemma,
    eHorizontalTranslationRightDilemma,
    eNumOfTranslationModes
};

enum class TranslationDecision
{
    eInvalidTranslationDecision = -1,
    eNoTranslation = 0,
    eDoTranslationUp,
    eDoTranslationDown,
    eDoTranslationLeft,
    eDoTranslationRight,
    eNumOfTranslationDecisions
};

struct RL_puzzleSolverAlgorithmRunningInfo : public RL_algorithmRunningInfo
{
    RL_puzzleSolverAlgorithmRunningInfo(const size_t& in_numOfRows = 0) : RL_algorithmRunningInfo(in_numOfRows) {}
    RL_puzzleSolverAlgorithmRunningInfo(const RL_puzzleSolverAlgorithmRunningInfo& in_other) = default;

    void Set(const RL_algorithmRunningInfo& in_other) override
        {operator=(static_cast<const RL_puzzleSolverAlgorithmRunningInfo&>(in_other));}

    void UpdateTranslationDecision(const TranslationDecision& in_transDecision);

    TranslationDecision m_firstLevelTransDecision = TranslationDecision::eInvalidTranslationDecision;
    TranslationDecision m_secondLevelTransDecision = TranslationDecision::eInvalidTranslationDecision;
    int32_t m_translationDilemmaLevel = 0;

protected:
    RL_puzzleSolverAlgorithmRunningInfo& operator=(const RL_puzzleSolverAlgorithmRunningInfo& in_other);
};

struct PieceNumberAndRotation
{
    PieceNumberAndRotation() : m_pieceNumber(-1), m_rotation(ImageRotation::eInvalidRotation) {};
    PieceNumberAndRotation(const int32_t in_pieceNumber, const ImageRotation in_rotation)
        : m_pieceNumber(in_pieceNumber), m_rotation(in_rotation) {}

    PieceNumberAndRotation& operator=(const PieceNumberAndRotation&) = default;
    bool operator==(const PieceNumberAndRotation& in_other) const 
    {return m_pieceNumber == in_other.m_pieceNumber && m_rotation == in_other.m_rotation;}

    bool IsValid() const { return m_pieceNumber >= 0 && ImageRotation::eInvalidRotation != m_rotation; }
    std::string GetString() const;

    int32_t m_pieceNumber;
    ImageRotation m_rotation;

    static const PieceNumberAndRotation m_invalidPieceNumberAndRotation;
};

class PieceObjectAndRot;
using PieceNumbersAndRotationsMatrix = Eigen::Matrix<PieceNumberAndRotation, Eigen::Dynamic, Eigen::Dynamic>;

//RunData struct contains data about one run of the puzzle solving algorithm
struct RunData
{
    int32_t m_runNumber = -1;

    size_t m_numOfIterations = 0;
    std::chrono::nanoseconds m_duration = std::chrono::nanoseconds(0);
    double m_averageTimePerIteration = 0;
    double m_directComparison = 0;
    double m_neighborComparison = 0;
    bool m_wasPuzzleCorrectlySolved = false;
    bool m_isAssignmentFeasible = false; //Meaning whether the assignment solution assigns exactly one piece to each location 
    double m_assignedPiecesPercentage = 0;
                    
    TranslationDecision m_firstLevelTransDecision = TranslationDecision::eInvalidTranslationDecision;
    TranslationDecision m_secondLevelTransDecision = TranslationDecision::eInvalidTranslationDecision;

    bool m_isType2Puzzle = false;

    std::chrono::nanoseconds m_type2RunTimeForTwoRotations = std::chrono::nanoseconds(0);

    static RunData ComputeAverageRunData(const std::vector<RunData>& in_runDataVec); 
};

namespace RLSolverFileSystemUtils
{
    std::string GetIterationImageFileName(const int32_t in_iterationNum, const std::string& in_currOutputFolder);
    std::string GetScreenOutputFilePath();
    std::string GetMultipleRunsTextSummaryFilePath();
    std::string GetIterationsFolderPath(const std::string& in_currOutputFolder);
    std::string GetOriginalImagePath(const int32_t in_runNumber);
    std::string GetPuzzleImagePath(const int32_t in_runNumber);
    std::string GetSolutionImagePath(const std::string& in_currOutputFolder);
    std::string GetPiecesFolderPath(const int32_t in_runNumber);
}


namespace RLSolverGeneralUtils
{
    Orientation GetOppositeOrientation(const Orientation in_orientation);

    Orientation GetNeighboringRelation(const RowColInfo& in_firstCoord, const RowColInfo& in_secondCoord);

    std::string GetOrientationString(const Orientation in_orientation);
    std::string GetRotationString(const ImageRotation in_rotation, const bool in_withRotationSign = true);
    std::string GetDissimilarityTypeString(const DissimilarityType in_dissimilarityType);
    std::string GetPuzzleTypeString(const PuzzleType in_puzzleType);
    std::string GetTranslationDecisionString(const TranslationDecision in_transDecision);
    std::string GetTranslationDilemmaString(const TranslationMode in_transDilemma);
    
    TranslationDecision GetTranslationDecisionFromDilemma(const TranslationMode& in_transDilemma);
}

using namespace RLSolverGeneralUtils;

class PieceObjectAndRot;

struct DissimilarityInfo
{
    using RotatedPieceAndRankPair = std::pair<const PieceObjectAndRot*, int32_t>;
    
    std::vector<double> m_sortedDissimilarityValues;
    std::vector<RotatedPieceAndRankPair> m_sortedOtherRotatedPieces;
};