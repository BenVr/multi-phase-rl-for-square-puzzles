#include "PuzzleRL_SolverUtils.h"
#include "FileSystemUtils.h"
#include "PuzzleRL_SolverConstants.h"
#include "RelaxationLabelingSolverBaseConstants.h"
#include <sstream>

const PieceNumberAndRotation PieceNumberAndRotation::m_invalidPieceNumberAndRotation(-1, ImageRotation::eInvalidRotation);

/************************************************************************/
std::string PieceNumberAndRotation::GetString() const
/************************************************************************/
{
    return "piece " + std::to_string(m_pieceNumber) + ", " + RLSolverGeneralUtils::GetRotationString(m_rotation);
}

/************************************************************************/
void RL_puzzleSolverAlgorithmRunningInfo::UpdateTranslationDecision(const TranslationDecision& in_transDecision)
/************************************************************************/
{
    if (TranslationDecision::eInvalidTranslationDecision == m_firstLevelTransDecision)
        m_firstLevelTransDecision = in_transDecision;
    else if (TranslationDecision::eInvalidTranslationDecision == m_secondLevelTransDecision)
        m_secondLevelTransDecision = in_transDecision;
    else
        Utilities::LogAndAbort("In 'RL_puzzleSolverAlgorithmRunningInfo::UpdateTranslationDecision()': shouldn't get here");    
}

/************************************************************************/
RL_puzzleSolverAlgorithmRunningInfo& RL_puzzleSolverAlgorithmRunningInfo::operator=(const RL_puzzleSolverAlgorithmRunningInfo& in_other)
/************************************************************************/
{
    if (&in_other != this)
    {
        RL_algorithmRunningInfo::operator=(in_other);

        m_firstLevelTransDecision = in_other.m_firstLevelTransDecision;  
        m_secondLevelTransDecision = in_other.m_secondLevelTransDecision;
        m_translationDilemmaLevel = in_other.m_translationDilemmaLevel;
    }

    return *this;
};

/************************************************************************/
RunData RunData::ComputeAverageRunData(const std::vector<RunData>& in_runDataVec)
//Function input: in_runDataVec, a vector of RunData structs
//Function output: another RunData struct which contains the averages stats of the runs in in_runDataVec
/************************************************************************/
{
    RunData averageRunData;
    for (const RunData& runData : in_runDataVec)
    {
        averageRunData.m_numOfIterations += runData.m_numOfIterations;
        averageRunData.m_duration += runData.m_duration;
        averageRunData.m_directComparison += runData.m_directComparison;
        averageRunData.m_neighborComparison += runData.m_neighborComparison;
        averageRunData.m_assignedPiecesPercentage += runData.m_assignedPiecesPercentage;
                                          
        averageRunData.m_isType2Puzzle = runData.m_isType2Puzzle;

        averageRunData.m_type2RunTimeForTwoRotations += runData.m_type2RunTimeForTwoRotations;
    }

    averageRunData.m_numOfIterations /= in_runDataVec.size();
    averageRunData.m_duration /= in_runDataVec.size();
    averageRunData.m_directComparison /= in_runDataVec.size();
    averageRunData.m_neighborComparison /= in_runDataVec.size();
    averageRunData.m_assignedPiecesPercentage /= in_runDataVec.size();

    averageRunData.m_type2RunTimeForTwoRotations /= in_runDataVec.size();

    return averageRunData;
}

/************************************************************************/
std::string RLSolverFileSystemUtils::GetIterationImageFileName(const int32_t in_iterationNum, const std::string& in_currOutputFolder)
/************************************************************************/
{
    std::stringstream fileName;
    fileName << GetIterationsFolderPath(in_currOutputFolder) << in_iterationNum << ".png";
    return fileName.str();
}

/************************************************************************/
std::string RLSolverFileSystemUtils::GetScreenOutputFilePath()
/************************************************************************/
{
    std::stringstream fileName;
    fileName << OUTPUT_DIRECTORY_PATH.string() << SCREEN_OUTPUT_FILE.string();
    return fileName.str();
}

/************************************************************************/
std::string RLSolverFileSystemUtils::GetMultipleRunsTextSummaryFilePath()
/************************************************************************/
{
    std::stringstream fileName;
    fileName << OUTPUT_DIRECTORY_PATH.string() << MULTIPLE_RUNS_TEXT_SUMMARY_FILE.string();
    return fileName.str();
}

/************************************************************************/
std::string RLSolverFileSystemUtils::GetIterationsFolderPath(const std::string& in_currOutputFolder)
/************************************************************************/
{
    return FileSystemUtils::GetPathInFolder(in_currOutputFolder, ITERATIONS_FOLDER_NAME.string());
}

/************************************************************************/
std::string RLSolverFileSystemUtils::GetOriginalImagePath(const int32_t in_runNumber)
/************************************************************************/
{
    return FileSystemUtils::GetPathInRunFolder(in_runNumber, ORIGINAL_IMAGE_FILE_NAME.string());
}

/************************************************************************/
std::string RLSolverFileSystemUtils::GetPuzzleImagePath(const int32_t in_runNumber)
/************************************************************************/
{
    return FileSystemUtils::GetPathInRunFolder(in_runNumber, PUZZLE_IMAGE_FILE_NAME.string());
}

/************************************************************************/
std::string RLSolverFileSystemUtils::GetSolutionImagePath(const std::string& in_currOutputFolder)
/************************************************************************/
{
    return FileSystemUtils::GetPathInFolder(in_currOutputFolder, SOLUTION_IMAGE_FILE_NAME.string());
}

/************************************************************************/
std::string RLSolverFileSystemUtils::GetPiecesFolderPath(const int32_t in_runNumber)
/************************************************************************/
{
    return FileSystemUtils::GetPathInRunFolder(in_runNumber, PRINT_ALL_PIECES_PATH.string());
}

/************************************************************************/
Orientation RLSolverGeneralUtils::GetOppositeOrientation(const Orientation in_orientation)
/************************************************************************/
{
    Orientation oppositeOrienataion = eInvalidOrientation;
    switch (in_orientation)
    {
    case eDown:
        oppositeOrienataion = eUp;
        break;

    case eUp:
        oppositeOrienataion = eDown;
        break;

    case eRight:
        oppositeOrienataion = eLeft;
        break;

    case eLeft:
        oppositeOrienataion = eRight;
        break;

    default:
        Utilities::LogAndAbort("");
    }

    return oppositeOrienataion;
}

/************************************************************************/
Orientation RLSolverGeneralUtils::GetNeighboringRelation(const RowColInfo& in_firstCoord, const RowColInfo& in_secondCoord)
//Function input: in_firstCoord: first coordinate; in_secondCoord: first coordinate
//Function output: orientation representing the neighboring relation between 'in_firstCoord' and 'in_secondCoord'
//Function objective: described in "Function output"
/************************************************************************/
{
    Orientation retVal = eInvalidOrientation;

    if (RowColInfo::GetTopNeighborCoord(in_firstCoord.m_row, in_firstCoord.m_col) == in_secondCoord)
        retVal = eUp;
    else if (RowColInfo::GetBottomNeighborCoord(in_firstCoord.m_row, in_firstCoord.m_col) == in_secondCoord)
        retVal = eDown;
    else if (RowColInfo::GetLeftNeighborCoord(in_firstCoord.m_row, in_firstCoord.m_col) == in_secondCoord)
        retVal = eLeft;
    else if (RowColInfo::GetRightNeighborCoord(in_firstCoord.m_row, in_firstCoord.m_col) == in_secondCoord)
        retVal = eRight;
    
    return retVal;
}

/************************************************************************/
std::string RLSolverGeneralUtils::GetPuzzleTypeString(const PuzzleType in_puzzleType)
/************************************************************************/
{
    std::string str;

    switch (in_puzzleType)
    {
    case PuzzleType::eType1_NoRotation:
        str = "type 1 (no rotation)";
        break;

    case PuzzleType::eType2_UknownRotation:
        str = "type 2 (unknown rotation)";
        break;

    default:
        Utilities::LogAndAbort("Illegal puzzle type");
    }

    return str;
}

/************************************************************************/
std::string RLSolverGeneralUtils::GetTranslationDecisionString(const TranslationDecision in_tranDecision)
/************************************************************************/
{
    std::string str;

    switch (in_tranDecision)
    {
    case TranslationDecision::eNoTranslation:
        str = "no";
        break;

    case TranslationDecision::eDoTranslationUp:
        str = "up";
        break;

    case TranslationDecision::eDoTranslationDown:
        str = "down";
        break;

    case TranslationDecision::eDoTranslationLeft:
        str = "left";
        break;

    case TranslationDecision::eDoTranslationRight:
        str = "right";
        break;

    default:
        Utilities::LogAndAbort("In 'RLSolverGeneralUtils::GetTranslationDecisionString()': shouldn't get here");
    }

    str += " translation";

    return str;
}

/************************************************************************/
std::string RLSolverGeneralUtils::GetTranslationDilemmaString(const TranslationMode in_transDilemma)
/************************************************************************/
{
    std::string str;

    switch (in_transDilemma)
    {
        case TranslationMode::eVerticalTranslationUpDilemma:
            str = "up";
            break;

        case TranslationMode::eVerticalTranslationDownDilemma:
            str = "down";
            break;

        case TranslationMode::eHorizontalTranslationLeftDilemma:
            str = "left";
            break;

        case TranslationMode::eHorizontalTranslationRightDilemma:
            str = "right";
            break;

        default:
            Utilities::LogAndAbort("In 'RLSolverGeneralUtils::GetTranslationDilemmaString()': shouldn't get here");
            break;
    }

    str += " dilemma";

    return str;
}

/************************************************************************/
TranslationDecision RLSolverGeneralUtils::GetTranslationDecisionFromDilemma(const TranslationMode& in_transDilemma)
/************************************************************************/
{
    TranslationDecision transDecision = TranslationDecision::eInvalidTranslationDecision;

    switch (in_transDilemma)
    {
        case TranslationMode::eVerticalTranslationUpDilemma:
            transDecision = TranslationDecision::eDoTranslationUp;
            break;

        case TranslationMode::eVerticalTranslationDownDilemma:
            transDecision = TranslationDecision::eDoTranslationDown;
            break;

        case TranslationMode::eHorizontalTranslationLeftDilemma:
            transDecision = TranslationDecision::eDoTranslationLeft;
            break;

        case TranslationMode::eHorizontalTranslationRightDilemma:
            transDecision = TranslationDecision::eDoTranslationRight;
            break;

        default:
            Utilities::LogAndAbort("In 'RLSolverGeneralUtils::GetTranslationDecisionFromDilemma()': shouldn't get here");
            break;
    }

    return transDecision;
}

/************************************************************************/
std::string RLSolverGeneralUtils::GetOrientationString(const Orientation in_orientation)
/************************************************************************/
{
    std::string str;
    switch (in_orientation)
    {

    case eUp:
        str = "up";
        break;

    case eRight:
        str = "right";
        break;

    case eDown:
        str = "down";
        break;

    case eLeft:
        str = "left";
        break;

    default:
        Utilities::LogAndAbort("");
    }

    return str;
}

/************************************************************************/
std::string RLSolverGeneralUtils::GetRotationString(const ImageRotation in_rotation, const bool in_withRotationSign)
/************************************************************************/
{
    std::string str;

    switch (in_rotation)
    {
    case ImageRotation::e_0_degrees:
        str = "0";
        break;

    case ImageRotation::e_90_degrees:
        str = "90";
        break;

    case ImageRotation::e_180_degrees:
        str = "180";
        break;

    case ImageRotation::e_270_degrees:
        str = "270";
        break;

    default:
        Utilities::LogAndAbort("RLSolverGeneralUtils::GetRotationString() failed");
    }

    if (in_withRotationSign)
        str += "°";

    return str;
}

/************************************************************************/
std::string RLSolverGeneralUtils::GetDissimilarityTypeString(const DissimilarityType in_dissimilarityType)
/************************************************************************/
{
    std::string str;
    switch (in_dissimilarityType)
    {
    case DissimilarityType::eImprovedMGC:
        str = "improved MGC";
        break;

    default:
        Utilities::LogAndAbort("");
    }

    return str;
}