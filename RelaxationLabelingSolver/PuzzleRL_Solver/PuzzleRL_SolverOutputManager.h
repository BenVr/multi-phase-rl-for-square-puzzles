#pragma once
#include "PuzzleRL_SolverUtils.h"
#include "PuzzlePieceObject.h"

class PuzzleRL_Solver;

class PuzzleRL_SolverOutputManager
{
public:
    PuzzleRL_SolverOutputManager(const PuzzleRL_Solver* const in_puzzleRL_Solver) : m_puzzleSolver(in_puzzleRL_Solver) {}

    void OutputAllPiecesSeparately() const;

    std::string GetSinglePuzzleOutput(const RunData& in_runData, const long long& in_timeInit, const double in_finalALC) const;

    static void LogMultiplePuzzlesOutput(const std::vector<RunData>& in_runDataVector);

    std::string GetPiecesOrientationStr(const PieceObjectAndRot* in_firstRotatedPiece,
    const PieceObjectAndRot* in_secondRotatedPiece, const Orientation in_orien, const double& in_val = -1) const;

    void LogConstantPiecesInfo() const;

    void LogWrongAssignmentsInfo(const RowColInfoSet& in_wrongAssignedPieceCoordsSet, const PieceNumbersAndRotationsMatrix& in_rotatedSolution) const;

protected:
    std::string GetCompatibilityTypeString() const;

    void PrintWrongAssignmentsInRotatedSolutionVisualization(const RowColInfoSet& in_wrongAssignedPieceCoordsSet, 
        const PieceNumbersAndRotationsMatrix& in_rotatedSolution) const;

    const PuzzleRL_Solver* const m_puzzleSolver;

public:
    static void AddToMultipleRunsSummary(const std::string& in_strForTxt);

    static std::string GetStartTranslationDilemmaString(const int32_t& in_level, const std::string& in_transAction, const std::string& in_transDilemma);
    static std::string GetDoneTranslationDilemmaString(const int32_t& in_level, const std::string& in_transAction, const std::string& in_transDilemma);

    static bool m_areMultipleRuns;

    static std::string GetLevelTitleString(const int32_t& in_level, const std::string& in_title);

    static std::string Get1stLevelTitleString(const std::string& in_title)
    {
        return "***************************************   " + in_title + "   **************************************";
    };

    static std::string Get2ndLevelTitleString(const std::string& in_title)
    {
        return "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   " + in_title + "   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~";
    };

    static std::string Get3rdLevelTitleString(const std::string& in_title)
    {
        return "^^^^^^^^^^^^^^^^^^   " + in_title + "   ^^^^^^^^^^^^^^^^^^";
    };
    
protected:
    static std::string GetSeparatorString()
    {
        return "################################";
    };
};