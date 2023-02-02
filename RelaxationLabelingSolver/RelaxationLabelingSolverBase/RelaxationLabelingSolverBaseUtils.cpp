#include "RelaxationLabelingSolverBaseUtils.h"

/************************************************************************/
AnchoringData::AnchoringData(const size_t& in_numOfRows)
/************************************************************************/
{
    for (int32_t i = 0; i < in_numOfRows; ++i)
        m_nonAnchoredRowsSet.insert(i);
}

/************************************************************************/
int32_t AnchoringData::GetPieceAnchoredToLocationIndex(const int32_t in_locationRotationIndex) const
//Function input: in_locationRotationIndex: column index
//Function output: the row index anchored to column 'in_locationRotationIndex'  
//Function objective: described in "Function output"
/************************************************************************/
{
    int32_t pieceIndex = -1;
    for (const RowColInfo& currAnchoredEntry: m_anchoredEntriesSet)
    {
        if (currAnchoredEntry.m_col == in_locationRotationIndex)
        {
            pieceIndex = currAnchoredEntry.m_row;
            break;
        }
    }

    Utilities::LogAndAbortIf(-1 == pieceIndex, "-1 == pieceIndex in GetPieceAnchoredToLocationIndex");
    return pieceIndex;
}

/************************************************************************/
int32_t AnchoringData::GetPieceAnchoredToOneLocationRotationIndex(const std::set<int32_t>& in_labelIndexesSet,
    int32_t& out_locationRotationLabelIndex) const
//Function input: in_labelsSet: column indexes set; out_locationRotationLabelIndex: one column index
//Function output: the row index anchored to one column 'in_labelIndexesSet'  
//Function objective: described in "Function output"
/************************************************************************/
{
    int32_t pieceIndex = -1;
    for (const RowColInfo& currAnchoredEntry: m_anchoredEntriesSet)
    {
        if (Utilities::IsItemInSet(currAnchoredEntry.m_col, in_labelIndexesSet))
        {
            Utilities::LogAndAbortIf(-1 != pieceIndex, 
                "in 'AnchoringData::GetPieceAnchoredToOneLocationRotationIndex()': two labels of the same position are anchored");
            out_locationRotationLabelIndex = currAnchoredEntry.m_col;
            pieceIndex = currAnchoredEntry.m_row;
        }
    }

    Utilities::LogAndAbortIf(-1 == pieceIndex, "-1 == pieceIndex in GetPieceAnchoredToOneLocationRotationIndex");
    return pieceIndex;
}

/************************************************************************/
RL_algorithmRunningInfo& RL_algorithmRunningInfo::operator=(const RL_algorithmRunningInfo& in_other)
/************************************************************************/
{
    if (this != &in_other)
    {
        m_anchoringData = in_other.m_anchoringData;
        m_iterationNum = in_other.m_iterationNum;
        m_finalAlc = in_other.m_finalAlc;
        m_solutionChangedValues = in_other.m_solutionChangedValues;
        m_averageTimePerIteration = in_other.m_averageTimePerIteration;
        sumOfIterationsTimeInMilliseconds = in_other.sumOfIterationsTimeInMilliseconds;
        m_initTime = in_other.m_initTime;
        m_totalTime = in_other.m_totalTime;
    }

    return *this;
}