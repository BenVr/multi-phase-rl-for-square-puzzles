#pragma once
#include "Utilities.h"

struct AnchoringData
{
    AnchoringData(const size_t& in_numOfRows);

    AnchoringData(const AnchoringData& in_other) = default;
    AnchoringData& operator=(const AnchoringData& in_other) = default;

    int32_t GetPieceAnchoredToLocationIndex(const int32_t in_locationRotationIndex) const;
    int32_t GetPieceAnchoredToOneLocationRotationIndex(const std::set<int32_t>& in_labelIndexesSet, int32_t& out_locationRotationLabelIndex) const;

    std::set<int32_t> m_anchoredRowsSet;
    std::set<RowColInfo> m_anchoredEntriesSet;
    std::set<int32_t> m_nonAnchoredRowsSet;
};

struct RL_algorithmRunningInfo
{
    RL_algorithmRunningInfo() = delete;
    RL_algorithmRunningInfo(const size_t& in_numOfRows) :m_anchoringData(in_numOfRows) {}
    virtual ~RL_algorithmRunningInfo() = default;

    virtual void Set(const RL_algorithmRunningInfo& in_other) = 0;

    AnchoringData m_anchoringData;
    int32_t m_iterationNum = 0;
    double m_finalAlc = -1.0;

    Utilities::GraphData m_solutionChangedValues;

    double m_averageTimePerIteration = 0;
    long long sumOfIterationsTimeInMilliseconds = 0;

    long long m_initTime = 0;
    std::chrono::nanoseconds m_totalTime;

protected:
    RL_algorithmRunningInfo(const RL_algorithmRunningInfo& in_other) = default;
    RL_algorithmRunningInfo& operator=(const RL_algorithmRunningInfo& in_other);
};
