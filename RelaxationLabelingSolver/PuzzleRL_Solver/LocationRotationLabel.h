#pragma once

#include "Label.h"
#include "ImageUtils.h"

class LocationRotationLabel : public Label
{
public:
    LocationRotationLabel(const int32_t in_row, const int32_t in_column, const ImageRotation in_rotation) :
        m_row(in_row), m_column(in_column), m_rotation(in_rotation) {};

    std::string GetString() const override;

    bool IsTopNeighborOf(const LocationRotationLabel& in_other) const {return (m_row + 1) == in_other.m_row && m_column == in_other.m_column;};
    bool IsBottomNeighborOf(const LocationRotationLabel& in_other) const {return (m_row - 1) == in_other.m_row && m_column == in_other.m_column;};
    bool IsLeftNeighborOf(const LocationRotationLabel& in_other) const {return (m_column + 1) == in_other.m_column && m_row == in_other.m_row;};
    bool IsRightNeighborOf(const LocationRotationLabel& in_other) const {return (m_column - 1) == in_other.m_column && m_row == in_other.m_row;};

    bool IsNeighborOf(const LocationRotationLabel& in_other) const
    {return IsTopNeighborOf(in_other) || IsBottomNeighborOf(in_other) || IsLeftNeighborOf(in_other)|| IsRightNeighborOf(in_other);};

    const int32_t m_row;
    const int32_t m_column;
    const ImageRotation m_rotation;

    static void SetPuzzleType(const PuzzleType& in_puzzleType) {m_puzzleType = in_puzzleType;};

protected:
    static PuzzleType m_puzzleType;
};

using LocationRotationLabelsVector = std::vector<const LocationRotationLabel*>;
