#include "LocationRotationLabel.h"

PuzzleType LocationRotationLabel::m_puzzleType = PuzzleType::eType1_NoRotation;

/************************************************************************/
std::string LocationRotationLabel::GetString() const
/************************************************************************/
{
    std::stringstream str;

    str << "row: " << m_row << ", " << "col: " << m_column;
    if (PuzzleType::eType2_UknownRotation == m_puzzleType)
        str << ", " << RLSolverGeneralUtils::GetRotationString(m_rotation);

    return str.str();
}
