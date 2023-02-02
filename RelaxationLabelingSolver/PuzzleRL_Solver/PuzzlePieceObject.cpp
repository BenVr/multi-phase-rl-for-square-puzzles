#include "PuzzlePieceObject.h"

/************************************************************************/
PuzzlePieceObject::PuzzlePieceObject(const int32_t in_pieceNumber, const cv::Mat& in_piece) :
m_pieceNumber(in_pieceNumber),
m_piece(in_piece)
/************************************************************************/
{
}

/************************************************************************/
std::string PuzzlePieceObject::GetString() const
/************************************************************************/
{
    std::stringstream str;
    str << "piece #" << m_pieceNumber;
    return str.str();
}

/************************************************************************/
std::string PuzzlePieceObject::GetStringForPiecePrint() const
/************************************************************************/
{
    std::stringstream str;
    str << m_pieceNumber << ".png";
    return str.str();
}

/************************************************************************/
PieceObjectAndRot::PieceObjectAndRot(const PuzzlePieceObject* in_pieceObject, const ImageRotation in_rotation, const int32_t in_index) :
m_pieceObject(in_pieceObject), 
m_rotation(in_rotation),
m_index(in_index),
m_rotatedPiece(ImageUtils::RotateImage(in_pieceObject->GetPiece(), m_rotation)) 
/************************************************************************/
{
}

/************************************************************************/
std::string PieceObjectAndRot::GetString() const
/************************************************************************/
{
    return m_pieceObject->GetString() + ", " + RLSolverGeneralUtils::GetRotationString(m_rotation);
}

/************************************************************************/
std::vector<ExtendedPixelType> PieceObjectAndRot::GetEdgePixelsForDissimilarityComputation(const Orientation in_orien) const
//Function input: in_orien: orientation
//Function output: get edge pixels that take part in dissimilarity computation (dissimilarity is 'DISSIMILARITY_TYPE').
    //(in case two edges take part in dissimilarity computation, we return concatenation of them)
//Function objective: described in "Function output"
/************************************************************************/
{
    std::vector<ExtendedPixelType> edgePixelsVec;

    switch (in_orien)
    {
        case Orientation::eUp:
            edgePixelsVec = GetTopEdgePixelsForDissimilarityComputation();
            break;

        case Orientation::eRight:
            edgePixelsVec = GetRightEdgePixelsForDissimilarityComputation();
            break;

        case Orientation::eDown:
            edgePixelsVec = GetBottomEdgePixelsForDissimilarityComputation();
            break;

        case Orientation::eLeft:
            edgePixelsVec = GetLeftEdgePixelsForDissimilarityComputation();
            break;

        default:
            Utilities::LogAndAbort("'PieceObjectAndRot::GetEdgePixelsForDissimilarityComputation()': Shouldn't get here");
            break;
    }

    return edgePixelsVec;
}

/************************************************************************/
std::vector<ExtendedPixelType> PieceObjectAndRot::GetBottomEdgePixelsForDissimilarityComputation() const
//Function input: none
//Function output: get edge pixels that take part in dissimilarity computation of bottom edge (dissimilarity is 'DISSIMILARITY_TYPE')
//Function objective: described in "Function output"
/************************************************************************/
{
    return ImageUtils::GetBottomEdgePixelsForDissimilarityComputation(m_rotatedPiece);
}

/************************************************************************/
std::vector<ExtendedPixelType> PieceObjectAndRot::GetTopEdgePixelsForDissimilarityComputation() const
//Function input: none
//Function output: get edge pixels that take part in dissimilarity computation of top edge (dissimilarity is 'DISSIMILARITY_TYPE')
//Function objective: described in "Function output"
/************************************************************************/
{
    return ImageUtils::GetTopEdgePixelsForDissimilarityComputation(m_rotatedPiece);
}

/************************************************************************/
std::vector<ExtendedPixelType> PieceObjectAndRot::GetRightEdgePixelsForDissimilarityComputation() const
//Function input: none
//Function output: get edge pixels that take part in dissimilarity computation of right edge (dissimilarity is 'DISSIMILARITY_TYPE')
//Function objective: described in "Function output"
/************************************************************************/
{
    return ImageUtils::GetRightEdgePixelsForDissimilarityComputation(m_rotatedPiece);
}

/************************************************************************/
std::vector<ExtendedPixelType> PieceObjectAndRot::GetLeftEdgePixelsForDissimilarityComputation() const
//Function input: none
//Function output: get edge pixels that take part in dissimilarity computation of left edge (dissimilarity is 'DISSIMILARITY_TYPE')
//Function objective: described in "Function output"
/************************************************************************/
{
    return ImageUtils::GetLeftEdgePixelsForDissimilarityComputation(m_rotatedPiece);
}

/************************************************************************/
const PieceObjectAndRot* PieceObjectAndRot::GetRotatedPieceFromPieceNumberAndRotation(const int32_t in_pieceNumber, const ImageRotation in_rotation, 
        const RotatedPiecesVector& in_pieceObjectAndRotVec)
/************************************************************************/
{
    const PieceObjectAndRot* retVal = nullptr;
    if (-1 != in_pieceNumber)
    {
        const auto itr = std::find_if(in_pieceObjectAndRotVec.begin(), in_pieceObjectAndRotVec.end(),
            [in_pieceNumber, in_rotation](const PieceObjectAndRot* in_rotatedPiece)
            {return in_pieceNumber == in_rotatedPiece->m_pieceObject->m_pieceNumber && in_rotation == in_rotatedPiece->m_rotation; });

        Utilities::LogAndAbortIf(itr == in_pieceObjectAndRotVec.end(), "'PieceObjectAndRot::GetRotatedPieceFromPieceNumberAndRotation()' failed");
        retVal = *itr;
    }

    return retVal;
}