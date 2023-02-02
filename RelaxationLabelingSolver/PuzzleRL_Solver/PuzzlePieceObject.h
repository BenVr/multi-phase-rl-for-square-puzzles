#pragma once

#include "Object.h"
#include "ImageUtils.h"
#include "PuzzleRL_SolverConstants.h"
#include <vector>

class PuzzlePieceObject;
class PieceObjectAndRot;

using RotatedPiecesPair = std::pair<const PieceObjectAndRot*, const PieceObjectAndRot*>;
using PuzzlePieceObjectsVector = std::vector<const PuzzlePieceObject*>;
using PuzzlePieceObjectsSet = std::set<const PuzzlePieceObject*>;
using RotatedPiecesVector = std::vector<const PieceObjectAndRot*>;
using RotatedPiecesSet = std::set<const PieceObjectAndRot*>;
using PieceObjectAndRotationValPair = std::pair<const PieceObjectAndRot*, double>;

class PuzzlePieceObject : public Object
{
public:
    PuzzlePieceObject(const int32_t in_pieceNumber, const cv::Mat& in_piece);

    std::string GetString() const override;
    std::string GetStringForPiecePrint() const;

    const cv::Mat& GetPiece() const { return m_piece; }

    const int32_t m_pieceNumber;

protected:

    const cv::Mat m_piece;
};

class PieceObjectAndRot
{
public:
    PieceObjectAndRot(const PuzzlePieceObject* in_pieceObject, const ImageRotation in_rotation, const int32_t in_index);

    //PieceObjectAndRot(const PieceObjectAndRot&) = default;
    bool operator==(const PieceObjectAndRot& in_other) const {return *m_pieceObject == *in_other.m_pieceObject && m_rotation == in_other.m_rotation;}
    bool operator!=(const PieceObjectAndRot& in_other) const { return !(*this == in_other); }

    std::string GetString() const;
    const cv::Mat& GetRotatedPiece() const { return m_rotatedPiece;}

    std::vector<PixelType> GetBottomEdgePixels() const {return ImageUtils::GetBottomEdgePixels(m_rotatedPiece);}
    std::vector<PixelType> GetTopEdgePixels() const {return ImageUtils::GetTopEdgePixels(m_rotatedPiece);}
    std::vector<PixelType> GetRightEdgePixels() const {return ImageUtils::GetRightEdgePixels(m_rotatedPiece);}
    std::vector<PixelType> GetLeftEdgePixels() const {return ImageUtils::GetLeftEdgePixels(m_rotatedPiece);}

    std::vector<PixelType> GetSecondBottomEdge() const {return ImageUtils::GetSecondBottomEdge(m_rotatedPiece);}
    std::vector<PixelType> GetSecondTopEdge() const {return ImageUtils::GetSecondTopEdge(m_rotatedPiece);}
    std::vector<PixelType> GetSecondRightEdge() const {return ImageUtils::GetSecondRightEdge(m_rotatedPiece);}
    std::vector<PixelType> GetSecondLeftEdge() const {return ImageUtils::GetSecondLeftEdge(m_rotatedPiece);}

    std::vector<ExtendedPixelType> GetEdgePixelsForDissimilarityComputation(const Orientation in_orien) const;
    std::vector<ExtendedPixelType> GetBottomEdgePixelsForDissimilarityComputation() const;
    std::vector<ExtendedPixelType> GetTopEdgePixelsForDissimilarityComputation() const;
    std::vector<ExtendedPixelType> GetRightEdgePixelsForDissimilarityComputation() const;
    std::vector<ExtendedPixelType> GetLeftEdgePixelsForDissimilarityComputation() const;
              
    const PuzzlePieceObject* m_pieceObject;
    const ImageRotation m_rotation;
    const int32_t m_index;

    static bool ArePieceObjectsIdentical(const PieceObjectAndRot & in_lhs, const PieceObjectAndRot & in_rhs) 
    { return *in_lhs.m_pieceObject == *in_rhs.m_pieceObject; }

    static const PieceObjectAndRot* GetRotatedPieceFromPieceNumberAndRotation(const int32_t in_pieceNumber, const ImageRotation in_rotation, 
        const RotatedPiecesVector& in_pieceObjectAndRotVec);

protected:

    const cv::Mat m_rotatedPiece;
};