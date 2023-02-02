#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include "PuzzleRL_SolverUtils.h"


using PixelType = cv::Vec3b;
//ExtendedPixelType is like PixelType, but in ExtendedPixelType each channel value is short and not uchar like in PixelType
using ExtendedPixelType = cv::Vec3s;
static constexpr int32_t numOfChannels = PixelType::channels;

namespace ImageUtils
{
    struct Colors
    {
        static const PixelType m_grayLab; 
    };

    struct GroundTruthSolutionInfo
    {
        PieceNumbersAndRotationsMatrix m_assignmentMatrix;
        std::map<int32_t, cv::Mat> m_piecesWithNoTextMap;

        bool DoesRelationExistInGroundTruth(const PieceNumberAndRotation& in_firstPiece, const PieceNumberAndRotation& in_secondPiece,
            const Orientation& in_orien) const;
    };

    inline int32_t GetNumOfPiecesPerRow(const cv::Mat& in_image, const int32_t in_pieceSize) { return in_image.rows / in_pieceSize; }
    inline int32_t GetNumOfPiecesPerCol(const cv::Mat& in_image, const int32_t in_pieceSize) { return in_image.cols / in_pieceSize; }

    cv::Mat TruncatedImage(const cv::Mat in_image, const int32_t in_rowPixels, const int32_t in_colPixels);
    void SetPieceToImageCoordinates(const cv::Mat& in_pieceImage, const Utilities::RowColInfo& in_coordinates, cv::Mat& inout_image);
    void SetRotatedPieceInLocation(const cv::Mat& in_pieceImage, const ImageRotation in_rotation,
        const Utilities::RowColInfo& in_rowCol, cv::Mat& inout_image);

    void AddPiecesGridToImage(cv::Mat& inout_image, const int32_t in_pieceSize);
    void PrintTextToPieceImage(cv::Mat& inout_pieceImage, const std::string& in_text, const cv::Scalar& in_textColor, const bool in_checkPixelChange = true);

    cv::Mat RotateImage(const cv::Mat& in_image, const ImageRotation in_rotation);
    PieceNumbersAndRotationsMatrix RotatePuzzleAssignmentMatrix(const PieceNumbersAndRotationsMatrix& in_puzzleAssignmentMatrix, const ImageRotation in_rotation);
    PieceNumbersAndRotationsMatrix RotateAssignmetMatrixBy90DegreesClockwise(const PieceNumbersAndRotationsMatrix& in_puzzleAssignmentMatrix);
    RotationsVec GetRotationsToCheck(const bool in_isSquarePuzzle, const PuzzleType in_puzzleType);

    std::vector<ExtendedPixelType> GetBottomEdgePixelsForDissimilarityComputation(const cv::Mat& in_piece);
    std::vector<ExtendedPixelType> GetTopEdgePixelsForDissimilarityComputation(const cv::Mat& in_piece);
    std::vector<ExtendedPixelType> GetRightEdgePixelsForDissimilarityComputation(const cv::Mat& in_piece);
    std::vector<ExtendedPixelType> GetLeftEdgePixelsForDissimilarityComputation(const cv::Mat& in_piece);

    std::vector<PixelType> GetBottomEdgePixels(const cv::Mat& in_piece);
    std::vector<PixelType> GetTopEdgePixels(const cv::Mat& in_piece);
    std::vector<PixelType> GetRightEdgePixels(const cv::Mat& in_piece);
    std::vector<PixelType> GetLeftEdgePixels(const cv::Mat& in_piece);

    std::vector<PixelType> GetSecondBottomEdge(const cv::Mat& in_piece);
    std::vector<PixelType> GetSecondTopEdge(const cv::Mat& in_piece);
    std::vector<PixelType> GetSecondRightEdge(const cv::Mat& in_piece);
    std::vector<PixelType> GetSecondLeftEdge(const cv::Mat& in_piece);

    std::vector<PixelType> GetRowOfPixels(const cv::Mat& in_piece, const int32_t in_row);
    std::vector<PixelType> GetColumnOfPixels(const cv::Mat& in_piece, const int32_t in_col);

    std::vector<ExtendedPixelType> GetAsExtendedPixelVector(const std::vector<PixelType>& in_vec);

    cv::Mat GetImageMatrixFromAssignementAndPieces(const PieceNumbersAndRotationsMatrix& in_assignmentMatrix, const int32_t in_pieceSize, 
        const std::map<int32_t, cv::Mat>& in_piecesMap);

    int32_t GetNumOfAssignedPiecesInPieceAssignmentMatrix(const PieceNumbersAndRotationsMatrix& in_assignmentMatrix);

    double ComputeDirectComparison(const GroundTruthSolutionInfo& in_groundTruthSolutionInfo, const PieceNumbersAndRotationsMatrix& in_solutionMatrix,
        const int32_t in_numOfPieces, RowColInfoSet& out_WrongAssignedPieceCoordsSet);
    double ComputeNeighborComparison(const GroundTruthSolutionInfo& in_groundTruthSolutionInfo, const PieceNumbersAndRotationsMatrix& in_solutionMatrix);

    ImageRotation GetRotationDifference(const ImageRotation& in_firstRotation, const ImageRotation& in_secondRotation);

    ImageRotation GetCounterRotation(const ImageRotation in_rotation);

    cv::Mat ReadImage(const std::string& in_file);
    void WriteImage(const std::string& in_file, const cv::Mat& in_image);

    ImageRotation SumSquareImageRotations(const ImageRotation in_firstRotation, const ImageRotation in_secondRotation);

    ImageRotation GetRotationFromSourceAndDestEdgeOrientations(const Orientation in_sourceOrien, const Orientation in_destOrien);

    inline ExtendedPixelType GetImprovedMGC_Delta(const cv::Mat& in_piece, const int32_t in_row, const int32_t in_col)
    {
        return in_piece.at<PixelType>(in_row, in_col) - in_piece.at<PixelType>(in_row - 1, in_col);
    };

    PixelType ConvertPixelFromLAB_ToBGR(const PixelType& in_pixelLAB);
    PixelType ConvertPixelFromBGR_ToLAB(const PixelType& in_pixelBGR);

    static constexpr cv::ColorConversionCodes colorConversionToBGR = cv::COLOR_Lab2BGR;
    static constexpr cv::ColorConversionCodes colorConversionFromBGR = cv::COLOR_BGR2Lab;

    //imagesType is CV_8UC3, which means that each pixel has 3 channels, each with 256 bits 
    static constexpr int32_t imagesType = CV_8UC3;
}

using GroundTruthSolutionInfo = ImageUtils::GroundTruthSolutionInfo;