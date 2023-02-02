#pragma once

#include <opencv2/core/mat.hpp>
#include <Eigen/Core>
#include "ImageUtils.h"
#include "PuzzlePieceObject.h"
#include "PuzzleRL_SolverUtils.h"

struct PieceInfo
{
    PieceInfo(const cv::Mat& in_piece, const cv::Mat& m_pieceNoLabel, const int32_t in_pieceNumber, const Utilities::RowColInfo& in_coordinatesInPuzzleImage)
    : m_piece(in_piece), m_pieceNoText(m_pieceNoLabel), m_pieceNumber(in_pieceNumber), 
    m_coordinatesInPuzzleImage(in_coordinatesInPuzzleImage) {}

    const cv::Mat m_piece;
    const cv::Mat m_pieceNoText;
    const int32_t m_pieceNumber;
    const Utilities::RowColInfo m_coordinatesInPuzzleImage;
};

using PiecesInfoVector = std::vector<PieceInfo>;

class PuzzleImage
{
public:
    PuzzleImage(const cv::Mat& in_image, const int in_pieceSize, const int32_t in_numOfPiecesPerRow,
        const int32_t in_numOfPiecesPerCol, const PuzzleType in_puzzleType, const bool in_shouldAddPieceStringsToImage);
    
    int32_t GetNumOfRowPieces() const { return static_cast<int32_t>(m_shufflingMatrix.rows()); }
    int32_t GetNumOfColPieces() const { return static_cast<int32_t>(m_shufflingMatrix.cols()); }
    int32_t GetNumOfPieces() const { return GetNumOfRowPieces() * GetNumOfColPieces(); }
    int32_t GetPieceSize() const { return m_pieceSize; }
    bool IsSquarePuzzle() const { return GetNumOfRowPieces() == GetNumOfColPieces(); }

    cv::Mat GetImage() const { return m_image; }
    cv::Mat GetOriginalImage() const;

    GroundTruthSolutionInfo GetGroundTruthSolutionInfo() const { return m_groundTruthInfo; }
    PiecesInfoVector GetPiecesInfoVectorFromImage() const { return m_piecesInfoVec; }
    
protected:
    void SetPieceInfoVector(const bool in_shouldAddPieceStringsToImage);
    void SetPuzzleImage();
    void SetGroundTruthMatrix();

    void AddPieceStringToPieceImage(cv::Mat& in_pieceImage, const int32_t in_pieceNumber) const;
    cv::Mat GetPuzzlePiece(const int32_t in_row, const int32_t in_col, const cv::Mat& in_image) const;

    cv::Mat m_image;
    const int32_t m_pieceSize;

    //m_shufflingMatrix contains all the information needed in order to shuffle and rotate puzzle pieces
    const PieceNumbersAndRotationsMatrix m_shufflingMatrix;
    PiecesInfoVector m_piecesInfoVec;

    GroundTruthSolutionInfo m_groundTruthInfo;

public:
    static PuzzleImage CreateAndInitPuzzleImage(const cv::Mat& in_image, const int in_pieceSize, const PuzzleType in_puzzleType, const bool in_addLabelStrings);

protected:
    static PieceNumbersAndRotationsMatrix GetDefaultPieceNumbersAndRotationsMatrix(const int32_t in_numOfRows, const int32_t in_numOfCols);
    static PieceNumbersAndRotationsMatrix GetShufflingMatrix(const int32_t in_numOfRows, const int32_t in_numOfCols, const PuzzleType in_puzzleType);
    static void ShufflePieceNumbersInMatrix(PieceNumbersAndRotationsMatrix& inout_numbersAndRotationsMatrix);
    static void DrawPieceRotationsInMatrix(PieceNumbersAndRotationsMatrix& inout_numbersAndRotationsMatrix);

    static Utilities::RowColInfo GetMatrixRowAndColFromPieceNumber(const int32_t in_pieceNumber, const PieceNumbersAndRotationsMatrix& in_pieceNumbersAndRotationsMatrix);
};
