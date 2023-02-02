#include "PuzzleImage.h"
#include "ImageUtils.h"
#include "PuzzleRL_SolverConstants.h"
#include "Utilities.h"
#include "FileSystemUtils.h"
#include <random>

using namespace ImageUtils;
using namespace Utilities;

/************************************************************************/
PuzzleImage::PuzzleImage(const cv::Mat& in_image, const int in_pieceSize, const int32_t in_numOfPiecesPerRow,
    const int32_t in_numOfPiecesPerCol, const PuzzleType in_puzzleType, const bool in_shouldAddPieceStringsToImage) :
m_image(in_image),
m_pieceSize(in_pieceSize), 
m_shufflingMatrix(PuzzleImage::GetShufflingMatrix(in_numOfPiecesPerRow, in_numOfPiecesPerCol, in_puzzleType))
/************************************************************************/
{
    //1. Set m_piecesInfoVec
    SetPieceInfoVector(in_shouldAddPieceStringsToImage);

    //2. Set m_image
    SetPuzzleImage();

    //3. Set ground truth matrix
    SetGroundTruthMatrix();
}

/************************************************************************/
cv::Mat PuzzleImage::GetOriginalImage() const
//Function input: none
//Function output: original input image
//Function objective: described in "Function output"
/************************************************************************/
{
    //1. Init empty image
    cv::Mat originalImage = cv::Mat::zeros(m_image.rows, m_image.cols, ImageUtils::imagesType);

    //2. For all pieces
    for (const PieceInfo& currPieceInfo : m_piecesInfoVec)
    {
        //2.1. Get piece ground truth coordinates
        const RowColInfo groundTruthCoordinates = GetMatrixRowAndColFromPieceNumber(currPieceInfo.m_pieceNumber, m_groundTruthInfo.m_assignmentMatrix);

        //2.2. Get the rotation that should be applied for this piece
        const ImageRotation rotation = m_groundTruthInfo.m_assignmentMatrix(groundTruthCoordinates.m_row, groundTruthCoordinates.m_col).m_rotation;

        //2.3. Get and rotate piece image
        cv::Mat currPiece = currPieceInfo.m_piece;
        currPiece = ImageUtils::RotateImage(currPiece, rotation);

        //2.4. Copy piece to its coordinates
        ImageUtils::SetPieceToImageCoordinates(currPiece, groundTruthCoordinates, originalImage);
    }

    return originalImage;
}

/************************************************************************/
void PuzzleImage::SetPieceInfoVector(const bool in_shouldAddPieceStringsToImage)
//Function input: in_shouldAddPieceStringsToImage: boolean indicating whether we should write piece number strings in image
//Function output: none
//Function objective: set m_piecesInfoVec, which contains all the piece information
/************************************************************************/
{
    //1. Reserve memory in m_piecesInfoVec
    m_piecesInfoVec.reserve(GetNumOfPieces());
    
    //2. For all piece numbers
    for (int32_t pieceNumber = 0; pieceNumber < GetNumOfPieces(); ++pieceNumber)
    {
        //2.1. Get the coordinates of the entry containing pieceNumber, in shuffling matrix
        const RowColInfo pieceNumberCoordinates = GetMatrixRowAndColFromPieceNumber(pieceNumber, m_shufflingMatrix);

        //2.2. Get the puzzle piece corresponding to pieceNumberCoordinates
        cv::Mat currPiece = GetPuzzlePiece(pieceNumberCoordinates.m_row, pieceNumberCoordinates.m_col, m_image);
        cv::Mat currPieceNoText = currPiece.clone();

        //2.3. Add piece string if should to
        if (in_shouldAddPieceStringsToImage)
            AddPieceStringToPieceImage(currPiece, pieceNumber);

        //2.4. Get the rotation that should be applied to 'currPiece', and apply it for 'currPiece' and 'currPieceNoText' 
        const ImageRotation rotation = m_shufflingMatrix(pieceNumberCoordinates.m_row, pieceNumberCoordinates.m_col).m_rotation;
        currPiece = ImageUtils::RotateImage(currPiece, rotation);
        currPieceNoText = ImageUtils::RotateImage(currPieceNoText, rotation);

        //2.5. Get the coordinates corresponding to pieceNumber
        const RowColInfo coordinatesInPuzzleImage = RowColInfo::GetRowAndColFrom1DIndex(pieceNumber, static_cast<int32_t>(m_shufflingMatrix.cols()));

        //2.6. Push piece info to m_piecesInfoVec
        const PieceInfo pieceInfo(currPiece, currPieceNoText, pieceNumber, coordinatesInPuzzleImage);
        m_piecesInfoVec.push_back(pieceInfo);
    }
}

/************************************************************************/
void PuzzleImage::SetPuzzleImage()
//Function input: none
//Function output: none
//Function objective: set puzzle image according to m_piecesInfoVec
/************************************************************************/
{
    //1. Init empty image
    cv::Mat rotatedAndShuffledImage = cv::Mat::zeros(m_image.rows, m_image.cols, ImageUtils::imagesType);

    //2. For all pieces
    for (const PieceInfo& currPieceInfo : m_piecesInfoVec)
    {
        //2.1 Get the curr piece image, and its coordinates in the puzzle image
        const cv::Mat currPiece = currPieceInfo.m_piece;
        const RowColInfo coordinatesInPuzzleImage = currPieceInfo.m_coordinatesInPuzzleImage;

        //2.2. Copy currPiece to coordinatesInPuzzleImage in rotatedAndShuffledImage
        ImageUtils::SetPieceToImageCoordinates(currPiece, coordinatesInPuzzleImage, rotatedAndShuffledImage);
    }

    //3. Set rotated shuffled image to be m_image
    m_image = rotatedAndShuffledImage;
}

/************************************************************************/
void PuzzleImage::SetGroundTruthMatrix()
//Function input: none
//Function output: none
//Function objective: set m_groundTruthMatrix, which represents the real original piece assignment
/************************************************************************/
{
    //1. Init matrix
    m_groundTruthInfo.m_assignmentMatrix = PieceNumbersAndRotationsMatrix(GetNumOfRowPieces(), GetNumOfColPieces());

    //2. For all pieces
    for (const PieceInfo& currPieceInfo : m_piecesInfoVec)
    {
        //2.1. Get piece number and its original coordinates
        const int32_t pieceNumber = currPieceInfo.m_pieceNumber;
        const RowColInfo originalCoordinates = GetMatrixRowAndColFromPieceNumber(pieceNumber, m_shufflingMatrix);

        //2.2. Get the counter rotation of the piece (in order to get the original rotation, the counter rotation should be applied)
        const ImageRotation counterRotation =
            ImageUtils::GetCounterRotation(m_shufflingMatrix(originalCoordinates.m_row, originalCoordinates.m_col).m_rotation);

        //2.3. Set value 'in m_groundTruthInfo.m_assignmentMatrix'
        m_groundTruthInfo.m_assignmentMatrix(originalCoordinates.m_row, originalCoordinates.m_col) = PieceNumberAndRotation(pieceNumber, counterRotation);

        //2.4. Set piece with no text to 'm_groundTruthInfo.m_piecesWithNoTextMap'
        m_groundTruthInfo.m_piecesWithNoTextMap.emplace(currPieceInfo.m_pieceNumber, currPieceInfo.m_pieceNoText);
    }
}

/************************************************************************/
void PuzzleImage::AddPieceStringToPieceImage(cv::Mat& in_pieceImage, const int32_t in_pieceNumber) const
//Function input: in_pieceImage: piece image; in_pieceNumber: piece number
//Function output: none
//Function objective: add piece string to in_pieceImage
/************************************************************************/
{
    const std::string numberStr = "piece #" + std::to_string(in_pieceNumber);

    ImageUtils::PrintTextToPieceImage(in_pieceImage, numberStr, cv::Scalar(255, 0, 0));
}

/************************************************************************/
cv::Mat PuzzleImage::GetPuzzlePiece(const int32_t in_row, const int32_t in_col, const cv::Mat& in_image) const
/************************************************************************/
{
    const cv::Range rowRange(in_row * m_pieceSize, (in_row + 1) * m_pieceSize);
    const cv::Range colRange(in_col * m_pieceSize, (in_col + 1) * m_pieceSize);

    return in_image(rowRange, colRange);
}

/************************************************************************/
PieceNumbersAndRotationsMatrix PuzzleImage::GetShufflingMatrix(const int32_t in_numOfRows, const int32_t in_numOfCols,
    const PuzzleType in_puzzleType)
//Function input: in_numOfRows: number of rows in the output matrix; in_numOfCols: number of columns in the output matrix;
//  in_puzzleType: puzzle type
//Function output: matrix of 'PieceNumberAndRotation' elements with shuffled piece numbers and random rotations
//Function objective: described in "Function output"
/************************************************************************/
{
    //1. Init matrix
    PieceNumbersAndRotationsMatrix numbersAndRotationsMatrix = GetDefaultPieceNumbersAndRotationsMatrix(in_numOfRows, in_numOfCols);

    //2. Shuffle piece numbers
    ShufflePieceNumbersInMatrix(numbersAndRotationsMatrix);

    //3. Draw pieces rotations (if needed)
    if (in_puzzleType == PuzzleType::eType2_UknownRotation)
        DrawPieceRotationsInMatrix(numbersAndRotationsMatrix);

    return numbersAndRotationsMatrix;
}

/************************************************************************/
PuzzleImage PuzzleImage::CreateAndInitPuzzleImage(const cv::Mat& in_image, const int in_pieceSize, const PuzzleType in_puzzleType, const bool in_addLabelStrings)
//Function input: in_image: source image; in_pieceSize: piece size in pixel (for example, if in_pieceSize=3, then each piece is a 3x3 pixels square)
//  in_puzzleType: puzzle type; in_addLabelStrings: boolean indicating if whether we should add labels strings to piece
//Function output: PuzzleImage instance
//Function objective: create  puzzle from in_image, with piece size of in_pieceSize
/************************************************************************/
{
    //1. Compute number of piece per row and per column
    const int32_t numOfPiecesPerRow = ImageUtils::GetNumOfPiecesPerRow(in_image, in_pieceSize);
    const int32_t numOfPiecesPerCol = ImageUtils::GetNumOfPiecesPerCol(in_image, in_pieceSize);

    //2. Create PuzzleImage
    const PuzzleImage puzzleImage(in_image, in_pieceSize, numOfPiecesPerRow, numOfPiecesPerCol, in_puzzleType, in_addLabelStrings);

    return puzzleImage;
}

/************************************************************************/
PieceNumbersAndRotationsMatrix PuzzleImage::GetDefaultPieceNumbersAndRotationsMatrix(const int32_t in_numOfRows, const int32_t in_numOfCols)
//Function input: in_numOfRows: number of rows in the output matrix; in_numOfCols: number of columns in the output matrix
//Function output: matrix of 'PieceNumberAndRotation' elements with 'in_numOfRows' rows and 'in_numOfCols' columns, 
//  such that the piece number of top left entry is 0, the one to its right is 1, and so on. All rotations are 'e_0_degrees'
//Function objective: described in "Function output"
/************************************************************************/
{
    //1. Init matrix
    PieceNumbersAndRotationsMatrix matrix(in_numOfRows, in_numOfCols);

    //2. Set matrix values
    for (int row = 0; row < in_numOfRows; ++row)
    {
        for (int col = 0; col < in_numOfCols; ++col)
        {
            const int32_t pieceNumber = row * in_numOfCols + col;
            const ImageRotation rotation = ImageRotation::e_0_degrees;
            
            matrix(row, col) = PieceNumberAndRotation(pieceNumber, rotation);
        }
    }

    return matrix;
}

/************************************************************************/
void PuzzleImage::ShufflePieceNumbersInMatrix(PieceNumbersAndRotationsMatrix& inout_numbersAndRotationsMatrix)
//Function input: none
//Function output: none
//Function objective: shuffle entries in inout_numbersAndRotationsMatrix
/************************************************************************/
{
    //1. Convert the piece numbers in inout_numbersAndRotationsMatrix to std::vector
    std::vector<int> piecesVec;
    for (int32_t row = 0; row < inout_numbersAndRotationsMatrix.rows(); ++row)
    {
        for (int32_t col = 0; col < inout_numbersAndRotationsMatrix.cols(); ++col)
        {
            piecesVec.push_back(inout_numbersAndRotationsMatrix(row, col).m_pieceNumber);
        }
    }

    //2. Shuffled converted std::vector
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(piecesVec.begin(), piecesVec.end(), g);

    //3. Set shuffled values from std::vector back to inout_numbersAndRotationsMatrix
    const int* data_array = piecesVec.data();
    for (size_t vecIdx = 0; vecIdx < piecesVec.size(); ++vecIdx)
    {
        const RowColInfo coordinate = Utilities::RowColInfo::GetRowAndColFrom1DIndex(vecIdx, inout_numbersAndRotationsMatrix.cols());
        inout_numbersAndRotationsMatrix(coordinate.m_row, coordinate.m_col).m_pieceNumber = data_array[vecIdx];
    }
}

/************************************************************************/
void PuzzleImage::DrawPieceRotationsInMatrix(PieceNumbersAndRotationsMatrix& inout_numbersAndRotationsMatrix)
//Function input: none
//Function output: none
//Function objective: set random ImageRotation values in the entries of inout_numbersAndRotationsMatrix
/************************************************************************/
{
    //1. Use current time as seed for random generator
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    for (int32_t row = 0; row < inout_numbersAndRotationsMatrix.rows(); ++row)
    {
        for (int32_t col = 0; col < inout_numbersAndRotationsMatrix.cols(); ++col)
        {
            //2.1. Draw rotation and set it to inout_numbersAndRotationsMatrix
            inout_numbersAndRotationsMatrix(row, col).m_rotation = static_cast<ImageRotation>(std::rand() % ImageRotation::eNumOfImageRotations);
        }
    }
}

/************************************************************************/
Utilities::RowColInfo PuzzleImage::GetMatrixRowAndColFromPieceNumber(const int32_t in_pieceNumber,
    const PieceNumbersAndRotationsMatrix& in_pieceNumbersAndRotationsMatrix)
//Function input: in_pieceNumber: puzzle number; in_pieceNumbersAndRotationsMatrix: matrix
//Function output: row and column of the entry containing in_pieceNumber in in_pieceNumbersAndRotationsMatrix
//Function objective: described in "Function output"
/************************************************************************/
{
    //1. Init rowColInfo
    RowColInfo rowColInfo(RowColInfo::m_invalidRowColInfo);

    //2. Loop till entry containing in_pieceNumber is found
    bool wasFound = false;
    for (int32_t row = 0; row < in_pieceNumbersAndRotationsMatrix.rows() && !wasFound; ++row)
    {
        for (int32_t col = 0; col < in_pieceNumbersAndRotationsMatrix.cols() && !wasFound; ++col)
        {
            if (in_pieceNumber == in_pieceNumbersAndRotationsMatrix(row, col).m_pieceNumber)
            {
                rowColInfo = RowColInfo(row, col);
                wasFound = true;
            }
        }
    }

    //3. Crash if entry was not found, and return
    Utilities::LogAndAbortIf(!wasFound, "'PuzzleImage::GetMatrixRowAndColFromPieceNumber()' failed");

    return rowColInfo;
}