#include "ImageUtils.h"
#include "PuzzleRL_SolverConstants.h"
#include "Utilities.h"

const PixelType ImageUtils::Colors::m_grayLab(198, 128, 128); 

/************************************************************************/
cv::Mat ImageUtils::TruncatedImage(const cv::Mat in_image, const int32_t in_rowPixels, const int32_t in_colPixels)
//Function input: in_image: source image; in_rowPixels: number of rows pixels in truncated image;
//  in_rowPixels: number of column pixels in truncated image
//Function output: truncated image according to input
//Function objective: described in "Function output"
/************************************************************************/
{
    const cv::Mat truncatedImage = in_image(cv::Range(0, in_rowPixels), cv::Range(0, in_colPixels));
    return truncatedImage;
}

/************************************************************************/
void ImageUtils::SetPieceToImageCoordinates(const cv::Mat& in_pieceImage, const Utilities::RowColInfo& in_coordinates, cv::Mat& inout_image)
//Function input: in_pieceImage: piece image; in_coordinates: target row and column for the image to be set in 'inout_image';
//  inout_image: the image that 'in_pieceImage' should be set in
//Function output: none
//Function objective: set piece image in coordinates 'in_coordinates' in 'inout_image'
/************************************************************************/
{
    //1. Set pieceSize (number of row pixels or column pixels in in_pieceImage)
    const int32_t pieceSize = in_pieceImage.rows;

    //2. Set ranges for the piece in 'inout_image'
    const cv::Range rowRange(in_coordinates.m_row * pieceSize, (in_coordinates.m_row + 1) * pieceSize);
    const cv::Range colRange(in_coordinates.m_col * pieceSize, (in_coordinates.m_col + 1) * pieceSize);

    //3. Set piece in above ranged in 'inout_image'
    in_pieceImage.copyTo(inout_image(rowRange, colRange));
}

/************************************************************************/
void ImageUtils::SetRotatedPieceInLocation(const cv::Mat& in_pieceImage, const ImageRotation in_rotation,
    const Utilities::RowColInfo& in_rowCol, cv::Mat& inout_image)
/************************************************************************/
{
    //1. Get rotated piece
    const cv::Mat rotatedPiece = ImageUtils::RotateImage(in_pieceImage, in_rotation);

    //2. Assign rotatedPiece in its location 
    ImageUtils::SetPieceToImageCoordinates(rotatedPiece, in_rowCol, inout_image);
}

/************************************************************************/
void ImageUtils::AddPiecesGridToImage(cv::Mat& inout_image, const int32_t in_pieceSize)
//Function input: inout_image: image to add pieces grid to; in_pieceSize: piece size
//Function output: none
//Function objective: add pieces grid to 'inout_image'
/************************************************************************/
{
    //Pay attention to avoid confusion: in OpenCv a point is (x, y) (and not (row, col), like in Eigen)

    //1. Init variables
    const cv::Vec3b blackColorLab = cv::Vec3b(0, 128, 128);
    constexpr int32_t lineThickness = 1;

    //2. Add rows grid
    const int32_t numOfRowPieces = GetNumOfPiecesPerRow(inout_image, in_pieceSize);
    for (int i = 1; i < numOfRowPieces; ++i)
    {
        const int32_t rowY_Coord = (i * in_pieceSize) - 1;
        const cv::Point leftPoint(0, rowY_Coord);
        const cv::Point rightPoint(inout_image.cols - 1, rowY_Coord);
        cv::line(inout_image, leftPoint, rightPoint, blackColorLab, lineThickness);
    }

    //3. Add cols grid
    const int32_t numOfColPieces = GetNumOfPiecesPerCol(inout_image, in_pieceSize);
    for (int j = 1; j < numOfColPieces; ++j)
    {
        const int32_t colX_Coord = (j * in_pieceSize) - 1;
        const cv::Point leftPoint(colX_Coord, 0);
        const cv::Point rightPoint(colX_Coord, inout_image.rows - 1);
        cv::line(inout_image, leftPoint, rightPoint, blackColorLab, lineThickness);
    }
}

/************************************************************************/
void ImageUtils::PrintTextToPieceImage(cv::Mat& inout_pieceImage, const std::string& in_text, const cv::Scalar& in_textColor, const bool in_checkPixelChange)
//Function input: inout_pieceImage: piece image; in_text: text to print; in_textColor: text color;
    //in_checkPixelChange: boolean indicating to check if important pixels were changed
//Function output: none
//Function objective: add the text in 'in_text' in color 'in_textColor' to 'inout_pieceImage'
/************************************************************************/
{
    //1. Set some algorithm information (the algorithm uses only the two extreme rows and columns for each piece)
    const int32_t pieceSize = inout_pieceImage.rows;
    constexpr int32_t numOfPixelEdgesUsedInAlgorithm = 3;
    constexpr int32_t mostLeftAndTopTextPixelIndexPossible = numOfPixelEdgesUsedInAlgorithm;
    const int32_t mostRightAndBottomTextPixelIndexPossible = pieceSize - 1 - numOfPixelEdgesUsedInAlgorithm;

    //2. Set x and y offsets (we want them to be no smaller than 'mostLeftAndTopTextPixelIndexPossible' - so the information used by the algorithm is not changed)
    const int32_t xProportionalOffest = std::max(mostLeftAndTopTextPixelIndexPossible, static_cast<int32_t>(pieceSize * 0.1));
    const int32_t yProportionalOffest = std::max(mostLeftAndTopTextPixelIndexPossible, static_cast<int32_t>(pieceSize * 0.3));

    //3. Set the point to write txt in in_pieceImage
    const cv::Point pointToWriteTxt(xProportionalOffest, yProportionalOffest);

    //4. Determine desired scale
    const cv::Size textSizeWith1Scale = cv::getTextSize(in_text, cv::FONT_HERSHEY_SIMPLEX, 1, 1, nullptr);
    const double desiredScale = static_cast<double>(-2 * (xProportionalOffest + 1) + pieceSize) / static_cast<double>(textSizeWith1Scale.width);

    //5. Check that desired scale is ok
    const cv::Size actualTextSize = cv::getTextSize(in_text, cv::FONT_HERSHEY_SIMPLEX, desiredScale, 1, nullptr);
    const cv::Size actualTextBottomRightCorner = actualTextSize + cv::Size(xProportionalOffest, yProportionalOffest);
    Utilities::LogAndAbortIf(actualTextBottomRightCorner.height > mostRightAndBottomTextPixelIndexPossible || 
        actualTextBottomRightCorner.width > mostRightAndBottomTextPixelIndexPossible,
        "Too big piece image text - changes algorithm behavior");

    const cv::Mat in_pieceImage = inout_pieceImage.clone();

    //6. Write text
    cv::putText(inout_pieceImage, in_text, pointToWriteTxt, cv::FONT_HERSHEY_SIMPLEX, desiredScale, in_textColor);

    //7. Make sure we changed only non-important pixels
    if (in_checkPixelChange)
    {
        const std::vector<ExtendedPixelType> bottomInPiece = GetBottomEdgePixelsForDissimilarityComputation(in_pieceImage); 
        const std::vector<ExtendedPixelType> bottomModifiedPiece = GetBottomEdgePixelsForDissimilarityComputation(inout_pieceImage); 
        Utilities::LogAndAbortIf(bottomInPiece != bottomModifiedPiece, "In 'ImageUtils::PrintTextToPieceImage()': bottomInPiece != bottomModifiedPiece");

        const std::vector<ExtendedPixelType> topInPiece = GetTopEdgePixelsForDissimilarityComputation(in_pieceImage); 
        const std::vector<ExtendedPixelType> topModifiedPiece = GetTopEdgePixelsForDissimilarityComputation(inout_pieceImage); 
        Utilities::LogAndAbortIf(topInPiece != topModifiedPiece, "In 'ImageUtils::PrintTextToPieceImage()': topInPiece != topModifiedPiece");

        const std::vector<ExtendedPixelType> rightInPiece = GetRightEdgePixelsForDissimilarityComputation(in_pieceImage); 
        const std::vector<ExtendedPixelType> rightModifiedPiece = GetRightEdgePixelsForDissimilarityComputation(inout_pieceImage); 
        Utilities::LogAndAbortIf(rightInPiece != rightModifiedPiece, "In 'ImageUtils::PrintTextToPieceImage()': rightInPiece != rightModifiedPiece");

        const std::vector<ExtendedPixelType> leftInPiece = GetLeftEdgePixelsForDissimilarityComputation(in_pieceImage); 
        const std::vector<ExtendedPixelType> leftModifiedPiece = GetLeftEdgePixelsForDissimilarityComputation(inout_pieceImage); 
        Utilities::LogAndAbortIf(leftInPiece != leftModifiedPiece, "In 'ImageUtils::PrintTextToPieceImage()': leftInPiece != leftModifiedPiece");
    }
}

/************************************************************************/
cv::Mat ImageUtils::RotateImage(const cv::Mat& in_image, const ImageRotation in_rotation)
//Function input: in_image: input image; in_rotation: rotation that should be applied to in_image
//Function output: in_image after being rotated according to in_rotation
//Function objective: described in "Function output"
/************************************************************************/
{
    cv::Mat rotatedImage;

    switch (in_rotation)
    {
    //Should do nothing in this case
    case ImageRotation::e_0_degrees:
        rotatedImage = in_image;
        break;

    case ImageRotation::e_90_degrees:
        cv::rotate(in_image, rotatedImage, cv::ROTATE_90_CLOCKWISE);
        break;

    case ImageRotation::e_180_degrees:
        cv::rotate(in_image, rotatedImage, cv::ROTATE_180);
        break;

    case ImageRotation::e_270_degrees:
        cv::rotate(in_image, rotatedImage, cv::ROTATE_90_COUNTERCLOCKWISE);
        break;

    default:
        Utilities::LogAndAbort("ImageUtils::RotateImage() failed");
    }

    return rotatedImage;
}

/************************************************************************/
PieceNumbersAndRotationsMatrix ImageUtils::RotatePuzzleAssignmentMatrix(const PieceNumbersAndRotationsMatrix& in_puzzleAssignmentMatrix, const ImageRotation in_rotation)
//Function input: in_puzzleAssignmentMatrix: input matrix to rotate; in_rotation: rotation that should be applied to in_puzzleAssignmentMatrix
//Function output: in_puzzleAssignmentMatrix rotated by in_rotation
//Function objective: described in "Function output"
/************************************************************************/
{
    PieceNumbersAndRotationsMatrix rotatedAssignmentMatrix = in_puzzleAssignmentMatrix;
    for (int i = 0; i < in_rotation; ++i)
        rotatedAssignmentMatrix = RotateAssignmetMatrixBy90DegreesClockwise(rotatedAssignmentMatrix);

    return rotatedAssignmentMatrix;
}

/************************************************************************/
PieceNumbersAndRotationsMatrix ImageUtils::RotateAssignmetMatrixBy90DegreesClockwise(const PieceNumbersAndRotationsMatrix& in_puzzleAssignmentMatrix)
/************************************************************************/
{
    //Cols and rows are switched
    PieceNumbersAndRotationsMatrix rotatedMatrix = 
        PieceNumbersAndRotationsMatrix(in_puzzleAssignmentMatrix.cols(), in_puzzleAssignmentMatrix.rows());

    for (int32_t row = 0; row < in_puzzleAssignmentMatrix.rows(); ++row)
    {
        const int32_t destinationCol = static_cast<int32_t>(in_puzzleAssignmentMatrix.rows()) - 1 - row;

        for (int32_t col = 0; col < in_puzzleAssignmentMatrix.cols(); ++col)
        {
            const int32_t destinationRow = col;

            PieceNumberAndRotation pieceNumberAndRotation = in_puzzleAssignmentMatrix(row, col);
            if (pieceNumberAndRotation.m_pieceNumber != -1)
                pieceNumberAndRotation.m_rotation = SumSquareImageRotations(pieceNumberAndRotation.m_rotation, ImageRotation::e_90_degrees);

            rotatedMatrix(destinationRow, destinationCol) = pieceNumberAndRotation;
        }
    }

    return rotatedMatrix;
}

/************************************************************************/
RotationsVec ImageUtils::GetRotationsToCheck(const bool in_isSquarePuzzle, const PuzzleType in_puzzleType)
/************************************************************************/
{
    RotationsVec rotationsToCheck;

    //1. If it's type 1 puzzle - there's only one rotation to check (the 'no-rotation' state)
    if (PuzzleType::eType1_NoRotation == in_puzzleType)
    {
        rotationsToCheck = { ImageRotation::e_0_degrees };
    }
    //2. If it's type 2 puzzle
    else
    {
        //2.1. If it's a square puzzle - the puzzle may be rotated in all 4 rotation
        if (in_isSquarePuzzle)
            rotationsToCheck = { ImageRotation::e_0_degrees, ImageRotation::e_90_degrees,
                ImageRotation::e_180_degrees, ImageRotation::e_270_degrees };
        //2.2. If it's non-square puzzle - the puzzle may be rotated 0 or 180 degrees (90 or 270 degrees rotation will 'change' the puzzle dimension)
        else
            rotationsToCheck = { ImageRotation::e_0_degrees, ImageRotation::e_180_degrees };
    }

    return rotationsToCheck;
}

/************************************************************************/
std::vector<ExtendedPixelType> ImageUtils::GetBottomEdgePixelsForDissimilarityComputation(const cv::Mat& in_piece)
//Function input: in_piece: puzzle piece
//Function output: get edge pixels that take part in dissimilarity computation of bottom edge (dissimilarity is 'DISSIMILARITY_TYPE')
//Function objective: described in "Function output"
/************************************************************************/
{
    std::vector<ExtendedPixelType> bottomEdgePixelsVec;

    switch (DISSIMILARITY_TYPE)
    {
        case DissimilarityType::eImprovedMGC:
        {
            bottomEdgePixelsVec = GetAsExtendedPixelVector(GetBottomEdgePixels(in_piece));
            const std::vector<ExtendedPixelType> secondBottomEdgePixelsVec = GetAsExtendedPixelVector(GetSecondBottomEdge(in_piece));
            bottomEdgePixelsVec.insert(bottomEdgePixelsVec.end(), secondBottomEdgePixelsVec.begin(), secondBottomEdgePixelsVec.end());
            break;
        }

        default:
            Utilities::LogAndAbort("'ImageUtils::GetBottomEdgePixelsForDissimilarityComputation()': Shouldn't get here");
            break;
    }

    return bottomEdgePixelsVec;
}

/************************************************************************/
std::vector<ExtendedPixelType> ImageUtils::GetTopEdgePixelsForDissimilarityComputation(const cv::Mat& in_piece)
//Function input: in_piece: puzzle piece
//Function output: get edge pixels that take part in dissimilarity computation of top edge (dissimilarity is 'DISSIMILARITY_TYPE')
//Function objective: described in "Function output"
/************************************************************************/
{
    std::vector<ExtendedPixelType> topEdgePixelsVec;

    switch (DISSIMILARITY_TYPE)
    {
        case DissimilarityType::eImprovedMGC:
        {
            topEdgePixelsVec = GetAsExtendedPixelVector(GetTopEdgePixels(in_piece));
            const std::vector<ExtendedPixelType> secondTopEdgePixelsVec = GetAsExtendedPixelVector(GetSecondTopEdge(in_piece));
            topEdgePixelsVec.insert(topEdgePixelsVec.end(), secondTopEdgePixelsVec.begin(), secondTopEdgePixelsVec.end());
            break;
        }

        default:
            Utilities::LogAndAbort("'ImageUtils::GetTopEdgePixelsForDissimilarityComputation()': Shouldn't get here");
            break;
    }

    return topEdgePixelsVec;
}

/************************************************************************/
std::vector<ExtendedPixelType> ImageUtils::GetRightEdgePixelsForDissimilarityComputation(const cv::Mat& in_piece)
//Function input: in_piece: puzzle piece
//Function output: get edge pixels that take part in dissimilarity computation of right edge (dissimilarity is 'DISSIMILARITY_TYPE')
//Function objective: described in "Function output"
/************************************************************************/
{
    std::vector<ExtendedPixelType> rightEdgePixelsVec;

    switch (DISSIMILARITY_TYPE)
    {
        case DissimilarityType::eImprovedMGC:
        {
            rightEdgePixelsVec = GetAsExtendedPixelVector(GetRightEdgePixels(in_piece));
            const std::vector<ExtendedPixelType> secondRightEdgePixelsVec = GetAsExtendedPixelVector(GetSecondRightEdge(in_piece));
            rightEdgePixelsVec.insert(rightEdgePixelsVec.end(), secondRightEdgePixelsVec.begin(), secondRightEdgePixelsVec.end());
            break;
        }

        default:
            Utilities::LogAndAbort("'ImageUtils::GetRightEdgePixelsForDissimilarityComputation()': Shouldn't get here");
            break;
    }

    return rightEdgePixelsVec;
}

/************************************************************************/
std::vector<ExtendedPixelType> ImageUtils::GetLeftEdgePixelsForDissimilarityComputation(const cv::Mat& in_piece)
//Function input: in_piece: puzzle piece
//Function output: get edge pixels that take part in dissimilarity computation of left edge (dissimilarity is 'DISSIMILARITY_TYPE')
//Function objective: described in "Function output"
/************************************************************************/
{
    std::vector<ExtendedPixelType> leftEdgePixelsVec;

    switch (DISSIMILARITY_TYPE)
    {
        case DissimilarityType::eImprovedMGC:
        {
            leftEdgePixelsVec = GetAsExtendedPixelVector(GetLeftEdgePixels(in_piece));
            const std::vector<ExtendedPixelType> secondLeftEdgePixelsVec = GetAsExtendedPixelVector(GetSecondLeftEdge(in_piece));
            leftEdgePixelsVec.insert(leftEdgePixelsVec.end(), secondLeftEdgePixelsVec.begin(), secondLeftEdgePixelsVec.end());
            break;
        }

        default:
            Utilities::LogAndAbort("'ImageUtils::GetLeftEdgePixelsForDissimilarityComputation()': Shouldn't get here");
            break;
    }
    
    return leftEdgePixelsVec;
}

/************************************************************************/
std::vector<PixelType> ImageUtils::GetBottomEdgePixels(const cv::Mat& in_piece)
/************************************************************************/
{
    const int32_t lastRowIndex = in_piece.rows - 1;
    const std::vector<PixelType> vec = GetRowOfPixels(in_piece, lastRowIndex);

    return vec;
}

/************************************************************************/
std::vector<PixelType> ImageUtils::GetTopEdgePixels(const cv::Mat& in_piece)
/************************************************************************/
{
    const int32_t firstRowIndex = 0;
    const std::vector<PixelType> vec = GetRowOfPixels(in_piece, firstRowIndex);

    return vec;
}

/************************************************************************/
std::vector<PixelType> ImageUtils::GetRightEdgePixels(const cv::Mat& in_piece)
/************************************************************************/
{
    const int32_t lastColIndex = in_piece.cols - 1;
    const std::vector<PixelType> vec = GetColumnOfPixels(in_piece, lastColIndex);

    return vec;
}

/************************************************************************/
std::vector<PixelType> ImageUtils::GetLeftEdgePixels(const cv::Mat& in_piece)
/************************************************************************/
{
    const int32_t firstColIndex = 0;
    const std::vector<PixelType> vec = GetColumnOfPixels(in_piece, firstColIndex);

    return vec;
}

/************************************************************************/
std::vector<PixelType> ImageUtils::GetSecondBottomEdge(const cv::Mat& in_piece)
/************************************************************************/
{
    const int32_t secondBottomRowIndex = in_piece.rows - 2;
    return GetRowOfPixels(in_piece, secondBottomRowIndex);
}

/************************************************************************/
std::vector<PixelType> ImageUtils::GetSecondTopEdge(const cv::Mat& in_piece)
/************************************************************************/
{
    const int32_t secondTopRowIndex = 1;
    return GetRowOfPixels(in_piece, secondTopRowIndex);
}

/************************************************************************/
std::vector<PixelType> ImageUtils::GetSecondRightEdge(const cv::Mat& in_piece)
/************************************************************************/
{
    const int32_t secondRightColIndex = in_piece.cols - 2;
    return GetColumnOfPixels(in_piece, secondRightColIndex);
}

/************************************************************************/
std::vector<PixelType> ImageUtils::GetSecondLeftEdge(const cv::Mat& in_piece)
/************************************************************************/
{
    const int32_t secondLeftColIndex = 1;
    return GetColumnOfPixels(in_piece, secondLeftColIndex);
}

/************************************************************************/
std::vector<PixelType> ImageUtils::GetRowOfPixels(const cv::Mat& in_piece, const int32_t in_row)
/************************************************************************/
{
    std::vector<PixelType> vec;
    for (int32_t i = 0; i < in_piece.cols; ++i)
        vec.push_back(in_piece.at<PixelType>(in_row, i));

    return vec;
}

/************************************************************************/
std::vector<PixelType> ImageUtils::GetColumnOfPixels(const cv::Mat& in_piece, const int32_t in_col)
/************************************************************************/
{
    std::vector<PixelType> vec;
    for (int32_t i = 0; i < in_piece.rows; ++i)
        vec.push_back(in_piece.at<PixelType>(i, in_col));

    return vec;
}

/************************************************************************/
std::vector<ExtendedPixelType> ImageUtils::GetAsExtendedPixelVector(const std::vector<PixelType>& in_vec)
/************************************************************************/
{
    std::vector<ExtendedPixelType> result;
    for (int32_t i = 0; i < static_cast<int32_t>(in_vec.size()); ++i)
    {
        cv::Vec3s exPixelVal;
        for (int32_t color = 0; color < numOfChannels; ++color)
        {
            exPixelVal[color] = static_cast<short>(in_vec[i][color]);
        }
        result.push_back(exPixelVal);
    }

    return result;
}

/************************************************************************/
cv::Mat ImageUtils::GetImageMatrixFromAssignementAndPieces(const PieceNumbersAndRotationsMatrix& in_assignmentMatrix, const int32_t in_pieceSize, 
    const std::map<int32_t, cv::Mat>& in_piecesMap)
/************************************************************************/
{
    cv::Mat image = cv::Mat(static_cast<int32_t>(in_assignmentMatrix.rows()) * in_pieceSize, 
        static_cast<int32_t>(in_assignmentMatrix.cols()) * in_pieceSize, ImageUtils::imagesType, ImageUtils::Colors::m_grayLab);

    for (int32_t i = 0; i < in_assignmentMatrix.rows(); ++i)
    {
        for (int32_t j = 0; j < in_assignmentMatrix.cols(); ++j)
        {
            const PieceNumberAndRotation currPieceNumAndRot = in_assignmentMatrix(i, j);
            if (currPieceNumAndRot.IsValid())
            {
                const cv::Mat currPiece = in_piecesMap.at(currPieceNumAndRot.m_pieceNumber);
                SetRotatedPieceInLocation(currPiece, currPieceNumAndRot.m_rotation, Utilities::RowColInfo(i, j), image);
            }

        }
    }

    return image;
}

/************************************************************************/
int32_t ImageUtils::GetNumOfAssignedPiecesInPieceAssignmentMatrix(const PieceNumbersAndRotationsMatrix& in_assignmentMatrix)
//Function input: in_assignmentMatrix: input assignment matrix
//Function output: number of pieces assigned in 'in_assignmentMatrix'
//Function objective: described in "Function output"
/************************************************************************/
{
    int32_t numOfAssignedPieces = 0;

    for (int32_t i = 0; i < in_assignmentMatrix.rows(); ++i)
    {
        for (int32_t j = 0; j < in_assignmentMatrix.cols(); ++j)
        {
            const PieceNumberAndRotation currPieceNumberAndRotation = in_assignmentMatrix(i, j);
            if (currPieceNumberAndRotation.IsValid())
                ++numOfAssignedPieces;
        }
    }

    return numOfAssignedPieces;
}

/************************************************************************/
double ImageUtils::ComputeDirectComparison(const GroundTruthSolutionInfo& in_groundTruthSolutionInfo, const PieceNumbersAndRotationsMatrix& in_solutionMatrix,
    const int32_t in_numOfPieces, RowColInfoSet& out_WrongAssignedPieceCoordsSet)
//Function input: in_groundTruthSolutionInfo: ground truth solution info; in_solutionMatrix: matrix representing a suggested solution; 
    //in_numOfPieces: number of puzzle piece;
    //out_WrongAssignedPieceCoordsSet: set to assign in wrong coordinates in solution
//Function output: number in the range [0,1] representing the direct comparison performance measure of the solution in in_solutionMatrix
//Function objective: described in "Function output"
/************************************************************************/
{
    //1. Init variables
    int32_t correctPiecesPlacements = 0;

    //2. Count number of correct assignments between ground truth solution and algorithm solution
    for (int i = 0; i < in_solutionMatrix.rows(); ++i)
    {
        for (int j = 0; j < in_solutionMatrix.cols(); ++j)
        {
            const PieceNumberAndRotation groundTruthPieceNumAndRot = in_groundTruthSolutionInfo.m_assignmentMatrix(i, j);
            const PieceNumberAndRotation solutionPieceNumAndRot = in_solutionMatrix(i, j);

            //2.1. If the ground truth rotated piece and the solution rotated piece are identical
            bool foundMatch = false;
            if (groundTruthPieceNumAndRot == solutionPieceNumAndRot)
            {
                foundMatch = true;
            }

            if (foundMatch)
                ++correctPiecesPlacements;
            else if (solutionPieceNumAndRot.IsValid())
                out_WrongAssignedPieceCoordsSet.emplace(i, j);
        }
    }

    //3. Direct comparison is the num of correct assignments, divided by the total num of pieces
    const double directComparison = static_cast<double>(correctPiecesPlacements) / static_cast<double>(in_numOfPieces);

    return directComparison;
}

/************************************************************************/
double ImageUtils::ComputeNeighborComparison(const GroundTruthSolutionInfo& in_groundTruthSolutionInfo, 
    const PieceNumbersAndRotationsMatrix& in_solutionMatrix)
//Function input: in_groundTruthSolutionInfo: ground truth solution info; in_solutionMatrix: matrix representing a suggested solution 
//Function output: number in the range [0,1] representing the neighbor comparison performance measure of the solution in in_solutionMatrix
//Function objective: described in "Function output"
/************************************************************************/
{
    //1. Init variables
    int32_t numOfCorrectNeighboringRelations = 0;

    //2. Check horizontal relations
    for (int32_t i = 0; i < in_solutionMatrix.rows(); ++i)
    {
        for (int32_t j = 0; j < in_solutionMatrix.cols() - 1; ++j)
        {
            const PieceNumberAndRotation currSolPieceNumAndRot = in_solutionMatrix(i, j);

            if (currSolPieceNumAndRot.IsValid())
            {
                const PieceNumberAndRotation currRightNeighbor = in_solutionMatrix(i, j + 1);

                if (currRightNeighbor.IsValid())
                {
                    const bool areRealNeighbors = 
                        in_groundTruthSolutionInfo.DoesRelationExistInGroundTruth(currSolPieceNumAndRot, currRightNeighbor, eRight);

                    if (areRealNeighbors)
                        ++numOfCorrectNeighboringRelations;
                }
            }
        }
    }

    //3. Check vertical relations
    for (int32_t j = 0; j < in_solutionMatrix.cols(); ++j)
    {
        for (int32_t i = 0; i < in_solutionMatrix.rows() - 1; ++i)
        {
            const PieceNumberAndRotation currSolPieceNumAndRot = in_solutionMatrix(i, j);

            if (currSolPieceNumAndRot.IsValid())
            {
                const PieceNumberAndRotation currBottomNeighbor = in_solutionMatrix(i + 1, j);

                if (currBottomNeighbor.IsValid())
                {
                    const bool areRealNeighbors = 
                        in_groundTruthSolutionInfo.DoesRelationExistInGroundTruth(currSolPieceNumAndRot, currBottomNeighbor, eDown);

                    if (areRealNeighbors)
                        ++numOfCorrectNeighboringRelations;
                }
            }
        }
    }

    //4. Calc neighbor comparison
    const size_t numOfHorizontalRelations = in_solutionMatrix.rows() * (in_solutionMatrix.cols() - 1); 
    const size_t numOfVerticalRelations = in_solutionMatrix.cols() * (in_solutionMatrix.rows() - 1);
    const size_t totalNumOfNeighboringRelations = numOfHorizontalRelations + numOfVerticalRelations;

    const double neighborComparison = static_cast<double>(numOfCorrectNeighboringRelations) / static_cast<double>(totalNumOfNeighboringRelations);

    return neighborComparison;
}

/************************************************************************/
ImageRotation ImageUtils::GetRotationDifference(const ImageRotation& in_firstRotation, const ImageRotation& in_secondRotation)
//rotationDifference is the rotation that should be applied for in_secondRotation so it will have the same rotation as in_firstRotation
/************************************************************************/
{
    int32_t integerDiff = (static_cast<int32_t>(in_firstRotation) - static_cast<int32_t>(in_secondRotation)) 
        % static_cast<int32_t>(ImageRotation::eNumOfImageRotations);

    if (integerDiff < 0)
        integerDiff += ImageRotation::eNumOfImageRotations;

    const ImageRotation rotationDiff = static_cast<ImageRotation>(integerDiff);

    return rotationDiff;
}

/************************************************************************/
ImageRotation ImageUtils::GetCounterRotation(const ImageRotation in_rotation)
/************************************************************************/
{
    ImageRotation counterRotation = e_0_degrees;

    switch (in_rotation)
    {
    case ImageRotation::e_0_degrees:
        counterRotation = e_0_degrees;
        break;

    case ImageRotation::e_90_degrees:
        counterRotation = e_270_degrees;
        break;

    case ImageRotation::e_180_degrees:
        counterRotation = e_180_degrees;
        break;

    case ImageRotation::e_270_degrees:
        counterRotation = e_90_degrees;
        break;

    default:
        Utilities::LogAndAbort("ImageUtils::GetCounterRotation() failed");
    }

    return counterRotation;
}

/************************************************************************/
cv::Mat ImageUtils::ReadImage(const std::string& in_file)
/************************************************************************/
{
    //1. Read image
    cv::Mat image = cv::imread(in_file, cv::IMREAD_COLOR);

    //2. Convert the image from BGR to LAB
    cv::cvtColor(image, image, colorConversionFromBGR);

    return image;
}

/************************************************************************/
void ImageUtils::WriteImage(const std::string& in_file, const cv::Mat& in_image)
/************************************************************************/
{
    //2. Convert the image from LAB to BGR
    cv::Mat convertedImage;
    cv::cvtColor(in_image, convertedImage, colorConversionToBGR);
    
    //2. Write image
    const bool didWriteSucceed = cv::imwrite(in_file, convertedImage);
    Utilities::LogAndAbortIf(!didWriteSucceed, "'ImageUtils::WriteImage()': writing '" + in_file + "' failed");
}

/************************************************************************/
ImageRotation ImageUtils::SumSquareImageRotations(const ImageRotation in_firstRotation, const ImageRotation in_secondRotation)
/************************************************************************/
{
    const int32_t intResult = (static_cast<int32_t>(in_firstRotation) + static_cast<int32_t>(in_secondRotation))
        % static_cast<int32_t>(ImageRotation::eNumOfImageRotations);

    const ImageRotation result = static_cast<ImageRotation>(intResult);

    return result;
}

/************************************************************************/
ImageRotation ImageUtils::GetRotationFromSourceAndDestEdgeOrientations(const Orientation in_sourceOrien, const Orientation in_destOrien)
//Function input: in_sourceOrien: source edge orientation; in_destOrien: destination edge orientation
//Function output: the rotation the should be applied to change edge orientation from 'in_sourceOrien' to 'in_destOrien'
//Function objective: described in "Function output"
/************************************************************************/
{
    ImageRotation imageRotation = ImageRotation::eInvalidRotation;

    switch (in_sourceOrien)
    {
    case eUp:
        if (eUp == in_destOrien)
            imageRotation = ImageRotation::e_0_degrees;
        else if (eRight == in_destOrien)
            imageRotation = ImageRotation::e_90_degrees;
        else if (eDown == in_destOrien)
            imageRotation = ImageRotation::e_180_degrees;
        else if (eLeft == in_destOrien)
            imageRotation = ImageRotation::e_270_degrees;
        else
            Utilities::LogAndAbort("");
        break;

    case eRight:
        if (eUp == in_destOrien)
            imageRotation = ImageRotation::e_270_degrees;
        else if (eRight == in_destOrien)
            imageRotation = ImageRotation::e_0_degrees;
        else if (eDown == in_destOrien)
            imageRotation = ImageRotation::e_90_degrees;
        else if (eLeft == in_destOrien)
            imageRotation = ImageRotation::e_180_degrees;
        else
            Utilities::LogAndAbort("");
        break;

    case eDown:
        if (eUp == in_destOrien)
            imageRotation = ImageRotation::e_180_degrees;
        else if (eRight == in_destOrien)
            imageRotation = ImageRotation::e_270_degrees;
        else if (eDown == in_destOrien)
            imageRotation = ImageRotation::e_0_degrees;
        else if (eLeft == in_destOrien)
            imageRotation = ImageRotation::e_90_degrees;
        else
            Utilities::LogAndAbort("");
        break;

    case eLeft:
        if (eUp == in_destOrien)
            imageRotation = ImageRotation::e_90_degrees;
        else if (eRight == in_destOrien)
            imageRotation = ImageRotation::e_180_degrees;
        else if (eDown == in_destOrien)
            imageRotation = ImageRotation::e_270_degrees;
        else if (eLeft == in_destOrien)
            imageRotation = ImageRotation::e_0_degrees;
        else
            Utilities::LogAndAbort("");
        break;

    default:
        Utilities::LogAndAbort("");
    }

    return imageRotation;
}

/************************************************************************/
PixelType ImageUtils::ConvertPixelFromLAB_ToBGR(const PixelType& in_pixelLAB)
/************************************************************************/
{
    cv::Mat dummyImage(1, 1, imagesType);

    dummyImage.at<PixelType>(0, 0) = in_pixelLAB;
    cv::cvtColor(dummyImage, dummyImage, colorConversionToBGR);
    const PixelType pixelBGR = dummyImage.at<PixelType>(0, 0);

    return pixelBGR;
}

/************************************************************************/
PixelType ImageUtils::ConvertPixelFromBGR_ToLAB(const PixelType& in_pixelBGR)
/************************************************************************/
{
    cv::Mat dummyImage(1, 1, imagesType);

    dummyImage.at<PixelType>(0, 0) = in_pixelBGR;
    cv::cvtColor(dummyImage, dummyImage, colorConversionFromBGR);
    const PixelType pixelLAB = dummyImage.at<PixelType>(0, 0);

    return pixelLAB;
}

/************************************************************************/
bool ImageUtils::GroundTruthSolutionInfo::DoesRelationExistInGroundTruth(const PieceNumberAndRotation& in_firstPiece,
    const PieceNumberAndRotation& in_secondPiece, const Orientation& in_orien) const
//Function input: in_firstPiece: first piece; in_secondPiece: second piece; in_orien: orientation;
//Function output: boolean indicating whether 'in_firstPiece' and 'in_secondPiece' are neighbors in 'in_orien' in ground truth solution
    //(and we also take rotation in account)
//Function objective: described in "Function output"
/************************************************************************/
{
    //1. Find rotations and coordinates of 'in_firstPiece' ad 'in_secondPiece' in GT
    ImageRotation firstPieceRotationInGT = ImageRotation::eInvalidRotation;
    RowColInfo firstPieceCoord = RowColInfo::m_invalidRowColInfo;

    ImageRotation secondPieceRotationInGT = ImageRotation::eInvalidRotation;
    RowColInfo secondPieceCoord = RowColInfo::m_invalidRowColInfo;

    for (int32_t i = 0; i < m_assignmentMatrix.rows(); ++i)
    {
        for (int32_t j = 0; j < m_assignmentMatrix.cols(); ++j)
        {
            const PieceNumberAndRotation currPieceNumberAndRotation = m_assignmentMatrix(i, j);
            if (currPieceNumberAndRotation.m_pieceNumber == in_firstPiece.m_pieceNumber)
            {
                firstPieceRotationInGT = currPieceNumberAndRotation.m_rotation;
                firstPieceCoord = RowColInfo(i, j);
            }
                                        
            if (currPieceNumberAndRotation.m_pieceNumber == in_secondPiece.m_pieceNumber)
            {
                secondPieceRotationInGT = currPieceNumberAndRotation.m_rotation;
                secondPieceCoord = RowColInfo(i, j);
            }
        }
    }

    Utilities::LogAndAbortIf(ImageRotation::eInvalidRotation == firstPieceRotationInGT || ImageRotation::eInvalidRotation == secondPieceRotationInGT 
        || !firstPieceCoord.IsValid() || !secondPieceCoord.IsValid(), "In 'ImageUtils::GroundTruthSolutionInfo::DoesRelationExistInGroundTruth()'");

    //2. If piece are not neighbors in GT - return false
    const Orientation neighboringRelation = RLSolverGeneralUtils::GetNeighboringRelation(firstPieceCoord, secondPieceCoord);
    if (eInvalidOrientation == neighboringRelation)
        return false;

    //3. Get rotation differences (for example, firstRotationDifference is
    //the rotation that should be applied for 'in_firstPiece' so it will have the same rotation as the same piece in the GT solution)
    const ImageRotation firstRotationDifference = GetRotationDifference(firstPieceRotationInGT, in_firstPiece.m_rotation);
    const ImageRotation secondRotationDifference = GetRotationDifference(secondPieceRotationInGT, in_secondPiece.m_rotation);

    //4. If the we should apply the same rotation for the two pieces in order to make them have the rotation of the pieces in the GT
    bool retVal = false;
    if (firstRotationDifference == secondRotationDifference)
    {
        //5. If the pieces are in the correct neighboring relation
        const ImageRotation rotationToConvertOrientations = ImageUtils::GetRotationFromSourceAndDestEdgeOrientations(in_orien, neighboringRelation);
        if (firstRotationDifference == rotationToConvertOrientations)
            retVal = true;
    }

    return retVal;
}