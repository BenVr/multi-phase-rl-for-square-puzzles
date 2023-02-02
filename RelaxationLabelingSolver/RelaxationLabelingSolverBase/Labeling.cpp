#include "Labeling.h"
#include "RelaxationLabelingSolverBaseConstants.h"

std::vector<const Object*> Labeling::m_objectsIndicesVector;
std::vector<const Label*> Labeling::m_labelsIndicesVector;

/************************************************************************/
Labeling::Labeling(const size_t in_numOfObjects, const size_t in_numOfLabels, const double in_initValue) :
    m_labelingMatrix(Eigen::MatrixXd::Constant(in_numOfObjects, in_numOfLabels, in_initValue))
/************************************************************************/
{
}

/************************************************************************/
Labeling::Labeling(const Eigen::MatrixXd& in_labelingMatrix) :
    m_labelingMatrix(in_labelingMatrix)
/************************************************************************/
{
}

/************************************************************************/
LabelAndProbPair Labeling::GetMaxLabelAndProb(const Object& in_object) const
/************************************************************************/
{
    const auto& objectRow = m_labelingMatrix.row(in_object.m_index);
    int32_t indexOfMaxCoeff;
    const double maxProb = objectRow.maxCoeff(&indexOfMaxCoeff);

    return LabelAndProbPair(m_labelsIndicesVector[indexOfMaxCoeff], maxProb);
}

/************************************************************************/
double Labeling::MultiplyAndSumLabelings(const Labeling& in_firstLabeling, const Labeling& in_secondLabeling)
/************************************************************************/
{
    //It is more accurate to sum in ascending order
    //(see: https://stackoverflow.com/questions/6699066/in-which-order-should-floats-be-added-to-get-the-most-precise-result),
    //but computationally expensive
    constexpr bool moreAccurateMethod = false;

    if constexpr (moreAccurateMethod)
    {
        std::multiset<double> valuesMultiset;

        for (int32_t i = 0; i < in_firstLabeling.m_labelingMatrix.rows(); ++i)
        {
            for (int32_t j = 0; j < in_firstLabeling.m_labelingMatrix.cols(); ++j)
            {
                const double currFirstVal = in_firstLabeling.m_labelingMatrix(i, j);
                const double currSecondVal= in_secondLabeling.m_labelingMatrix(i, j);
                const double mult = currFirstVal * currSecondVal;
                valuesMultiset.insert(mult);
            }
        }

        double sum = 0.0;
        for (double currVal: valuesMultiset)
            sum += currVal;

        return sum;
    }
    else
    {
        return (in_firstLabeling.m_labelingMatrix.array() * in_secondLabeling.m_labelingMatrix.array()).sum();
    }
}

/************************************************************************/
void Labeling::SetStaticData(const ObjectsVector& in_objectsPtrsVec, const LabelsVector& in_labelsPtrsVec)
/************************************************************************/
{
    //1. Set m_objectsIndicesVector and m_labelsIndicesVector
    Labeling::m_objectsIndicesVector.assign(in_objectsPtrsVec.size(), nullptr);
    for (const Object* object : in_objectsPtrsVec)
        Labeling::m_objectsIndicesVector.at(object->m_index) = object;

    Labeling::m_labelsIndicesVector.assign(in_labelsPtrsVec.size(), nullptr);
    for (const Label* label : in_labelsPtrsVec)
        Labeling::m_labelsIndicesVector.at(label->m_index) = label;
}

/************************************************************************/
void Labeling::ClearStaticData()
/************************************************************************/
{
    m_objectsIndicesVector.clear();
    m_labelsIndicesVector.clear();
}

/************************************************************************/
Labeling Labeling::GetPermutationPartOfLabeling() const
//Function input: none
//Function output: Labeling that represents the permutation part of labeling   
//Function objective: described in "Function output"
/************************************************************************/
{
    Eigen::MatrixXd permutationPartOfMatrix = Eigen::MatrixXd::Constant(m_objectsIndicesVector.size(), m_labelsIndicesVector.size(), 0);

    const int32_t numOfColumnsToCheck = IsSquareLabeling()? 1 : m_numOfRotationsInType2Puzzles;

    for (const Object* object : m_objectsIndicesVector)
    {
        //1. If the max value is 1
        const std::pair<const Label*, double> maxLabelAndProb = GetMaxLabelAndProb(*object);
        if (1.0 == maxLabelAndProb.second)
        {
            //2. Get the sum of relevant columns
            const Label* label = maxLabelAndProb.first;

            double colSum = 0.0;
            const int32_t firstColToCheck = label->m_index - (label->m_index % numOfColumnsToCheck);
            for (int32_t j = firstColToCheck; j < (firstColToCheck + numOfColumnsToCheck); ++j)
            {
                const double currSum = m_labelingMatrix.col(j).maxCoeff();
                colSum += currSum; 
            }

            //3. If the sum of relevant columns is 1 (means that other values are 0)
            if (1.0 == colSum)
                permutationPartOfMatrix(object->m_index, label->m_index) = 1;
        }
    }

    return Labeling(permutationPartOfMatrix);
}

/************************************************************************/
bool Labeling::IsBinaryRow(const int32_t& in_rowIndex) const
/************************************************************************/
{
    for (int32_t j = 0; j < m_labelingMatrix.cols(); ++j)
    {
        const double currVal = m_labelingMatrix(in_rowIndex, j);
        if (currVal != 0 && currVal != 1)
            return false;
    }

    return true;
}

/************************************************************************/
bool Labeling::IsColBinaryWithSumOne(const int32_t& in_colIndex) const
/************************************************************************/
{
    if (m_labelingMatrix.col(in_colIndex).sum() != 1.0)
        return false;

    for (int32_t i = 0; i < m_labelingMatrix.rows(); ++i)
    {
        const double currVal = m_labelingMatrix(i, in_colIndex);
        if (currVal != 0 && currVal != 1)
            return false;
    }

    return true;
}

/************************************************************************/
bool Labeling::IsBinaryLabeling() const
//Function input: none
//Function output: boolean indicating whether the labeling matrix is binary (meaning, its entries are 0 or 1)  
//Function objective: described in "Function output"
/************************************************************************/
{
    for (int32_t i = 0; i < m_labelingMatrix.rows(); ++i)
    {
        const bool isBinaryRow = IsBinaryRow(i);
        if (!isBinaryRow)
            return false;
    }

    return true;
}

/************************************************************************/
bool Labeling::IsPermutationLabeling(const bool in_shouldLog) const
//Function input: in_shouldLog
//Function output: boolean indicating whether the labeling matrix is a permutation matrix  
//Function objective: described in "Function output"
/************************************************************************/
{
    bool retVal = false;
    if (IsBinaryLabeling())
    {
        if (IsSquareLabeling())
        {
            retVal = Utilities::IsDoublyStochasticMatrixWithPrecision(m_labelingMatrix, -1, false, in_shouldLog);
        }
        else
        {
            retVal = IsType2DoublyStochasticMatrixWithPrecision(in_shouldLog);
        }
    }

    return retVal; 
}

/************************************************************************/
bool Labeling::IsType2DoublyStochasticMatrixWithPrecision(const bool in_shouldLog) const
//Function input: in_shouldLog
//Function output: boolean indicating whether the labeling matrix is a doubly stochastic type 2 matrix 
//Function objective: described in "Function output"
/************************************************************************/
{
    const Eigen::MatrixXd squareMatrix = GetType1MatrixFromType2Matrix(m_labelingMatrix);
    return Utilities::IsDoublyStochasticMatrixWithPrecision(squareMatrix, 1, false, in_shouldLog);
}

/************************************************************************/
Eigen::MatrixXi Labeling::GetTopLabelsMatrix() const
/************************************************************************/
{
    Eigen::MatrixXi topLabelsMatrix = Eigen::MatrixXi::Constant(m_objectsIndicesVector.size(), m_labelsIndicesVector.size(), 0);

    for (const Object* object : m_objectsIndicesVector)
    {
        const std::pair<const Label*, double> maxLabelAndProb = GetMaxLabelAndProb(*object);
        const Label* label = maxLabelAndProb.first;

        topLabelsMatrix(object->m_index, label->m_index) = 1;
    }

    return topLabelsMatrix;
}

/************************************************************************/
Eigen::MatrixXi Labeling::GetDefiniteBinaryMatrix() const
//Function input: none
//Function output: binary matrix with 1's only in entries (i,j) which are max values in both their rows and columns
//Function objective: described in "Function output"
/************************************************************************/
{
    Eigen::MatrixXi binaryMatrix;
    if (IsSquareLabeling())
        binaryMatrix = Labeling::GetDefiniteBinaryMatrix(m_labelingMatrix, 1);
    else
        binaryMatrix = Labeling::GetDefiniteBinaryMatrix(m_labelingMatrix, m_numOfRotationsInType2Puzzles);

    return binaryMatrix;
}

/************************************************************************/
Eigen::MatrixXi Labeling::GetDefiniteBinaryMatrix(const Eigen::MatrixXd& in_matrix, const int32_t in_numOfPossibleRotations)
//Function input: in_matrix: matrix; in_numOfPossibleRotations: num of possible rotations (says if it's type 1 ot type 2 puzzle)
//Function output: binary matrix with 1's only in entries (i,j) which are max values in both their rows and column/s in 'in_matrix'
//Function objective: described in "Function output"
/************************************************************************/
{
    Utilities::LogAndAbortIf(in_matrix.rows() * in_numOfPossibleRotations != in_matrix.cols(),
        "In 'Labeling::GetDefiniteBinaryMatrix()': 'in_matrix' dimension issue");

    Eigen::MatrixXi binaryMatrix = Eigen::MatrixXi::Zero(in_matrix.rows(), in_matrix.cols());

    for (int32_t rowIndex = 0; rowIndex < in_matrix.rows(); ++rowIndex)
    {
        //1. Get max in row
        int32_t maxColIndex;
        const double maxValInRow = in_matrix.row(rowIndex).maxCoeff(&maxColIndex);

        //2. Get max in column (for type 1) or columns (for type 2)
        double maxValInLocationColumns = -1;
        const int32_t firstColToCheck = maxColIndex - (maxColIndex % in_numOfPossibleRotations);
        for (int32_t j = firstColToCheck; j < (firstColToCheck + in_numOfPossibleRotations); ++j)
        {
            const double maxValInCurrCol = in_matrix.col(j).maxCoeff();
            maxValInLocationColumns = std::max(maxValInLocationColumns, maxValInCurrCol); 
        }

        //3. If the max value in row is also the max value in columns
        if (maxValInRow == maxValInLocationColumns)
        {
            //4. Get num of incidents of the max value in row
            int32_t numOfEntriesWithValInRow = 0;
            for (int32_t j = 0; j < in_matrix.cols(); ++j)
            {
                const double currVal = in_matrix(rowIndex, j); 
                if (currVal == maxValInRow)
                    ++numOfEntriesWithValInRow;
            }
            Utilities::LogAndAbortIf(numOfEntriesWithValInRow == 0, "numOfEntriesWithValInRow == 0");

            //5. Get num of incidents of the max value in column/s
            int32_t numOfEntriesWithValInCol = 0;
            for (int32_t j = firstColToCheck; j < (firstColToCheck + in_numOfPossibleRotations); ++j)
            {
                for (int32_t i = 0; i < in_matrix.rows(); ++i)
                {
                    const double currVal = in_matrix(i, j); 
                    if (currVal == maxValInRow)
                        ++numOfEntriesWithValInCol;
                }
            }
            Utilities::LogAndAbortIf(numOfEntriesWithValInCol == 0, "numOfEntriesWithValInCol == 0");

            //6. If the entry in (object->m_index, maxColIndex) is max in its row and columns - assign 1
            if (numOfEntriesWithValInRow == 1 && numOfEntriesWithValInCol == 1)
                binaryMatrix(rowIndex, maxColIndex) = 1;
        }
    }

    return binaryMatrix;
}

/************************************************************************/
Eigen::MatrixXd Labeling::GetType1MatrixFromType2Matrix(const Eigen::MatrixXd& in_matrix)
//Function input: in_matrix: matrix
//Function output: square matrix that represent the type 2 'in_matrix' 
//Function objective: described in "Function output"
/************************************************************************/
{
    Utilities::LogAndAbortIf(in_matrix.rows() * m_numOfRotationsInType2Puzzles != in_matrix.cols(),
        "In 'Labeling::GetType1MatrixFromType2Matrix()': 'in_matrix' dimension issue");

    Eigen::MatrixXd squareMatrix = Eigen::MatrixXd::Zero(in_matrix.rows(), in_matrix.rows());

    for (int32_t objIndex = 0; objIndex < in_matrix.rows(); ++objIndex)
    {
        for (int32_t labelIndex = 0; labelIndex < in_matrix.cols(); labelIndex += m_numOfRotationsInType2Puzzles)
        {
            //Sum all rotation values
            double sum = 0.0;
            for (int32_t rot = 0; rot < m_numOfRotationsInType2Puzzles; ++rot)
            {
                sum += in_matrix(objIndex, labelIndex + rot);
            }

            //Compute label index in square matrix
            const int32_t labelIndexInSquareMat = labelIndex / m_numOfRotationsInType2Puzzles;
            squareMatrix(objIndex, labelIndexInSquareMat) = sum;
        }
    }

    return squareMatrix;
}
