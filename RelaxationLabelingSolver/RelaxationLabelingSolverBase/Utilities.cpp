#include "Utilities.h"
#include "RelaxationLabelingSolverBaseConstants.h"
#include <Eigen/LU>
#include <Eigen/QR>

Utilities::Logger g_logger;

using namespace Utilities;
const RowColInfo RowColInfo::m_invalidRowColInfo(RowColInfo::invalidIndex, RowColInfo::invalidIndex);
int32_t TechnicalParameters::m_iterationsFrequencyOfImagePrintsDuringAlg = 1;

/************************************************************************/
bool RowColInfo::operator<(const RowColInfo& in_other) const
/************************************************************************/
{
    if (m_row == in_other.m_row)
        return m_col < in_other.m_col;
    else
        return m_row < in_other.m_row;
}

/************************************************************************/
bool Utilities::RowColInfo::AreNeighbors(const RowColInfo& in_lhs, const RowColInfo& in_rhs)
/************************************************************************/
{
    return (in_lhs.GetTopNeighborCoord(in_lhs.m_row, in_lhs.m_col) == in_rhs || 
        in_lhs.GetBottomNeighborCoord(in_lhs.m_row, in_lhs.m_col) == in_rhs ||
        in_lhs.GetLeftNeighborCoord(in_lhs.m_row, in_lhs.m_col) == in_rhs ||
        in_lhs.GetRightNeighborCoord(in_lhs.m_row, in_lhs.m_col) == in_rhs);
}

/************************************************************************/
Utilities::RowColInfo Utilities::RowColInfo::NeighborCoordByOrien(const int32_t in_row, const int32_t in_col, const Orientation in_orien)
/************************************************************************/
{
    RowColInfo retVal = m_invalidRowColInfo;
    switch (in_orien)
    {
    case eDown:
        retVal = GetBottomNeighborCoord(in_row, in_col);
        break;

    case eUp:
        retVal = GetTopNeighborCoord(in_row, in_col);
        break;

    case eRight:
        retVal = GetRightNeighborCoord(in_row, in_col);
        break;

    case eLeft:
        retVal = GetLeftNeighborCoord(in_row, in_col);
        break;

    default:
        Utilities::LogAndAbort("");
    }

    return retVal;
}

/************************************************************************/
void Utilities::LogAndAbort(const std::string& in_str)
/************************************************************************/
{
    g_logger.BruteLog(in_str + " - crashing...");
    std::abort();
}

/************************************************************************/
void Utilities::LogAndAbortIf(const bool in_condition, const std::string& in_str)
/************************************************************************/
{
    if (in_condition)
        LogAndAbort(in_str);
}

/************************************************************************/
void Utilities::Log(const std::string& in_str)
/************************************************************************/
{
    g_logger << in_str << std::endl;
}

/************************************************************************/
double Utilities::RoundWithPrecision(const double in_val, const int32_t in_numOfDigits)
//Function input: in_val: value to round; in_numOfDigits: num of digits to have after rounding
//Function output: rounded 'in_val'
//Function objective: described in "Function output"
/************************************************************************/
{
    //If 'in_numOfDigits' is -1 - we don't apply rounding
    if (-1 == in_numOfDigits)
    {
        return in_val;
    }
    else
    {
        const double tenPowerNumOfDigits = pow(10.0, in_numOfDigits);
        const double roundedVal = round(in_val * tenPowerNumOfDigits) / tenPowerNumOfDigits;
        return roundedVal;
    }
}

/************************************************************************/
bool Utilities::IsDoublyStochasticMatrixWithPrecision(const Eigen::MatrixXd& in_matrix, const int32_t in_numOfDigits, 
    const bool in_shouldAbort, const bool in_shouldLog)
//Function input: in_matrix: matrix; in_numOfDigits: row and columns sum precision;
    //in_shouldAbort: boolean indicating whether we should abort in case the matrix is not doubly stochastic; 
    //in_shouldLog: boolean indicating whether we should log information in case the matrix is not doubly stochastic; 
//Function output: boolean indicating whether if 'in_matrix' is doubly stochastic matrix  
//Function objective: described in "Function output"
/************************************************************************/
{
    //1. Check that in_matrix is square
    Utilities::LogAndAbortIf(in_matrix.rows() != in_matrix.cols(), "Matrix is not square in 'Utilities::IsStochasticMatrixWithPrecision()'");
    const size_t matrixDimension = in_matrix.rows();

    bool retVal = true;

    //2. Check each row and column sums
    for (size_t i = 0; i < matrixDimension; ++i)
    {
        double rowSum = in_matrix.row(i).sum();
        rowSum = Utilities::RoundWithPrecision(rowSum, in_numOfDigits);
        if (1.0 != rowSum)
        {
            if (in_shouldLog)
                Utilities::Log("row sum is not 1: row index is: " + std::to_string(i) + ", actual sum is: " + std::to_string(rowSum));

            retVal = false;
        }

        double colSum = in_matrix.col(i).sum();
        colSum = Utilities::RoundWithPrecision(colSum, in_numOfDigits);
        if (1.0 != colSum)
        {
            if (in_shouldLog)
                Utilities::Log("column sum is not 1: column index is: " + std::to_string(i) + ", actual sum is: " + std::to_string(colSum));

            retVal = false;
        }
    }

    if (in_shouldAbort && retVal == false)
        Utilities::LogAndAbort("crashing in 'Utilities::IsStochasticMatrixWithPrecision()'");

    return retVal;
}

/************************************************************************/
Eigen::MatrixXd Utilities::ComputeInverseOfCovarianceMatrix(const Eigen::MatrixXd& in_samplesMatrix)
//Function input: in_samplesMatrix: matrix representing the samples (each row is a sample)
//Function output: inverse covariance matrix of the data given as input to this function
//Function objective: described in "Function output"
/************************************************************************/
{                                                                                                                                                               
    //1. Compute sample covariance matrix
    const Eigen::Vector3d meanVector = in_samplesMatrix.colwise().mean();
    const Eigen::MatrixXd centeredSamples = (in_samplesMatrix.rowwise() - meanVector.transpose());
    const Eigen::MatrixXd covarianceMatrix = (centeredSamples.transpose() * centeredSamples) / static_cast<double>(in_samplesMatrix.rows() - 1);

    //2. Check that the covariance matrix is positive semi-definite (involves two checks: symmetry and existence of Choleski decomposition)
    //(if the covariance matrix is positive semi-definite, we know for sure that the covariance matrix inverse is positive semi-definite)
    Utilities::LogAndAbortIf(!covarianceMatrix.isApprox(covarianceMatrix.transpose()), "Covariance matrix is not symmetric");
    Utilities::LogAndAbortIf(Eigen::NumericalIssue == Eigen::LLT<Eigen::MatrixXd>(covarianceMatrix).info(), "Covariance matrix doesn't have valid Choleski decomposition");

    //3. Invert the covariance matrix
    const bool isCovarianceMatrixInvertible = Eigen::FullPivLU<Eigen::MatrixXd>(covarianceMatrix).isInvertible();
    Utilities::LogAndAbortIf(!isCovarianceMatrixInvertible, "Covariance matrix is not invertible");
    const Eigen::MatrixXd covarianceMatrixInverse = covarianceMatrix.inverse();

    return covarianceMatrixInverse;
}