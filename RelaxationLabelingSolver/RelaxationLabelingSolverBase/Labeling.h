#pragma once

#include <Eigen/Core>
#include "Label.h"
#include "Object.h"

using LabelAndProbPair = std::pair<const Label*, double>;

class Labeling
{
public:
    Labeling(const size_t in_numOfObjects, const size_t in_numOfLabels, const double in_initValue = 0);
    Labeling(const Eigen::MatrixXd& in_labelingMatrix);

    void SetLabelingValue(const Object& in_object, const Label& in_label, const double& in_val)
    {
        m_labelingMatrix(in_object.m_index, in_label.m_index) = in_val;
    }
    
    double GetLabelingValue(const Object& in_object, const Label& in_label) const
    {
        return m_labelingMatrix(in_object.m_index, in_label.m_index);
    }

    void SetLabelingValue(const int32_t& in_objectIdx, const int32_t& in_labelIdx, const double& in_val)
    {
        m_labelingMatrix(in_objectIdx, in_labelIdx) = in_val;
    }
    
    double GetLabelingValue(const int32_t& in_objectIdx, const int32_t& in_labelIdx) const
    {
        return m_labelingMatrix(in_objectIdx, in_labelIdx);
    }

    LabelAndProbPair GetMaxLabelAndProb(const Object& in_object) const;
    Eigen::MatrixXd GetLabelingMatrix() const {return m_labelingMatrix;}

    Labeling GetBinaryLabeling() const {return Labeling(GetDefiniteBinaryMatrix().cast<double>());}
    Labeling GetPermutationPartOfLabeling() const;

    bool IsBinaryRow(const int32_t& in_rowIndex) const;
    bool IsColBinaryWithSumOne(const int32_t& in_colIndex) const;

    bool IsBinaryLabeling() const;
    bool IsPermutationLabeling(const bool in_shouldLog = true) const;

    bool IsType2DoublyStochasticMatrixWithPrecision(const bool in_shouldLog = true) const;
    
    static double MultiplyAndSumLabelings(const Labeling& in_firstLabeling, const Labeling& in_secondLabeling);
    static void SetStaticData(const ObjectsVector& in_objectsPtrsVec, const LabelsVector& in_labelsPtrsVec);
    static void ClearStaticData();
    bool IsSquareLabeling() const { return m_labelingMatrix.rows() == m_labelingMatrix.cols(); }

protected:
    Eigen::MatrixXi GetTopLabelsMatrix() const;
    Eigen::MatrixXi GetDefiniteBinaryMatrix() const;

    static Eigen::MatrixXi GetDefiniteBinaryMatrix(const Eigen::MatrixXd& in_matrix, const int32_t in_numOfPossibleRotations);
    static Eigen::MatrixXd GetType1MatrixFromType2Matrix(const Eigen::MatrixXd& in_matrix);

    static std::vector<const Object*> m_objectsIndicesVector;
    static std::vector<const Label*> m_labelsIndicesVector;
    //'m_numOfRotationsInType2Puzzles' doesn't belong to 'Labeling', but we need to have it here
    static constexpr int32_t m_numOfRotationsInType2Puzzles = 4;

    Eigen::MatrixXd m_labelingMatrix;
};

using SupportType = Labeling;
using LabelingEntryInfo = std::pair<RowColInfo, double>;
