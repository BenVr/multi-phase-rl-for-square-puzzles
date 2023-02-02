#pragma once

#include "RelaxationLabelingSolverBaseConstants.h"
#include "RelaxationLabelingSolverBaseUtils.h"
#include "Labeling.h"
#include "Label.h"
#include "Object.h"
#include "Utilities.h"
#include <map>
#include <vector>

#include "FileSystemUtils.h"
#include "XML_Configuration.h"

class RelaxationLabelingSolverBase
{
public:
    RelaxationLabelingSolverBase(const XML_Configuration& in_xmlConfig, const int32_t in_runNumber, const int32_t in_maxRLIterations);

    virtual ~RelaxationLabelingSolverBase();

    //Public functions
public:
    virtual void Init() = 0;
    void SolveProblem(RL_algorithmRunningInfo* inout_algRunningInfo);

    //Protected pure virtual functions
protected:
    virtual Labeling GetInitLabeling() const = 0;
    virtual bool HasSolutionChangedInIteration(const Labeling& in_newLabeling) const = 0;
    virtual double GetCompatibilityValue(const Object* in_object1, const Label* in_label1,
        const Object* in_object2, const Label* in_label2) const = 0;
    virtual void OutputLabelingDuringAlg(const Labeling& in_labeling, const int32_t in_iterationNum) const = 0;
    virtual void OutputSolutionLabeling(const Labeling& in_labeling) const = 0;
    virtual void PostMakeLabelingConsistent(const Labeling& in_labeling) = 0;
    virtual bool ShouldDoRandomAnchoringAndThenBreak(const std::set<int32_t>& in_anchoredRowsSet, std::string& out_breakAfterAnchoringStr) const = 0;
    virtual RowColInfoSet GetSetOfEntriesToBeAnchored(const Labeling& in_labeling, const AnchoringData& in_anchoringData) const = 0;
    virtual Labeling DoAnchoring(const Labeling& in_labeling, const RowColInfoSet& in_allEntriesToBeAnchoredSet) const = 0;
    virtual bool DoTranslationAndPossiblyBranchAlg(Labeling& inout_labeling, const RowColInfoSet& in_newJustAnchoredEntries, 
        RL_algorithmRunningInfo& inout_algRunningInfo, bool& out_shouldBreakAfterTranslation) const = 0;
    virtual bool AreTranslationsDone() const = 0;
    virtual void DoPostConvergenceAnchoring(Labeling& inout_labeling, AnchoringData& inout_anchoringData) const = 0;
    virtual void AnchorNonAnchoredObjects(Labeling& inout_labeling, AnchoringData& inout_anchoringData) const = 0;
    virtual RowColInfo GetMaxEntryInNonAnchoredRow(const Labeling& in_labeling, const AnchoringData& in_anchoringData) const = 0;
    virtual bool VerifyAnchoringLegality(const RowColInfo& in_newEntryToBeAnchored, const AnchoringData& in_anchoringData) const = 0;
                                                           
    //Protected member functions and data members
protected:
    void DoPostConvergenceActions(const Labeling& in_labeling) const;

    void SetAllPossibleObjects(const ObjectsVector& in_allPossibleObjects) { m_allPossibleObjects = in_allPossibleObjects; }
    void SetAllPossibleLabels(const LabelsVector& in_allPossibleLabels) { m_allPossibleLabels = in_allPossibleLabels; }
    void InitSupportComputationMethodData();
    void MakeLabelingConsistent(Labeling& inout_labeling, RL_algorithmRunningInfo& inout_algRunningInfo) const;

    void DoOneUpdateStep(const Labeling& in_oldLabeling, const SupportType& in_support, Labeling& out_newLabeling) const;
    void DoOneUpdateStepEfficient(const Labeling& in_oldLabeling, const SupportType& in_support, Labeling& out_newLabeling, 
        const AnchoringData& in_anchoringData) const;
    void DoObjectUpdateStep(const Object* in_object, const Labeling& in_oldLabeling, const SupportType& in_support, Labeling& out_newLabeling) const;

    double ComputeAverageLocalConsistency(const Labeling& in_labeling, const SupportType& in_support) const;
    void GetSupport(const Labeling& in_labeling, SupportType& out_support) const;
    double GetObjectAndLabelSupport(const Object* in_object, const Label* in_label, const Labeling& in_labeling) const;
    double ComputePairwiseSupport(const Object* in_object, const Label* in_label, const Labeling& in_labeling) const;

    RowColInfoSet TryToAnchorObjects(Labeling& inout_labeling, AnchoringData& inout_anchoringData) const;
    void UpdateAndDoAnchoring(const RowColInfoSet& in_newEntriesToBeAnchoredSet,
        Labeling& inout_labeling, AnchoringData& inout_anchoringData) const;

    RowColInfoSet AnchorMaxEntryInLabelingMatrix(Labeling& inout_labeling, AnchoringData& inout_anchoringData) const;

    std::string GetPathInRunFolder(const std::string& in_pathInRunFolderStr) const {return FileSystemUtils::GetPathInFolder(m_currOutputFolder, in_pathInRunFolderStr);};
    void SetCurrentOutputFolder(const std::string& in_newOutputFolder) const {m_currOutputFolder = in_newOutputFolder;}

    mutable std::string m_currOutputFolder;

    LabelsVector m_allPossibleLabels;
    ObjectsVector m_allPossibleObjects;

#if IS_SAVE_ALL_PAIRS_SUPPORT_COMPUTATION_METHOD()
    void InitCorrelatedPairsMatrix(); 
    void DeleteCorrelatedPairsMatrix();
    double ComputePairwiseSupportWithCorrelatedPairsMatrix(const Object* in_object, const Label* in_label, const Labeling& in_labeling) const;

    using ObjectLabelPair = std::pair<const Object*, const Label*>;
    using ObjectLabelPairsVec = std::vector<ObjectLabelPair>;
    using ObjectLabelPairsVecMatrix = Eigen::Matrix<ObjectLabelPairsVec*, Eigen::Dynamic, Eigen::Dynamic>;

    virtual LabelsVector GetCorrelatedLabelsVec(const Label* in_label) const = 0;

    //m_correlatedPairsMatrix is a matrix in which we have an entry for 
    //each (object, label) pair, and each such entry contains pointer to vector of (object, label) pairs. 
    //This matrix is used during support computation 
    //The entry for a (object, label) pair is used in the following way:
    //m_correlatedPairsMatrix(object_i, label_lambda) - represents all the non-zero pairs that should 
    //take part in support computation of Pi(lambda)
    ObjectLabelPairsVecMatrix m_correlatedPairsMatrix;

#elif IS_STRAIGHTFORWARD_SUPPORT_COMPUTATION_METHOD()
    double ComputePairwiseSupportInStraightforwardWay(const Object* in_object, const Label* in_label, const Labeling& in_labeling) const;
#endif

    const int32_t m_runNumber;
    const int32_t m_maxRLIterations;

    Labeling* m_pSolution = nullptr;

    static constexpr double m_anchoringThreshold = 0.7;    
};






