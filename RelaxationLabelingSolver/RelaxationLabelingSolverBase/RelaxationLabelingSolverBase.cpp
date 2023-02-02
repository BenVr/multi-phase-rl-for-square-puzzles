#include <iostream>
#include "RelaxationLabelingSolverBase.h"
#include "Utilities.h"
#include <thread>
#include <random>

extern Utilities::Logger g_logger;

/************************************************************************/
RelaxationLabelingSolverBase::RelaxationLabelingSolverBase(const XML_Configuration& /*in_xmlConfig*/, const int32_t in_runNumber, const int32_t in_maxRLIterations) :
    m_runNumber(in_runNumber),
    m_maxRLIterations(in_maxRLIterations)
/************************************************************************/
{
}

/************************************************************************/
RelaxationLabelingSolverBase::~RelaxationLabelingSolverBase()
/************************************************************************/
{
    Labeling::ClearStaticData();
    Object::ZeroIndexCounter();
    Label::ZeroIndexCounter();

    delete m_pSolution;

#if IS_SAVE_ALL_PAIRS_SUPPORT_COMPUTATION_METHOD()
    DeleteCorrelatedPairsMatrix();
#endif
}

/************************************************************************/
void RelaxationLabelingSolverBase::Init()
/************************************************************************/
{
    Labeling::SetStaticData(m_allPossibleObjects, m_allPossibleLabels);

    m_pSolution = new Labeling(m_allPossibleObjects.size(), m_allPossibleLabels.size());
}

/************************************************************************/
void RelaxationLabelingSolverBase::InitSupportComputationMethodData()
/************************************************************************/
{
#if IS_SAVE_ALL_PAIRS_SUPPORT_COMPUTATION_METHOD()
    InitCorrelatedPairsMatrix();
#endif
}

#if IS_SAVE_ALL_PAIRS_SUPPORT_COMPUTATION_METHOD()

/************************************************************************/
void RelaxationLabelingSolverBase::InitCorrelatedPairsMatrix()
/************************************************************************/
{
    using LogInitInfoType = std::false_type;

    //1. Create m_correlatedPairsMatrix
    m_correlatedPairsMatrix = ObjectLabelPairsVecMatrix::Constant(m_allPossibleObjects.size(), m_allPossibleLabels.size(), nullptr);

    if constexpr (LogInitInfoType::value)
        g_logger << "'RelaxationLabelingSolverBase::InitCorrelatedPairsMatrix()': allocating memory..." << std::endl;

    //2. Allocate pointers in m_correlatedPairsMatrix
    std::for_each(executionPolicy, m_allPossibleObjects.begin(), m_allPossibleObjects.end(),
        [&](const Object* in_object) -> void
        {
            if constexpr (LogInitInfoType::value)
                g_logger << "'Allocating memory for object '" << in_object->GetString() << "'" << std::endl;

            for (const Label* label : m_allPossibleLabels)
            {
                m_correlatedPairsMatrix(in_object->m_index, label->m_index) = new ObjectLabelPairsVec();
            }
        });

    if constexpr (LogInitInfoType::value)
        g_logger << "'RelaxationLabelingSolverBase::InitCorrelatedPairsMatrix()': Setting values in memory..." << std::endl;

    //3. Set elements in m_correlatedPairsMatrix
    std::for_each(executionPolicy, m_allPossibleObjects.begin(), m_allPossibleObjects.end(),
        [&](const Object* in_pFirstObj) -> void
        {
            if constexpr (LogInitInfoType::value)
                g_logger << "'Setting values for object '" << in_pFirstObj->GetString() << "'" << std::endl;

            for (const Label* firstLabel : m_allPossibleLabels)
            {
                for (const Object* secondObject : m_allPossibleObjects)
                {
                    const LabelsVector& labelsCorrelatedWithFirstLabel = GetCorrelatedLabelsVec(firstLabel);

                    for (const Label* secondLabel: labelsCorrelatedWithFirstLabel)
                    {
                        const double val = GetCompatibilityValue(in_pFirstObj, firstLabel, secondObject, secondLabel);

                        //If compatibility value is zero - we don't save it (since it does not take part in support computation)
                        if (0.0 != val)
                            m_correlatedPairsMatrix(in_pFirstObj->m_index, firstLabel->m_index)->emplace_back(secondObject, secondLabel);
                    }
                }
            }
        }
    );

    if constexpr (LogInitInfoType::value)
        g_logger << "'RelaxationLabelingSolverBase::InitCorrelatedPairsMatrix()': Done!" << std::endl;
}

/************************************************************************/
void RelaxationLabelingSolverBase::DeleteCorrelatedPairsMatrix()
/************************************************************************/
{
    for (int32_t row = 0; row < m_correlatedPairsMatrix.rows(); ++row)
    {
        for (int32_t col = 0; col < m_correlatedPairsMatrix.cols(); ++col)
        {
            delete m_correlatedPairsMatrix(row, col);
        }
    }
}

#endif

/************************************************************************/
void RelaxationLabelingSolverBase::SolveProblem(RL_algorithmRunningInfo* inout_algRunningInfo)
/************************************************************************/
{
    //1. Get initial labeling, and make it consistent
    Labeling labeling = GetInitLabeling();
    MakeLabelingConsistent(labeling, *inout_algRunningInfo);

    //2. do post convergence tasks
    DoPostConvergenceActions(labeling);

    *m_pSolution = labeling.GetBinaryLabeling();
    PostMakeLabelingConsistent(labeling);
}

/************************************************************************/
void RelaxationLabelingSolverBase::DoPostConvergenceActions(const Labeling& in_labeling) const
/************************************************************************/
{
    //1. Verify that labeling is permutation
    Utilities::LogAndAbortIf(!in_labeling.IsPermutationLabeling(), 
        "In 'RelaxationLabelingSolverBase::DoPostConvergenceActions()': !binaryLabeling.IsPermutationLabeling()");

    //2. Output final labelings, and do post actions
    OutputSolutionLabeling(in_labeling);
}

/************************************************************************/
//This function is the heart of relaxation labeling algorithm
void RelaxationLabelingSolverBase::MakeLabelingConsistent(Labeling& inout_labeling, RL_algorithmRunningInfo& inout_algRunningInfo) const
/************************************************************************/
{
    //1. Init two labeling variables and one support variables
    Labeling oldLabeling = Labeling(inout_labeling);
    SupportType currentSupport = SupportType(m_allPossibleObjects.size(), m_allPossibleLabels.size());
    Labeling newLabeling = Labeling(m_allPossibleObjects.size(), m_allPossibleLabels.size());

    //2. Compute support, current labeling is oldLabeling
    GetSupport(oldLabeling, currentSupport);
    double oldAverageLocalConsistency = ComputeAverageLocalConsistency(oldLabeling, currentSupport);
    double newAverageLocalConsistency = 0.0;

    //3. Init variables and update 'inout_algRunningInfo' data
    bool isLabelingConsistent = false;
    int32_t lastIterationImageWasPrinted = 0;
    bool shouldBreakAfterTranslation = false;

    inout_algRunningInfo.m_solutionChangedValues.emplace_back(inout_algRunningInfo.m_iterationNum, oldAverageLocalConsistency);
    ++inout_algRunningInfo.m_iterationNum;

    bool anchorMaxEntryInLabelingMatrix = false;

    //3. While labeling is not consistent
    while (true)
    {
        g_logger << "~~~~~~~~~~" << std::endl;

        if (isLabelingConsistent)
            anchorMaxEntryInLabelingMatrix = true;

        //3.1. Init time
        Utilities::Timer timer;

        //3.2. Apply update step to get newLabeling
        //We first calculate the labeling in 'DoOneUpdateStep()' and then the support in 'GetSupport()'
        //since this way we get in each iteration the kth labeling and kth support, and we can calculate
        //the average local consistency out of it 
        //(average local consistency uses the kth labeling and kth support, not kth labeling and (k-1)th support)
        //DoOneUpdateStep(oldLabeling, currentSupport, newLabeling);
        DoOneUpdateStepEfficient(oldLabeling, currentSupport, newLabeling, inout_algRunningInfo.m_anchoringData);

        //3.3. Do anchoring (if needed)
        RowColInfoSet newEntriesToBeAnchored;
        if (anchorMaxEntryInLabelingMatrix)
        {
            newEntriesToBeAnchored = AnchorMaxEntryInLabelingMatrix(newLabeling, inout_algRunningInfo.m_anchoringData);
            anchorMaxEntryInLabelingMatrix = false;  
        }
        else
        {
            newEntriesToBeAnchored = TryToAnchorObjects(newLabeling, inout_algRunningInfo.m_anchoringData);
        }

        bool didAnchor = !newEntriesToBeAnchored.empty();
        if (didAnchor)
        {
            if (inout_algRunningInfo.m_anchoringData.m_anchoredRowsSet.size() == m_allPossibleObjects.size())
            {
                Utilities::Log("Breaking since all pieces are anchored!");
                break;
            }

            if (AreTranslationsDone())
            {
                std::string breakAfterAnchoringStr;
                if (ShouldDoRandomAnchoringAndThenBreak(inout_algRunningInfo.m_anchoringData.m_anchoredRowsSet, breakAfterAnchoringStr))
                {
                    AnchorNonAnchoredObjects(newLabeling, inout_algRunningInfo.m_anchoringData);
                    Utilities::Log(breakAfterAnchoringStr);
                    break;
                }
            }

            DoTranslationAndPossiblyBranchAlg(newLabeling, newEntriesToBeAnchored, inout_algRunningInfo, shouldBreakAfterTranslation);

            if (shouldBreakAfterTranslation)
            {
                Utilities::Log("Breaking after translation");
                break;
            }
        }

        //3.4. Compute new support
        GetSupport(newLabeling, currentSupport);

        //3.5. Compute alc, and determine if algorithm has converged (if it did, isLabelingConsistent is true)
        //(note: we cannot converge due to consistency right after anchoring piece)
        newAverageLocalConsistency = ComputeAverageLocalConsistency(newLabeling, currentSupport);

        if (didAnchor)
        {
            isLabelingConsistent = false; 
        }
        else
        {
            //By relaxation labeling theory, this if condition should be never be entered
            if (newAverageLocalConsistency < oldAverageLocalConsistency)
            {
                Utilities::Log("Old average local consistency is: " + std::to_string(oldAverageLocalConsistency));
                Utilities::Log("New average local consistency is: " + std::to_string(newAverageLocalConsistency));
                Utilities::LogAndAbort("newAverageLocalConsistency is lower than oldAverageLocalConsistency");
            }

            isLabelingConsistent = (newAverageLocalConsistency - oldAverageLocalConsistency) <= RelaxationLabelingSolverBaseConstants::epsilon;
        }

        //3.6. Compute diffConsistency (used just for logging)
        const double diffConsistency = newAverageLocalConsistency - oldAverageLocalConsistency;
       
        //3.7. Swap values
        oldAverageLocalConsistency = newAverageLocalConsistency;
        oldLabeling = newLabeling;
        
        //3.8. Finally, do some technical tasks, include logging 
        const bool hasSolutionChangedInIteration = HasSolutionChangedInIteration(newLabeling);

        const long long milliSecondsPerIteration = timer.GetMilliSecondsPassed();

        g_logger << "Current iteration is: " << inout_algRunningInfo.m_iterationNum ;
        g_logger << ((hasSolutionChangedInIteration) ? ", solution has changed" : "") << std::endl;
        g_logger << "Iteration took " << milliSecondsPerIteration << " milliseconds" << std::endl;
        g_logger << "Average local consistency is: " << newAverageLocalConsistency << std::endl;
        g_logger << "Difference of average local consistency values is: " << diffConsistency << std::endl;
        inout_algRunningInfo.sumOfIterationsTimeInMilliseconds += milliSecondsPerIteration;

        if (hasSolutionChangedInIteration)
        {
            inout_algRunningInfo.m_solutionChangedValues.emplace_back(inout_algRunningInfo.m_iterationNum, newAverageLocalConsistency);
            if (inout_algRunningInfo.m_iterationNum - lastIterationImageWasPrinted >= TechnicalParameters::GetIterationsFrequencyOfImagePrintsDuringAlg())
            {
                OutputLabelingDuringAlg(newLabeling, inout_algRunningInfo.m_iterationNum);
                lastIterationImageWasPrinted = inout_algRunningInfo.m_iterationNum;
            }
        }
        else if constexpr (!OUTPUT_ONLY_NEW_CURRENT_SOLUTIONS)
        {
            OutputLabelingDuringAlg(newLabeling, inout_algRunningInfo.m_iterationNum);
        }

        if (inout_algRunningInfo.m_iterationNum >= m_maxRLIterations)
        {
            Utilities::Log("Reached " + std::to_string(inout_algRunningInfo.m_iterationNum) + " iterations!!");
            break;
        }

        ++inout_algRunningInfo.m_iterationNum;
    }

    //4. Do post convergence anchoring if should to
    DoPostConvergenceAnchoring(newLabeling, inout_algRunningInfo.m_anchoringData);

    //5. Save m_averageTimePerIteration, m_finalAlc and set inout_labeling to newLabeling
    inout_algRunningInfo.m_averageTimePerIteration = 
        static_cast<double>(inout_algRunningInfo.sumOfIterationsTimeInMilliseconds) / inout_algRunningInfo.m_iterationNum;

    GetSupport(newLabeling, currentSupport);
    inout_algRunningInfo.m_finalAlc = ComputeAverageLocalConsistency(newLabeling, currentSupport);

    inout_labeling = newLabeling;

    //6. Log some info
    g_logger << ((inout_algRunningInfo.m_anchoringData.m_anchoredRowsSet.size() != m_allPossibleObjects.size())? "\nNot all objects are anchored" : "");

    g_logger << std::endl << "Solution image has changed in the following iterations: ";
    for (const std::pair<int, double>& element : inout_algRunningInfo.m_solutionChangedValues)
        g_logger << element.first << ",";
    g_logger << std::endl;
}

/************************************************************************/
RowColInfoSet RelaxationLabelingSolverBase::TryToAnchorObjects(Labeling& inout_labeling, AnchoringData& inout_anchoringData) const
//Function input: inout_labeling: labeling; inout_anchoringData: anchoring data
//Function output: boolean indicating whether new objects were anchored  
//Function objective: anchor objects in 'inout_labeling' (if should to), and update 'inout_anchoringData' accordingly
/************************************************************************/
{
    //1. Get set of rows indices to be anchored (rows represent the objects)
    const RowColInfoSet newEntriesToBeAnchoredSet = GetSetOfEntriesToBeAnchored(inout_labeling, inout_anchoringData);

    //2. If there are new objects to be anchored
    if (!newEntriesToBeAnchoredSet.empty())
    {
        UpdateAndDoAnchoring(newEntriesToBeAnchoredSet, inout_labeling, inout_anchoringData);
    }

    return newEntriesToBeAnchoredSet;
}

/************************************************************************/
void RelaxationLabelingSolverBase::UpdateAndDoAnchoring(const RowColInfoSet& in_newEntriesToBeAnchoredSet,
    Labeling& inout_labeling, AnchoringData& inout_anchoringData) const
//Function input: in_newEntriesToBeAnchoredSet: entries to be anchored, inout_labeling: labeling;
    //inout_anchoredRowsSet: set of anchored rows; inout_anchoredEntriesSet: set of anchored entries
//Function output: none  
//Function objective: anchor entries in 'in_newEntriesToBeAnchoredSet', and update 'inout_anchoredRowsSet' and 'inout_anchoredEntriesSet' accordingly
/************************************************************************/
{
    //1. Update 'inout_anchoringData'
    for (const RowColInfo& currEntry: in_newEntriesToBeAnchoredSet)
    {
        inout_anchoringData.m_anchoredRowsSet.insert(currEntry.m_row);
        inout_anchoringData.m_anchoredEntriesSet.insert(currEntry);
        inout_anchoringData.m_nonAnchoredRowsSet.erase(currEntry.m_row);
    }

    //2. Anchor objects (we should send 'inout_anchoredRowsSet' here, and not 'newRowsToBeAnchoredSet')
    inout_labeling = DoAnchoring(inout_labeling, inout_anchoringData.m_anchoredEntriesSet); 

    //3. Log new entries to be anchored
    g_logger << "Found new objects to be anchored: ";
    for (const RowColInfo& currEntry: in_newEntriesToBeAnchoredSet)
    {
        g_logger << "object '" << m_allPossibleObjects[currEntry.m_row]->GetString()
            << "' and label '" << m_allPossibleLabels[currEntry.m_col]->GetString() << "', ";
    }
    g_logger << std::endl;
}

/************************************************************************/
RowColInfoSet RelaxationLabelingSolverBase::AnchorMaxEntryInLabelingMatrix(Labeling& inout_labeling, AnchoringData& inout_anchoringData) const
//Function input: inout_labeling: labeling; inout_anchoredRowsSet: set of anchored rows; inout_anchoredEntriesSet: set of anchored entries
//Function output: set of new anchored entries  
//Function objective: anchor max non-anchored object in 'inout_labeling', and update 'inout_anchoredRowsSet' and 'inout_anchoredEntriesSet' accordingly
/************************************************************************/
{
    //1. Get max non-anchored entry
    const RowColInfo maxEntry = GetMaxEntryInNonAnchoredRow(inout_labeling, inout_anchoringData);
    Utilities::LogAndAbortIf(!maxEntry.IsValid(), 
        "In 'RelaxationLabelingSolverBase::AnchorMaxEntryInLabelingMatrix': !maxEntry.IsValid()");

    //2. Update and do anchoring
    const RowColInfoSet newEntriesToBeAnchoredSet = {maxEntry};
    Utilities::Log("Anchoring max entry, with value of " + std::to_string(inout_labeling.GetLabelingValue(maxEntry.m_row, maxEntry.m_col)));
    UpdateAndDoAnchoring(newEntriesToBeAnchoredSet, inout_labeling, inout_anchoringData);

    return newEntriesToBeAnchoredSet;
}

/************************************************************************/
double RelaxationLabelingSolverBase::ComputeAverageLocalConsistency(const Labeling& in_labeling, const SupportType& in_support) const
/************************************************************************/
{
    return Labeling::MultiplyAndSumLabelings(in_labeling, in_support);
}

/************************************************************************/
void RelaxationLabelingSolverBase::DoOneUpdateStep(const Labeling& in_oldLabeling, const SupportType& in_support, Labeling& out_newLabeling) const
/************************************************************************/
{
    std::for_each(executionPolicy, m_allPossibleObjects.begin(), m_allPossibleObjects.end(),
    [&](const Object* in_object) -> void
    {
        DoObjectUpdateStep(in_object, in_oldLabeling, in_support, out_newLabeling);
    });
}

/************************************************************************/
void RelaxationLabelingSolverBase::DoOneUpdateStepEfficient(const Labeling& in_oldLabeling, const SupportType& in_support, Labeling& out_newLabeling,
    const AnchoringData& in_anchoringData) const
/************************************************************************/
{
    //1. Update non-anchored rows
    std::for_each(executionPolicy, in_anchoringData.m_nonAnchoredRowsSet.begin(), in_anchoringData.m_nonAnchoredRowsSet.end(),
    [&](const int32_t& in_objectIndex) -> void
    {
        DoObjectUpdateStep(m_allPossibleObjects[in_objectIndex], in_oldLabeling, in_support, out_newLabeling);
    });

    //2. Copy anchored rows from 'in_oldLabeling'
    std::for_each(executionPolicy, in_anchoringData.m_anchoredRowsSet.begin(), in_anchoringData.m_anchoredRowsSet.end(),
    [&](const int32_t& in_objectIndex) -> void
    {
        for (int j = 0; j < m_allPossibleLabels.size(); ++j)
        {
            const double val = in_oldLabeling.GetLabelingValue(in_objectIndex, j);
            out_newLabeling.SetLabelingValue(in_objectIndex, j, val);
        }
    });
}

/************************************************************************/
void RelaxationLabelingSolverBase::DoObjectUpdateStep(const Object* in_object, const Labeling& in_oldLabeling, const SupportType& in_support, 
    Labeling& out_newLabeling) const
/************************************************************************/
{
    std::vector<double> nonNormalizedLabelingValsVec = std::vector(m_allPossibleLabels.size(), 0.0);

    //1. Save all non normalized labeling values, and save normalizer in 'objectNormalizer'
    double objectNormalizer = 0.0;
    for (const Label* label: m_allPossibleLabels)
    {
        const double p_i_lam = in_oldLabeling.GetLabelingValue(*in_object, *label);
        const double q_i_lam = in_support.GetLabelingValue(*in_object, *label);

        double valWithNoNormalizer = 0;
        valWithNoNormalizer = p_i_lam * q_i_lam;

        nonNormalizedLabelingValsVec[label->m_index] = valWithNoNormalizer;
        objectNormalizer += valWithNoNormalizer;
    }

    //2. Log in case something went wrong and 'objectNormalizer' is 0
    if (0.0 == objectNormalizer)
    {
        Utilities::Log("0.0 == objectNormalizer, for object " + in_object->GetString()); 

        if (in_oldLabeling.GetLabelingMatrix().row(in_object->m_index).isZero())
            Utilities::Log("The reason is that object's " + in_object->GetString() + " row in the labeling matrix is all zeros");

        if constexpr (!DO_NOT_COMPUTE_UNNECESSARY_SUPPORT_VALUES)
        {
            if (in_support.GetLabelingMatrix().row(in_object->m_index).isZero())
                Utilities::Log("The reason is that object's " + in_object->GetString() + " row in the support matrix is all zeros");
        }
    }

    //3. Normalize value and set them to _outNewLabeling
    if (0.0 == objectNormalizer)
    {
        //For binary rows, we just copy the value from the old labeling
        //(important: this is actually a fix to a quite common problem with anchoring - that 0.0 == objectNormalizer) 
        for (const Label* label : m_allPossibleLabels)
        {
            const double val = in_oldLabeling.GetLabelingValue(*in_object, *label);
            out_newLabeling.SetLabelingValue(*in_object, *label, val);
        }
    }
    else
    {
        for (const Label* label : m_allPossibleLabels)
        {
            const double val = nonNormalizedLabelingValsVec[label->m_index] / objectNormalizer;
            Utilities::LogAndAbortIf(!(val >= 0 && val <= 1), "val is not in range of [0,1]");           
            out_newLabeling.SetLabelingValue(*in_object, *label, val);
        }
    }
}

/************************************************************************/
void RelaxationLabelingSolverBase::GetSupport(const Labeling& in_labeling, SupportType& out_support) const
/************************************************************************/
{
    std::for_each(executionPolicy, m_allPossibleObjects.begin(), m_allPossibleObjects.end(),
        [&](const Object* in_object) -> void
        {
            for (const Label* label : m_allPossibleLabels)
            {
                if constexpr (DO_NOT_COMPUTE_UNNECESSARY_SUPPORT_VALUES)
                {
                    // Here we do not compute the support value since we will not use it (if the corresponding labeling value is 0, the support value will never
                    // be used, since in both ALC and update rule computation we multiply it with the corresponding labeling value)

                    const double labelingVal = in_labeling.GetLabelingValue(*in_object, *label);

                    double value = 0.0;
                    if (0.0 != labelingVal)
                        value = GetObjectAndLabelSupport(in_object, label, in_labeling);

                    out_support.SetLabelingValue(*in_object, *label, value);
                }
                else
                {
                    const double value = GetObjectAndLabelSupport(in_object, label, in_labeling);
                    out_support.SetLabelingValue(*in_object, *label, value);
                }
            }
        }
    );
}

/************************************************************************/
double RelaxationLabelingSolverBase::GetObjectAndLabelSupport(const Object* in_object, const Label* in_label, const Labeling& in_labeling) const
/************************************************************************/
{
    return ComputePairwiseSupport(in_object, in_label, in_labeling);
}

/************************************************************************/
double RelaxationLabelingSolverBase::ComputePairwiseSupport(const Object* in_object, const Label* in_label, const Labeling& in_labeling) const
/************************************************************************/
{
#if IS_SAVE_ALL_PAIRS_SUPPORT_COMPUTATION_METHOD()
    return ComputePairwiseSupportWithCorrelatedPairsMatrix(in_object, in_label, in_labeling);
#elif IS_STRAIGHTFORWARD_SUPPORT_COMPUTATION_METHOD()
    return ComputePairwiseSupportInStraightforwardWay(in_object, in_label, in_labeling);
#endif
}

#if IS_SAVE_ALL_PAIRS_SUPPORT_COMPUTATION_METHOD()
/************************************************************************/
double RelaxationLabelingSolverBase::ComputePairwiseSupportWithCorrelatedPairsMatrix(const Object* in_object, const Label* in_label, const Labeling& in_labeling) const
/************************************************************************/
{
    //1. Compute pairwise support value for object 'in_object' and label 'in_label' with pairs in 'objectLabelPairsVec'
    double support = 0.0;
    const ObjectLabelPairsVec& objectLabelPairsVec = *m_correlatedPairsMatrix(in_object->m_index, in_label->m_index);
    for (const ObjectLabelPair& objectLabelPair : objectLabelPairsVec)
    {
        const double labelingVal = in_labeling.GetLabelingValue(*objectLabelPair.first, *objectLabelPair.second);
        if (0.0 != labelingVal)
        {
            const double compVal = GetCompatibilityValue(in_object, in_label, objectLabelPair.first, objectLabelPair.second);
            support += compVal * labelingVal;
        }
    }

    return support;
}
#endif

#if IS_STRAIGHTFORWARD_SUPPORT_COMPUTATION_METHOD()
/************************************************************************/
double RelaxationLabelingSolverBase::ComputePairwiseSupportInStraightforwardWay(const Object* in_object, const Label* in_label, const Labeling& in_labeling) const
/************************************************************************/
{
    //1. Compute pairwise support value for object 'in_object' and label 'in_label'
    double support = 0.0;
    for (const Object* currObject: m_allPossibleObjects)
    {
        for (const Label* currLabel: m_allPossibleLabels)
        {
            const double compVal = GetCompatibilityValue(in_object, in_label, currObject, currLabel);
            const double labelingVal = in_labeling.GetLabelingValue(*currObject, *currLabel);
            support += compVal * labelingVal;
        }
    }

    //2. Check that support is non-negative
    Utilities::LogAndAbortIf(support < 0.0, "support < 0.0");

    return support;
}
#endif