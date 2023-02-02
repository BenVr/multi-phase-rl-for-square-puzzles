#include "PuzzleRL_Solver.h"
#include "FileSystemUtils.h"
#include "PuzzleRL_SolverConstants.h"
#include <Eigen/Dense>

extern Utilities::Logger g_logger;

/************************************************************************/
PuzzleRL_Solver::PuzzleRL_Solver(const PuzzleImage& in_puzzleImage,
    const PuzzleRL_SolverConfiguration& in_config, const std::string& in_sourceImagePath, const ImageRotation& in_pieceRotationForType2Puzzles, 
    const int32_t in_runNumber) :
RelaxationLabelingSolverBase(in_config, in_runNumber, 
    PuzzleRL_Solver::GetMaxRL_Iterations(in_config.GetPuzzleType(), in_puzzleImage)),
m_puzzleImage(in_puzzleImage),
m_config(in_config),
m_sourceImagePath(in_sourceImagePath),
m_puzzleRL_SolverOutputManager(this),
m_piecesBlockTranslationManager(this)
/************************************************************************/
{
    Utilities::LogAndAbortIf(in_pieceRotationForType2Puzzles != eInvalidRotation && PuzzleType::eType2_UknownRotation != m_config.GetPuzzleType(),
        "In 'PuzzleRL_Solver::PuzzleRL_Solver()': rotation given for type 1 puzzle");

    if (in_pieceRotationForType2Puzzles != eInvalidRotation)
        m_allowedRotationForFirstRotationDeterminedPiece = in_pieceRotationForType2Puzzles;
        
    m_numOfPossiblePieceRotations = (m_config.GetPuzzleType() == PuzzleType::eType1_NoRotation) ? 1 : ImageRotation::eNumOfImageRotations;
}

/************************************************************************/
PuzzleRL_Solver::~PuzzleRL_Solver()
/************************************************************************/
{
    delete m_initLabeling;

    DeleteObjectsPool();
    DeleteLabelsPool();
    DeletePieceObjectsAndRotationsObjects();
}

/************************************************************************/
void PuzzleRL_Solver::Init()
//Function input: none
//Function output: none
//Function objective: initiate all data needed for PuzzleRL_Solver
/************************************************************************/
{
    //1. Init objects and labels in PuzzleRL_Solver
    InitAllObjectsPool();
    InitAllLabelsPool();

    //2. Init objects and labels in RelaxationLabelingSolverBase
    SetAllPossibleObjects(ObjectsVector(m_puzzlePieceObjectsPool.begin(), m_puzzlePieceObjectsPool.end()));
    SetAllPossibleLabels(LabelsVector(m_locationRotationLabelsPool.begin(), m_locationRotationLabelsPool.end()));

    //3. initiate data related to handling to pieces rotations together 
    InitPieceAndRotationsData();

    //4. Set compatibility function
    InitSolverCompatibility();

    //5. Call base class init
    RelaxationLabelingSolverBase::Init();

    //6. Set initial labeling
    SetInitialLabeling();

    //7. Init data related to support computation optimization
    RelaxationLabelingSolverBase::InitSupportComputationMethodData();
}

/************************************************************************/
void PuzzleRL_Solver::OutputLabelingDuringAlg(const Labeling& in_labeling, const int32_t in_iterationNum) const
//Function input: in_labeling: labeling, in_iterationNum: iteration number
//Function output: none
//Function objective: output iteration labeling image
/************************************************************************/
{
    //1. Get 'labelingToVisualize', and get 'image' from it
    const Labeling labelingToVisualize = in_labeling.GetPermutationPartOfLabeling();

    const cv::Mat image = GetAssemblyImageFromBinaryLabeling(labelingToVisualize);

    //2. Write image
    ImageUtils::WriteImage(GetIterationImageFileName(in_iterationNum), image);
}

/************************************************************************/
void PuzzleRL_Solver::OutputSolutionLabeling(const Labeling& in_labeling) const
//Function input: in_labeling: labeling
//Function output: none
//Function objective: output solution labeling image
/************************************************************************/
{
    //1. Get binary labeling, and get 'solutionImage' from it
    const Labeling binaryLabeling = in_labeling.GetBinaryLabeling();
    const cv::Mat solutionImage = GetAssemblyImageFromBinaryLabeling(binaryLabeling);

    //2. Write image
    ImageUtils::WriteImage(GetSolutionImagePath(), solutionImage);
}

/************************************************************************/
bool PuzzleRL_Solver::HasSolutionChangedInIteration(const Labeling& in_newLabeling) const
/************************************************************************/
{
    //1. Get new 'top labels matrix'
    const Eigen::MatrixXi newTopLabelsMatrix = in_newLabeling.GetPermutationPartOfLabeling().GetLabelingMatrix().cast<int>();
    
    //2. If the new 'top labels matrix' is the same as before (as m_currentTopLabelsMatrix) - return false
    if (newTopLabelsMatrix.isApprox(m_currentTopLabelsMatrix))
        return false;
    //3. If the new 'top labels matrix' is not the same as before - update m_currentTopLabelsMatrix and return true
    else
    {
        m_currentTopLabelsMatrix = newTopLabelsMatrix;
        return true;
    }
}

/************************************************************************/
void PuzzleRL_Solver::PostMakeLabelingConsistent(const Labeling& in_labeling)
//Function input: in_labeling: labeling
//Function output: none
//Function objective: set m_finalPieceAssignmentMatrix, which represents the piece assignment suggested by in_labeling
/************************************************************************/
{
    //1. Get binary labeling
    const Labeling binaryLabeling = in_labeling.GetBinaryLabeling();

    //2. Set 'm_finalPieceAssignmentMatrix' according to the piece assignment matrix created from 'binaryLabeling'
    m_finalPieceAssignmentMatrix = GetAssignmentMatrixFromBinaryLabeling(binaryLabeling);

    //3. Set 'm_isSolutionAssignmentFeasible' according to 'm_finalPieceAssignmentMatrix' 
    m_numOfAssignedPiecesInSolution = ImageUtils::GetNumOfAssignedPiecesInPieceAssignmentMatrix(m_finalPieceAssignmentMatrix);
    m_isSolutionAssignmentFeasible = (m_numOfAssignedPiecesInSolution == m_puzzleImage.GetNumOfPieces());
}

/************************************************************************/
bool PuzzleRL_Solver::ShouldDoRandomAnchoringAndThenBreak(const std::set<int32_t>& in_anchoredRowsSet, std::string& out_breakAfterAnchoringStr) const
//Function input: in_anchoredRowsSet: set of anchored rows; out_breakAfterAnchoringStr: string to be filled with break info
//Function output: boolean indicating whether new objects were anchored  
//Function objective: return boolean as described in "Function output", and update 'out_breakAfterAnchoringStr' with break info
/************************************************************************/
{
    bool shouldBreak = false;

    if (in_anchoredRowsSet.size() != m_allPossibleObjects.size() && AreAllNonAnchoredPiecesConstantPieces(in_anchoredRowsSet))
    {
        out_breakAfterAnchoringStr = "Breaking since all non anchored pieces are constant pieces!";
        shouldBreak = true;
    }

    return shouldBreak;
}

/************************************************************************/
bool PuzzleRL_Solver::AreAllNonAnchoredPiecesConstantPieces(const std::set<int32_t>& in_anchoredRowsSet) const
//Function input: in_anchoredRowsSet: set of anchored rows
//Function output: boolean indicating whether all non-anchored objects are constant pieces
//Function objective: described in "Function output"
/************************************************************************/
{
    //For all rows
    for (int32_t i = 0; i < m_puzzlePieceObjectsPool.size(); ++i)
    {
        //If the row is non-anchored
        if (in_anchoredRowsSet.count(i) == 0)
        {
            //Get the relevant piece
            const PuzzlePieceObject* pieceObject = GetPuzzlePieceObjectByIndex(i);

            //If the non-anchored piece is not a constant piece - return false
            if (!IsConstantPiece(pieceObject))
                return false;
        }
    }

    return true;
}

/************************************************************************/
BooleansMatrix PuzzleRL_Solver::GetAnchoredPiecesMatrix(const AnchoringData& in_anchoringData) const
/************************************************************************/
{
    BooleansMatrix anchoredPiecesMatrix = BooleansMatrix::Constant(m_puzzleImage.GetNumOfRowPieces(), m_puzzleImage.GetNumOfColPieces(), false);
    for (const RowColInfo& currAnchoredEntry : in_anchoringData.m_anchoredEntriesSet)
    {
        const LocationRotationLabel* anchoredLabel = m_locationRotationLabelsPool[currAnchoredEntry.m_col];
        anchoredPiecesMatrix(anchoredLabel->m_row, anchoredLabel->m_column) = true;
    }

    return anchoredPiecesMatrix;
}

/************************************************************************/
const PieceObjectAndRot* PuzzleRL_Solver::GetAnchoredRotatedPieceInCoord(const RowColInfo& in_coord,
    const AnchoringData& in_anchoringData) const
//Function input: in_coord: coordinate; in_anchoringData: anchoring data
//Function output: rotated piece anchored at position 'in_coord' 
//Function objective: described in "Function output"
/************************************************************************/
{
    //1. Get 'locationRotationLabelsVec' from 'in_coord' 
    const LocationRotationLabelsVector locationRotationLabelsVec = GetAllLocationRotationLabelsForLocation(in_coord);

    //2. Get one location rotation label, and the index of the piece anchored to it
    const LocationRotationLabel* locationRotationLabel = nullptr;
    int32_t pieceIndex = -1;

    if (PuzzleType::eType1_NoRotation == m_config.GetPuzzleType())
    {
        Utilities::LogAndAbortIf(locationRotationLabelsVec.size() != 1, "locationRotationLabelsVec.size() != 1");
        locationRotationLabel = locationRotationLabelsVec[0];
        pieceIndex = in_anchoringData.GetPieceAnchoredToLocationIndex(locationRotationLabel->m_index);
    }
    else
    {
        Utilities::LogAndAbortIf(locationRotationLabelsVec.size() != eNumOfPossibleOrientations, 
            "locationRotationLabelsVec.size() != eNumOfPossibleOrientations");

        std::set<int32_t> labelsSet;
        for (const LocationRotationLabel* currLabel: locationRotationLabelsVec)
            labelsSet.insert(currLabel->m_index);

        int32_t locationRotationLabelIndex = -1;
        pieceIndex = in_anchoringData.GetPieceAnchoredToOneLocationRotationIndex(labelsSet, locationRotationLabelIndex);
        locationRotationLabel = m_locationRotationLabelsPool[locationRotationLabelIndex];
    }

    //3. Get the rotated piece
    const PieceObjectAndRot* rotatedPiece = GetRotatedPieceFromPieceObjectAndRotation(m_puzzlePieceObjectsPool[pieceIndex], 
        locationRotationLabel->m_rotation);

    return rotatedPiece;
}

/************************************************************************/
RowColInfoSet PuzzleRL_Solver::GetSetOfEntriesToBeAnchored(const Labeling& in_labeling, const AnchoringData& in_anchoringData) const
//Function input: in_labeling: labeling; in_anchoredRowsSet: already-anchored rows
//Function output: set of new entries to be anchored 
//Function objective: described in "Function output"
/************************************************************************/
{
    //1. Init variables
    std::vector<LabelingEntryInfo> candidateEntriesForAnchoring;
    const Eigen::MatrixXd labelingMatrix = in_labeling.GetLabelingMatrix();

    //2. For all rows
    for (const int32_t& i : in_anchoringData.m_nonAnchoredRowsSet)
    {
        //2.1. If 'i' is non-anchored and its max entry is >= m_anchoringThreshold, then insert its max entry info to 'candidateEntriesForAnchoring' 
        int32_t columnOfMaxEntryInCurrRow;
        const double maxEntryInCurrRow = labelingMatrix.row(i).maxCoeff(&columnOfMaxEntryInCurrRow);

        if (maxEntryInCurrRow >= m_anchoringThreshold)
        {
            const Utilities::RowColInfo entry(i, columnOfMaxEntryInCurrRow);
            if (VerifyAnchoringLegality(entry, in_anchoringData))
                candidateEntriesForAnchoring.emplace_back(entry, maxEntryInCurrRow);
        }
    }

    //3. Sort anchoring candidates
    SortAnchoringCandidates(candidateEntriesForAnchoring, in_anchoringData);

    //4. Init 'entriesToBeAnchored' and 'colsToBeAnchored' ('colsToBeAnchored' is just for aborting in case of a problem)
    RowColInfoSet entriesToBeAnchored;

    //5. Add new rows to anchor to 'rowsToBeAnchored'
    if (!candidateEntriesForAnchoring.empty())
    {
        const Utilities::RowColInfo entryCoord = candidateEntriesForAnchoring[0].first;
        entriesToBeAnchored.insert(entryCoord);
    }

    return entriesToBeAnchored;
}

/************************************************************************/
Labeling PuzzleRL_Solver::DoAnchoring(const Labeling& in_labeling, 
    const RowColInfoSet& in_allEntriesToBeAnchoredSet) const
//Function input: in_labeling: labeling; in_allEntriesToBeAnchoredSet: set of anchored rows to anchor
//Function output: 'in_labeling' after anchoring entries 
//Function objective: described in "Function output"
/************************************************************************/
{
    //This function is implemented in 'PuzzleRL_Solver' due to type-2 puzzles
    //(since that in type 2 puzzles, for each anchored entry we zero 4 columns)
    
    //1. Init variables
    const int32_t numOfNonAnchoredObjects = static_cast<int32_t>(m_allPossibleObjects.size() - in_allEntriesToBeAnchoredSet.size());
    const double probValForRest = 1.0 / (m_numOfPossiblePieceRotations * static_cast<double>(numOfNonAnchoredObjects));

    //2. Init the 'newLabelingMatrix' which will finally be the labeling matrix of 'in_labeling', but with anchoring 
    Eigen::MatrixXd newLabelingMatrix = in_labeling.GetLabelingMatrix();
    newLabelingMatrix.setConstant(probValForRest);

    //3. In 'newLabelingMatrix', determine the rotation of pieces for which the rotation was already determined
    if (PuzzleType::eType2_UknownRotation == m_config.GetPuzzleType())
    {
        Labeling tempLabeling(newLabelingMatrix);
        ManipulateLabelingForType2PuzzlesBySinglePiece(tempLabeling, numOfNonAnchoredObjects);
        newLabelingMatrix = tempLabeling.GetLabelingMatrix();
    }

    //4. Anchor rows
    std::for_each(executionPolicy, in_allEntriesToBeAnchoredSet.begin(), in_allEntriesToBeAnchoredSet.end(),
    [&](const RowColInfo& in_entry) -> void
    {
        //4.1. Zero row
        newLabelingMatrix.row(in_entry.m_row).setZero();

        //4.2. Zero column/s (it's 4 columns in type 2 puzzle)
        const LocationRotationLabel* locationRotationLabel = m_locationRotationLabelsPool[in_entry.m_col];
        const LocationRotationLabelsVector locationRotationLabelsVec = 
            GetAllLocationRotationLabelsForLocation(Utilities::RowColInfo(locationRotationLabel->m_row, locationRotationLabel->m_column));

        for (const LocationRotationLabel* currLocationRotationLabel : locationRotationLabelsVec)
            newLabelingMatrix.col(currLocationRotationLabel->m_index).setZero();

        //4.3. Set relevant entry to 1
        newLabelingMatrix(in_entry.m_row, in_entry.m_col) = 1;
    });

    return Labeling(newLabelingMatrix);
}

/************************************************************************/
void PuzzleRL_Solver::SortAnchoringCandidates(std::vector<LabelingEntryInfo>& inout_candidateEntriesForAnchoring,
     const AnchoringData& in_anchoringData) const
//Function input: inout_candidateEntriesForAnchoring: anchoring candidates vector; in_anchoringData: anchoring data
//Function output: none 
//Function objective: sort 'inout_candidateEntriesForAnchoring' by relevance
/************************************************************************/
{
    //1. Set variables
    const int32_t numOfRows = m_puzzleImage.GetNumOfRowPieces();
    const int32_t numOfCols = m_puzzleImage.GetNumOfColPieces();
    const BooleansMatrix anchoredPiecesMatrix = GetAnchoredPiecesMatrix(in_anchoringData);

    //2. Set 'AnchoringCandidateInfo' struct
    struct AnchoringCandidateInfo
    {
        int32_t m_numOfBestBuddyNeighbors = 0;
        double m_sumOfNeighboringComps = 0.0;
    };            

    //3. Set 'getAnchoringCandidateInfo' function
    const auto getAnchoringCandidateInfo = 
    [&](const LabelingEntryInfo& in_entryInfo) -> AnchoringCandidateInfo
    {
        //3.1. Set variables
        AnchoringCandidateInfo info;

        const PieceObjectAndRot* rotatedPiece = nullptr; 
        RowColInfo coordinate = RowColInfo::m_invalidRowColInfo;
        GetRotatedPieceAndCoordinateFromLabelingEntry(in_entryInfo.first, rotatedPiece, coordinate);

        Utilities::LogAndAbortIf(true == anchoredPiecesMatrix(coordinate.m_row, coordinate.m_col), 
            "in 'getAnchoringCandidateInfo()': true == anchoredPiecesMatrix(coordinate.m_row, coordinate.m_col)");

        //3.2. Update data according to all neighbors
        for (Orientation orien = Orientation::eUp; orien < Orientation::eNumOfPossibleOrientations; orien = static_cast<Orientation>(orien + 1))
        {
            const RowColInfo currNeighbor = RowColInfo::NeighborCoordByOrien(coordinate.m_row, coordinate.m_col, orien);

            if (currNeighbor.IsValid(numOfRows, numOfCols) && true == anchoredPiecesMatrix(currNeighbor.m_row, currNeighbor.m_col))
            {
                const PieceObjectAndRot* neighborRotatedPiece = GetAnchoredRotatedPieceInCoord(currNeighbor, in_anchoringData);

                //Update 'info.m_sumOfNeighboringComps'
                info.m_sumOfNeighboringComps += m_piecesCompatibilities(rotatedPiece->m_index, neighborRotatedPiece->m_index, orien);

                const bool areBestBuddies = AreBestBuddies(rotatedPiece, neighborRotatedPiece, orien, m_piecesCompatibilities);
                if (areBestBuddies)
                {
                    //Update 'info.m_numOfBestBuddyNeighbors' and 'info.m_sumOfBestBuddyProps'
                    ++info.m_numOfBestBuddyNeighbors;
                }
            }
        }

        return info;
    };

    //4. Sort 'in_anchoringData' 
    std::sort(inout_candidateEntriesForAnchoring.begin(), inout_candidateEntriesForAnchoring.end(), 
    [&](const LabelingEntryInfo& in_lhs, const LabelingEntryInfo& in_rhs)
    {
        const AnchoringCandidateInfo lhsInfo = getAnchoringCandidateInfo(in_lhs);
        const AnchoringCandidateInfo rhsInfo = getAnchoringCandidateInfo(in_rhs);

        if (lhsInfo.m_numOfBestBuddyNeighbors != rhsInfo.m_numOfBestBuddyNeighbors)
        {
            return lhsInfo.m_numOfBestBuddyNeighbors > rhsInfo.m_numOfBestBuddyNeighbors;
        }
        else 
        {
            if (lhsInfo.m_sumOfNeighboringComps != rhsInfo.m_sumOfNeighboringComps)
            {
                return lhsInfo.m_sumOfNeighboringComps > rhsInfo.m_sumOfNeighboringComps; 
            }
            else
            {
                return in_lhs.second > in_rhs.second;
            }
        }
    });
}

/************************************************************************/
bool PuzzleRL_Solver::DoTranslationAndPossiblyBranchAlg(Labeling& inout_labeling, const RowColInfoSet& in_newJustAnchoredEntries,
    RL_algorithmRunningInfo& inout_algRunningInfo, bool& out_shouldBreakAfterTranslation) const
//Function input: inout_labeling: labeling; in_newJustAnchoredEntries: set of new just anchored entries;
    //inout_algRunningInfo: algorithm running info; out_shouldBreakAfterTranslation: boolean indicating whether we should break after this functiom
//Function output: boolean indicating whether translation was done or not
//Function objective: if needed, do vertical and horizontal translation of the pieces block represented by 'inout_labeling'
    //If there's a translation dilemma, "branch" the algorithm 
/************************************************************************/
{
    //1. Try to do translation
    const TranslationMode translationMode = m_piecesBlockTranslationManager.DoTranslation(inout_labeling, in_newJustAnchoredEntries, 
        inout_algRunningInfo.m_anchoringData.m_anchoredEntriesSet);

    //2. Decide what to do after translation:
    //a. If there was no translation or there was a translation - just set boolean
    //b. If there was a translation dilemma - branch the algorithm
    bool didTranslation = false;
    switch (translationMode)
    {
        case TranslationMode::eDidNoTranslation:
            out_shouldBreakAfterTranslation = false;
            didTranslation = false;
            break;

        case TranslationMode::eDidTranslation:
            out_shouldBreakAfterTranslation = false;
            didTranslation = true;
            break;

        case TranslationMode::eVerticalTranslationUpDilemma:
        case TranslationMode::eVerticalTranslationDownDilemma:
        case TranslationMode::eHorizontalTranslationLeftDilemma:
        case TranslationMode::eHorizontalTranslationRightDilemma:
            BranchDueToTranslationDilemma(inout_labeling, inout_algRunningInfo, translationMode);
            out_shouldBreakAfterTranslation = true;
            didTranslation = false;
            break;

        default:
            Utilities::LogAndAbort("In 'PuzzleRL_Solver::DoTranslationAndPossiblyBranchAlg()': shouldn't get here");
            break;
    }

    return didTranslation;
}

/************************************************************************/
void PuzzleRL_Solver::BranchDueToTranslationDilemma(Labeling& inout_labeling, RL_algorithmRunningInfo& inout_algRunningInfo,
    const TranslationMode& in_transDilemma) const
//Function input: inout_labeling: labeling; inout_algRunningInfo: algorithm running info;
    //in_transDecision: the translation that we wonder if to perform
//Function output: none
//Function objective: perform two options: translation according to 'in_transDecision', and doing nothing
    //update 'inout_labeling' and 'inout_algRunningInfo' according the best option (the one that achieved greater ALC) 
/************************************************************************/
{
    //1. First output "before decision" image
    OutputLabelingDuringAlg(inout_labeling, inout_algRunningInfo.m_iterationNum);

    //2. Disallow similar translations (so we'll not apply vertical/horizontal translation twice)
    m_piecesBlockTranslationManager.DisallowTranslation(in_transDilemma);

    //3. Set choices variables
    double transALC = 0;
    Labeling translationLabeling(inout_labeling);
    RL_puzzleSolverAlgorithmRunningInfo transAlgRunningInfo(*static_cast<RL_puzzleSolverAlgorithmRunningInfo*>(&inout_algRunningInfo));
    const TranslationDecision transDecision = RLSolverGeneralUtils::GetTranslationDecisionFromDilemma(in_transDilemma);

    double doNothingALC = 0;
    Labeling doNothingLabeling(inout_labeling);
    RL_puzzleSolverAlgorithmRunningInfo doNothingAlgRunningInfo(*static_cast<RL_puzzleSolverAlgorithmRunningInfo*>(&inout_algRunningInfo));

    //3. Apply first choice: translation
    ApplyTranslationDecisionAndPrePostActions(transDecision, in_transDilemma, translationLabeling, transAlgRunningInfo, transALC);

    //5. Apply second choice: doing nothing
    ApplyTranslationDecisionAndPrePostActions(TranslationDecision::eNoTranslation, in_transDilemma, doNothingLabeling, doNothingAlgRunningInfo, doNothingALC);

    //6. Update translation status
    m_piecesBlockTranslationManager.AllowTranslation(in_transDilemma);

    //7. Determine winning decision, and update variables accordingly
    if (transALC > doNothingALC)
    {
        Utilities::Log(GetTranslationDecisionString(transDecision) + " is better in " + GetTranslationDilemmaString(in_transDilemma) + "!!");
        Utilities::Log("since " + GetTranslationDecisionString(transDecision) + " > " + GetTranslationDecisionString(TranslationDecision::eNoTranslation) +
            " (" + std::to_string(transALC) + " > " + std::to_string(doNothingALC) + ")");

        inout_labeling = translationLabeling;
        inout_algRunningInfo.Set(transAlgRunningInfo);
    }
    //else if (transALC < doNothingALC)
    else if (transALC <= doNothingALC)
    {
        Utilities::Log(GetTranslationDecisionString(TranslationDecision::eNoTranslation) + " is better in " 
            + GetTranslationDilemmaString(in_transDilemma) + "!!");
        Utilities::Log("since " + GetTranslationDecisionString(transDecision) + " < " + GetTranslationDecisionString(TranslationDecision::eNoTranslation) +
            " (" + std::to_string(transALC) + " < " + std::to_string(doNothingALC) + ")");

        inout_labeling = doNothingLabeling;
        inout_algRunningInfo.Set(doNothingAlgRunningInfo);
    }
    else
    {
        Utilities::LogAndAbort("transALC == doNothingALC");    
    }
}

/************************************************************************/
void PuzzleRL_Solver::ApplyTranslationDecisionAndPrePostActions(const TranslationDecision& in_transDecision, const TranslationMode& in_transDilemma, 
    Labeling& inout_labeling, RL_puzzleSolverAlgorithmRunningInfo& inout_puzzleAlgRunningInfo, double &out_finalALC) const
//Function input: in_transDecision: translation decision; in_transDilemma: translation dilemma; inout_labeling: labeling;
    //inout_puzzleAlgRunningInfo: algorithm running info; out_finalALC: final alc value
//Function output: none
//Function objective: perform 'in_transDecision' with pre- and post-actions, and return updated 'inout_labeling' and 'inout_puzzleAlgRunningInfo'
/************************************************************************/
{
    //1. Pre-actions: increase dilemma level and update translation decision
    ++inout_puzzleAlgRunningInfo.m_translationDilemmaLevel;
    inout_puzzleAlgRunningInfo.UpdateTranslationDecision(in_transDecision);

    //2. Apply translation decision                
    ApplyTranslationDecision(in_transDecision, in_transDilemma, inout_labeling, inout_puzzleAlgRunningInfo);

    //3. Post-actions: decrease dilemma level and set ALC values (in case of constant pieces, they are computed according to non-manipulated comp)
    --inout_puzzleAlgRunningInfo.m_translationDilemmaLevel;

    if (m_constantPiecesVector.empty())
        out_finalALC = inout_puzzleAlgRunningInfo.m_finalAlc;
    else
        out_finalALC = GetAverageLocalConsistencyWithNonManipulatedCompatibility(inout_labeling);
}

/************************************************************************/
void PuzzleRL_Solver::ApplyTranslationDecision(const TranslationDecision& in_transDecision, const TranslationMode& in_transDilemma,
    Labeling& inout_labeling, RL_puzzleSolverAlgorithmRunningInfo& inout_puzzleAlgRunningInfo) const
//Function input: in_transDecision: translation decision; in_transDilemma: translation dilemma; inout_labeling: labeling;
    //inout_puzzleAlgRunningInfo: algorithm running info;
//Function output: none
//Function objective: perform 'in_transDecision', and return updated 'inout_labeling' and 'inout_puzzleAlgRunningInfo'
/************************************************************************/
{
    //1. Get strings
    const std::string transDecisionStr = RLSolverGeneralUtils::GetTranslationDecisionString(in_transDecision);
    const std::string transDilemmaStr = RLSolverGeneralUtils::GetTranslationDilemmaString(in_transDilemma);

    //2. Print start title 
    const std::string startTranslationTitle = 
        m_puzzleRL_SolverOutputManager.GetStartTranslationDilemmaString(inout_puzzleAlgRunningInfo.m_translationDilemmaLevel, transDecisionStr, transDilemmaStr);
    Utilities::Log(startTranslationTitle);

    //3. Make labeling consistent according to 'in_transDecision'
    MakeLabelingConsistentWithTranslationDecision(inout_labeling, in_transDecision, in_transDilemma, inout_puzzleAlgRunningInfo);

    //4. Print done title 
    const std::string doneTranslationTitle = 
        m_puzzleRL_SolverOutputManager.GetDoneTranslationDilemmaString(inout_puzzleAlgRunningInfo.m_translationDilemmaLevel, transDecisionStr, transDilemmaStr);
    Utilities::Log(doneTranslationTitle);
}

/************************************************************************/
void PuzzleRL_Solver::MakeLabelingConsistentWithTranslationDecision(Labeling& inout_labeling,
    const TranslationDecision& in_transDecision, const TranslationMode& in_transDilemma, RL_algorithmRunningInfo& inout_algRunningInfo) const
//Function input: inout_labeling: labeling; in_transDecision: translation decision; in_transDilemma: decision dilemma
    //inout_algRunningInfo: algorithm running info;
//Function output: none
//Function objective: apply translation of 'in_transDecision', and return updated 'inout_labeling' and 'inout_puzzleAlgRunningInfo'
/************************************************************************/
{
    //1. Save old run folder and update 'm_currentTopLabelsMatrix' 
    const std::string oldRunFolder = m_currOutputFolder;
    m_currentTopLabelsMatrix = inout_labeling.GetBinaryLabeling().GetLabelingMatrix().cast<int>();

    //2. Set current output folder
    SetCurrOutputFolderBeforeTranslationBranching(in_transDecision, in_transDilemma);

    //3. Apply 'in_transDecision' on 'inout_labeling' 
    m_piecesBlockTranslationManager.TranslatePiecesBlock(inout_labeling, in_transDecision, inout_algRunningInfo.m_anchoringData);

    //4. Set efficiency params in 'm_piecesBlockTranslationManager'
    m_piecesBlockTranslationManager.SetEfficiencyParams(inout_algRunningInfo.m_anchoringData.m_anchoredEntriesSet);

    //5. Output labeling after translation
    OutputLabelingDuringAlg(inout_labeling, inout_algRunningInfo.m_iterationNum);

    //6. Make labeling consistent
    MakeLabelingConsistent(inout_labeling, inout_algRunningInfo);

    //7. Do some actions after convergence
    DoPostConvergenceActions(inout_labeling);

    //8. Re-set the output folder
    SetCurrentOutputFolder(oldRunFolder);
}

/************************************************************************/
void PuzzleRL_Solver::SetCurrOutputFolderBeforeTranslationBranching(const TranslationDecision& in_transDecision, 
    const TranslationMode& in_transDilemma) const
/************************************************************************/
{
    const std::string transFolderStr = GetTranslationDilemmaString(in_transDilemma) + " - " + GetTranslationDecisionString(in_transDecision) + "/";
    SetAndCreateCurrentOutputAndIterationsFolders(FileSystemUtils::GetPathInFolder(m_currOutputFolder, transFolderStr));
}

/************************************************************************/
double PuzzleRL_Solver::GetAverageLocalConsistencyWithNonManipulatedCompatibility(const Labeling& in_labeling) const
//Function input: in_labeling: labeling;
//Function output: average local consistency computed according to non-manipulated compatibility (relevant only for case of constant pieces)
//Function objective: described in "Function output"
/************************************************************************/
{
    const SupportType support = GetCompatibilitySupportWithNonManipulatedCompatibility(in_labeling);
    const double alc = Labeling::MultiplyAndSumLabelings(in_labeling, support);

    return alc;
}

/************************************************************************/
SupportType PuzzleRL_Solver::GetCompatibilitySupportWithNonManipulatedCompatibility(const Labeling& in_labeling) const
//Function input: in_labeling: labeling;
//Function output: support computed according to non-manipulated compatibility (relevant only for case of constant pieces)
//Function objective: described in "Function output"
/************************************************************************/
{
    SupportType support = SupportType(m_allPossibleObjects.size(), m_allPossibleLabels.size());

    const auto GetCompatibilityValueReal = [&](const Object* in_object1, const Label* in_label1, const Object* in_object2, const Label* in_label2)
    {
        const double compVal = GetSolverCompatibilityValueWithPiecesComp(static_cast<const PuzzlePieceObject*>(in_object1),
        static_cast<const LocationRotationLabel*>(in_label1), static_cast<const PuzzlePieceObject*>(in_object2),
        static_cast<const LocationRotationLabel*>(in_label2), m_piecesCompatibilitiesNonManipulated);

        return compVal;
    };

    const auto getObjectAndLabelRealSupportFunc = [&](const Object* in_object, const Label* in_label, const Labeling& in_labeling)
    {
        //1. Compute pairwise support value for object 'in_object' and label 'in_label'
        double support = 0.0;
        const LabelsVector correlatedLabelsVec = GetCorrelatedLabelsVec(in_label);

        for (const Object* currObject: m_allPossibleObjects)
        {
            for (const Label* currLabel: correlatedLabelsVec)
            {
                const double compVal = GetCompatibilityValueReal(in_object, in_label, currObject, currLabel);
                const double labelingVal = in_labeling.GetLabelingValue(*currObject, *currLabel);
                support += compVal * labelingVal;
            }
        }

        //2. Check that support is non-negative
        Utilities::LogAndAbortIf(support < 0.0, "support < 0.0");

        return support;
    };


    std::for_each(executionPolicy, m_allPossibleObjects.begin(), m_allPossibleObjects.end(),
    [&](const Object* in_object) -> void
    {
        for (const Label* label : m_allPossibleLabels)
        {
            const double value = getObjectAndLabelRealSupportFunc(in_object, label, in_labeling);
            support.SetLabelingValue(*in_object, *label, value);
        }
    });

    return support;
}

/************************************************************************/
void PuzzleRL_Solver::DoPostConvergenceAnchoring(Labeling& inout_labeling, AnchoringData& inout_anchoringData) const
//Function input: inout_labeling: labeling; inout_anchoringData: anchoring data;
//Function output: none 
//Function objective: do random anchoring of non-anchored pieces in case of puzzle with constant pieces
/************************************************************************/
{
    //1. If all pieces are anchored- there's nothing to do
    if (inout_anchoringData.m_anchoredEntriesSet.size() == m_puzzlePieceObjectsPool.size())
        return;

    //2. If we have constant pieces and not all the object are anchored - we anchor them randomly
    AnchorNonAnchoredObjects(inout_labeling, inout_anchoringData);
}

/************************************************************************/
void PuzzleRL_Solver::AnchorNonAnchoredObjects(Labeling& inout_labeling, AnchoringData& inout_anchoringData) const
//Function input: inout_labeling: labeling; inout_anchoringData: anchoring data;
//Function output: none
//Function objective: anchor all non anchored objects in 'inout_labeling' (anchoring is done by values in 'inout_labeling')
/************************************************************************/
{
    Utilities::Log("Performing anchoring of all non anchored objects!");

    //While there are non anchored rows
    while (!inout_anchoringData.m_nonAnchoredRowsSet.empty())
    {
        const RowColInfo entryToAnchor = GetMaxEntryInNonAnchoredRow(inout_labeling, inout_anchoringData);

        const RowColInfoSet entriesToAnchorSet = {entryToAnchor};
        UpdateAndDoAnchoring(entriesToAnchorSet, inout_labeling, inout_anchoringData);
    }
}

/************************************************************************/
RowColInfo PuzzleRL_Solver::GetMaxEntryInNonAnchoredRow(const Labeling& in_labeling, const AnchoringData& in_anchoringData) const
//Function input: in_labeling: labeling; in_anchoredRowsSet: already-anchored rows
//Function output: max non-anchored entry in 'inout_labeling'
//Function objective: described in "Function output"
/************************************************************************/
{
    //1. Get set of anchored columns
    std::set<int32_t> anchoredColsSet;
    for (const RowColInfo& currEntry: in_anchoringData.m_anchoredEntriesSet)
    {
        const LocationRotationLabel* locationRotationLabel = m_locationRotationLabelsPool[currEntry.m_col];
        const LocationRotationLabelsVector locationRotationLabelsVec = 
            GetAllLocationRotationLabelsForLocation(Utilities::RowColInfo(locationRotationLabel->m_row, locationRotationLabel->m_column));

        for (const LocationRotationLabel* currLocationRotationLabel : locationRotationLabelsVec)
            anchoredColsSet.insert(currLocationRotationLabel->m_index);
    }

    //2. Find max non anchored entry
    const Eigen::MatrixXd labelingMatrix = in_labeling.GetLabelingMatrix();
    double currMax = -1;
    RowColInfo maxEntry = RowColInfo::m_invalidRowColInfo;
    for (int32_t i = 0; i < labelingMatrix.rows(); ++i)
    {
        if (!Utilities::IsItemInSet(i, in_anchoringData.m_anchoredRowsSet))
        {
            for (int32_t j = 0; j < labelingMatrix.cols(); ++j)
            {
                if (!Utilities::IsItemInSet(j, anchoredColsSet))
                {
                    const double currVal = labelingMatrix(i, j); 

                    if (currVal > currMax)
                    {
                        const RowColInfo currEntry(i, j);

                        if (VerifyAnchoringLegality(currEntry, in_anchoringData))
                        {
                            currMax = currVal;
                            maxEntry = currEntry;
                        }
                    }
                }
            }
        }
    }

    Utilities::LogAndAbortIf(currMax == -1, "In 'RelaxationLabelingSolverBase::GetMaxEntryInNonAnchoredRow()': currMax == -1");

    return maxEntry;
}

/************************************************************************/
bool PuzzleRL_Solver::VerifyAnchoringLegality(const RowColInfo& in_newEntryToBeAnchored, const AnchoringData& in_anchoringData) const
/************************************************************************/
{
    if (in_anchoringData.m_anchoredEntriesSet.empty())
        return true;

    const LocationRotationLabel* newLabel = m_locationRotationLabelsPool[in_newEntryToBeAnchored.m_col];
    const RowColInfo newLocationInPuzzle(newLabel->m_row, newLabel->m_column);

    bool foundNeighbor = false;
    for (const RowColInfo& currAnchoredEntry : in_anchoringData.m_anchoredEntriesSet)
    {
        const LocationRotationLabel* anchoredLabel = m_locationRotationLabelsPool[currAnchoredEntry.m_col];
        const RowColInfo anchoredLocationInPuzzle(anchoredLabel->m_row, anchoredLabel->m_column);

        if (RowColInfo::AreNeighbors(newLocationInPuzzle, anchoredLocationInPuzzle))
        {
            foundNeighbor = true;
            break;
        }
    }

    if (!foundNeighbor)
    {
        Utilities::Log("Candidate anchored entry " + in_newEntryToBeAnchored.GetString() + " (represents piece " + 
            m_puzzlePieceObjectsPool[in_newEntryToBeAnchored.m_row]->GetString() + " and location " + newLabel->GetString() + 
            ") does not have already anchored neighbor");

        return false;
    }

    return true;
}

/************************************************************************/
void PuzzleRL_Solver::DeleteObjectsPool()
/************************************************************************/
{
    for (const PuzzlePieceObject* puzzlePieceObject : m_puzzlePieceObjectsPool)
        delete puzzlePieceObject;

    m_puzzlePieceObjectsPool.clear();
}

/************************************************************************/
void PuzzleRL_Solver::DeleteLabelsPool()
/************************************************************************/
{
    for (const LocationRotationLabel* locationRotationLabel : m_locationRotationLabelsPool)
        delete locationRotationLabel;

    m_locationRotationLabelsPool.clear();
}

/************************************************************************/
void PuzzleRL_Solver::DeletePieceObjectsAndRotationsObjects()
/************************************************************************/
{
    for (const PieceObjectAndRot* pieceObjectAndRot : m_rotatedPiecesPool)
        delete pieceObjectAndRot;

    m_rotatedPiecesPool.clear();

    for (const PieceObjectAndRot* pieceObjectAndRot : m_rotatedPiecesExtraPool)
        delete pieceObjectAndRot;

    m_rotatedPiecesExtraPool.clear();
}

/************************************************************************/
void PuzzleRL_Solver::InitAllObjectsPool()
//Function input: none
//Function output: none
//Function objective: initiate all objects
/************************************************************************/
{
    //1. To be on the safe side, delete old objects pool
    DeleteObjectsPool();

    //2. Set m_puzzlePieceObjectsPool
    const PiecesInfoVector piecesInfoVec = m_puzzleImage.GetPiecesInfoVectorFromImage();
    for (const PieceInfo& pieceInfo : piecesInfoVec)
    {
        m_puzzlePieceObjectsPool.push_back(new PuzzlePieceObject(pieceInfo.m_pieceNumber, pieceInfo.m_piece));
    }

    //4. Output piece if needed
    if constexpr (OUTPUT_ALL_PIECES_SEPARATELY)
        m_puzzleRL_SolverOutputManager.OutputAllPiecesSeparately();
}

/************************************************************************/
void PuzzleRL_Solver::InitAllLabelsPool()
//Function input: none
//Function output: none
//Function objective: initiate all labels
/************************************************************************/
{
    //1. To be on the safe side, delete old labels pool
    DeleteLabelsPool();

#if IS_SAVE_ALL_PAIRS_SUPPORT_COMPUTATION_METHOD()
    m_neighborsLabelsMatrix = LabelVectorsMatrix::Constant(m_puzzleImage.GetNumOfRowPieces(), m_puzzleImage.GetNumOfColPieces(), LabelsVector());
#endif

    m_locationRotationLabelsMatrix = 
        LocationRotationLabelsVectorMatrix::Constant(m_puzzleImage.GetNumOfRowPieces(), m_puzzleImage.GetNumOfColPieces(), LocationRotationLabelsVector());

    //2. Set puzzle type to 'LocationRotationLabel'
    LocationRotationLabel::SetPuzzleType(m_config.GetPuzzleType());

    //3. Set m_locationRotationLabelsPool
    for (int32_t row = 0; row < m_puzzleImage.GetNumOfRowPieces(); ++row)
    {
        for (int32_t col = 0; col < m_puzzleImage.GetNumOfColPieces(); ++col)
        {
            LabelsVector labelsVec;
            LocationRotationLabelsVector locationRotationLabelsVec;

            for (ImageRotation rot = ImageRotation::e_0_degrees; rot < m_numOfPossiblePieceRotations; rot = static_cast<ImageRotation>(rot + 1))
            {
                const LocationRotationLabel* locationRotationLabel = new LocationRotationLabel(row, col, rot);
                m_locationRotationLabelsPool.push_back(locationRotationLabel);
                labelsVec.push_back(locationRotationLabel);
                locationRotationLabelsVec.push_back(locationRotationLabel);
            }

            m_locationRotationLabelsMatrix(row, col) = locationRotationLabelsVec;

            #if IS_SAVE_ALL_PAIRS_SUPPORT_COMPUTATION_METHOD()
                UpdateNeighborLabelsMatrix(row, col, labelsVec);
            #endif
        }
    }
}

/************************************************************************/
void PuzzleRL_Solver::InitPieceAndRotationsData()
//Function input: none
//Function output: none
//Function objective: initiate data related to handling all the combinations of pieces and possible rotations 
/************************************************************************/
{
    //1. To be on the safe side, delete vector
    DeletePieceObjectsAndRotationsObjects();

    //2. Create m_rotatedPieceIndexesMatrix and reserve memory in m_rotatedPiecesPool
    m_rotatedPieceIndexesMatrix = Eigen::MatrixXi::Constant(m_puzzlePieceObjectsPool.size(), m_numOfPossiblePieceRotations, -1);
    m_rotatedPiecesPool.reserve(m_rotatedPieceIndexesMatrix.size());

    //3. Fill m_rotatedPieceIndexesMatrix and m_rotatedPiecesPool
    int32_t currIndex = 0;
    for (const PuzzlePieceObject* puzzlePieceObj : m_puzzlePieceObjectsPool)
    {
        for (ImageRotation rot = ImageRotation::e_0_degrees; rot < m_numOfPossiblePieceRotations; rot = static_cast<ImageRotation>(rot + 1))
        {
            m_rotatedPieceIndexesMatrix(puzzlePieceObj->m_index, rot) = currIndex;
            m_rotatedPiecesPool.push_back(new PieceObjectAndRot(puzzlePieceObj, rot, currIndex));
            ++currIndex;
        }
    }

    //4. If it's type-1 puzzle, fill 'm_rotatedPiecesExtraPool' with pieces rotated by 90 degrees
    if (PuzzleType::eType1_NoRotation == m_config.GetPuzzleType())
    {
        for (const PuzzlePieceObject* puzzlePieceObj : m_puzzlePieceObjectsPool)
        {
            m_rotatedPiecesExtraPool.push_back(new PieceObjectAndRot(puzzlePieceObj, ImageRotation::e_90_degrees, -1));
        }
    }

    //5. Check for constant pieces
    CheckAndSetConstantPieces();
}

/************************************************************************/
void PuzzleRL_Solver::InitSolverCompatibility()
//Function input: none
//Function output: none
//Function objective: initiate all data relating to solver compatibility
/************************************************************************/
{
    ComputePieceDissimilarities();
    ComputePieceCompatibilitiesBasedOnDissimilarities();

    //We save 'm_piecesCompatibilitiesNonManipulated' in case of constant pieces to compute the "real alc" values
    //for translation branching decisions
    if (!m_constantPiecesVector.empty())
    {
        m_piecesCompatibilitiesNonManipulated = m_piecesCompatibilities;
        ApplyNormalizationAndSymmetrizationAndThresholding(m_piecesCompatibilitiesNonManipulated);

        ApplyConstantPiecesLogicToCompatibility(m_piecesCompatibilities);
    }

    ApplyNormalizationAndSymmetrizationAndThresholding(m_piecesCompatibilities);
    ApplyThresholding(m_piecesCompatibilities);

    //For now, we want to ensure that our piece compatibility measure is symmetric
    //VerifyCompatibilitySymmetry();
}

/************************************************************************/
double PuzzleRL_Solver::GetSolverCompatibilityValue(const PuzzlePieceObject* in_firstPiece, const LocationRotationLabel* in_firstLocRot,
    const PuzzlePieceObject* in_secondPiece, const LocationRotationLabel* in_secondLocRot) const
//Function input: two piece objects i and j, and two location-rotation labels lambda and miu
//Function output: value of rij(lambda, miu)
//Function objective: described in "Function output"
/************************************************************************/
{
    double val = 0.0;

    //If we have two identical objects
    if (*in_firstPiece == *in_secondPiece)
    {
        if (*in_firstLocRot == *in_secondLocRot)
            val = 0.0f;
        //Same piece in different locations, or same piece in the same location but in different rotations
        else
            val = 0.0f;
    }
    else
    {
        //If it's a different piece for the same locations
        if (in_firstLocRot->m_row == in_secondLocRot->m_row && in_firstLocRot->m_column == in_secondLocRot->m_column)
        {
            val = 0.0f;
        }
        //If the two objects and the two labels are not identical
        else if (in_firstLocRot->IsNeighborOf(*in_secondLocRot))
        {
            Orientation pieceOrientation = Orientation::eInvalidOrientation;

            if (in_secondLocRot->IsTopNeighborOf(*in_firstLocRot))
                pieceOrientation = Orientation::eUp;
            else if (in_secondLocRot->IsBottomNeighborOf(*in_firstLocRot))
                pieceOrientation = Orientation::eDown;
            else if (in_secondLocRot->IsLeftNeighborOf(*in_firstLocRot))
                pieceOrientation = Orientation::eLeft;
            else if (in_secondLocRot->IsRightNeighborOf(*in_firstLocRot))
                pieceOrientation = Orientation::eRight;
            else
                Utilities::LogAndAbort("");

            val = m_piecesCompatibilities(GetRotatedPieceIndex(in_firstPiece, in_firstLocRot->m_rotation),
                GetRotatedPieceIndex(in_secondPiece, in_secondLocRot->m_rotation), pieceOrientation);
        }
    }

    return val;
}

/************************************************************************/
double PuzzleRL_Solver::GetSolverCompatibilityValueWithPiecesComp(const PuzzlePieceObject* in_firstPiece, const LocationRotationLabel* in_firstLocRot,
    const PuzzlePieceObject* in_secondPiece, const LocationRotationLabel* in_secondLocRot, const PiecesTensorType& in_piecesComp) const
//Function input: two piece objects i and j, two location-rotation labels lambda and miu, and pieces comp 'in_piecesComp'
//Function output: value of rij(lambda, miu)
//Function objective: described in "Function output"
/************************************************************************/
{
    double val = 0.0;

    //If we have two identical objects
    if (*in_firstPiece == *in_secondPiece)
    {
        if (*in_firstLocRot == *in_secondLocRot)
            val = 0.0f;
        //Same piece in different locations, or same piece in the same location but in different rotations
        else
            val = 0.0f;
    }
    else
    {
        //If it's a different piece for the same locations
        if (in_firstLocRot->m_row == in_secondLocRot->m_row && in_firstLocRot->m_column == in_secondLocRot->m_column)
        {
            val = 0.0f;
        }
        //If the two objects and the two labels are not identical
        else if (in_firstLocRot->IsNeighborOf(*in_secondLocRot))
        {
            Orientation pieceOrientation = Orientation::eInvalidOrientation;

            if (in_secondLocRot->IsTopNeighborOf(*in_firstLocRot))
                pieceOrientation = Orientation::eUp;
            else if (in_secondLocRot->IsBottomNeighborOf(*in_firstLocRot))
                pieceOrientation = Orientation::eDown;
            else if (in_secondLocRot->IsLeftNeighborOf(*in_firstLocRot))
                pieceOrientation = Orientation::eLeft;
            else if (in_secondLocRot->IsRightNeighborOf(*in_firstLocRot))
                pieceOrientation = Orientation::eRight;
            else
                Utilities::LogAndAbort("");

            val = in_piecesComp(GetRotatedPieceIndex(in_firstPiece, in_firstLocRot->m_rotation),
                GetRotatedPieceIndex(in_secondPiece, in_secondLocRot->m_rotation), pieceOrientation);
        }
    }

    return val;
}

/************************************************************************/
void PuzzleRL_Solver::ComputePieceDissimilarities()
//Function input: none
//Function output: none
//Function objective: compute all pieces dissimilarities for real solver
/************************************************************************/
{
    //1. Calc Dissimilarities
    ComputePieceDissimilaritiesForDissimilarityType(DISSIMILARITY_TYPE);
    SetDissimilarityInfo(DISSIMILARITY_TYPE);
}

/************************************************************************/
void PuzzleRL_Solver::ComputePieceDissimilaritiesForDissimilarityType(const DissimilarityType in_dissimilarityType)
//Function input: none
//Function output: none
//Function objective: compute pieces dissimilarities for 'in_dissimilarityType' dissimilarity type
/************************************************************************/
{
    //2. Choose dissimilarity function
    const PieceDissimilarityFuncType dissimilarityFunc = GetDissimilarityFunction(in_dissimilarityType); 

    //3. Allocate pieceDissimilaritiesTensor
    const int32_t numOfRotatedPieces = GetNumOfRotatedPieces();
    m_piecesDissimilarities = PiecesTensorType(numOfRotatedPieces, numOfRotatedPieces, Orientation::eNumOfPossibleOrientations);
    m_piecesDissimilarities.setConstant(-1);

    //4. Init min and max values, and compute non-normalized Dissimilarities
    std::atomic<double> minDissimilarity = std::numeric_limits<double>::max();
    std::atomic<double> maxDissimilarity = std::numeric_limits<double>::lowest();

    //5. Set computeDissimilaritiesFunction, which gets two rotated pieces and orientation and returns their dissimilarity,
    //and apply it on all values
    const PieceRelationComputationFuncType computeDissimilaritiesFunction = 
        [&](const PieceObjectAndRot* in_firstRotatedPiece, const PieceObjectAndRot* in_secondRotatedPiece, const Orientation in_orien)
        {
            //a.1. Get the dissimilarity value
            const double val = dissimilarityFunc(this, in_firstRotatedPiece, in_secondRotatedPiece, in_orien);
            Utilities::LogAndAbortIf(val < 0, "piece dissimilarity value is smaller than 0, for " + 
                m_puzzleRL_SolverOutputManager.GetPiecesOrientationStr(in_firstRotatedPiece, in_secondRotatedPiece, in_orien));

            //a.2. Update min and max dissimilarities
            Utilities::UpdateAtomicMinimum(minDissimilarity, val);
            Utilities::UpdateAtomicMaximum(maxDissimilarity, val);

            //a.3. return value
            return val;
        };

    ApplyFunctionForAllRotatedPiecesRelations(computeDissimilaritiesFunction, m_piecesDissimilarities, true);

    //6. Set normalizeDissimilaritiesFunction, which gets two rotated pieces and orientation and returns their normalized dissimilarity,
    //and apply it on all values
    const PieceRelationComputationFuncType normalizeDissimilaritiesFunction = 
        [&](const PieceObjectAndRot* in_firstRotatedPiece, const PieceObjectAndRot* in_secondRotatedPiece, const Orientation in_orien)
    {
        const double dissimilarity = m_piecesDissimilarities(in_firstRotatedPiece->m_index, in_secondRotatedPiece->m_index, in_orien);
        const double normalizedDissimilarity = (dissimilarity - minDissimilarity) / (maxDissimilarity - minDissimilarity);

        Utilities::LogAndAbortIf(normalizedDissimilarity < 0, "normalized piece dissimilarity value is smaller than 0");

        return normalizedDissimilarity;
    };
}

/************************************************************************/
void PuzzleRL_Solver::ComputePieceCompatibilitiesBasedOnDissimilarities()
//Function input: none
//Function output: none
//Function objective: compute all pieces compatibilities for real solver
/************************************************************************/
{
    //1. Compute Compatibilities
    ComputePieceCompatibilitiesBasedOnDissimilaritiesForDissimilarityType(DISSIMILARITY_TYPE);
}

/************************************************************************/
void PuzzleRL_Solver::ComputePieceCompatibilitiesBasedOnDissimilaritiesForDissimilarityType(const DissimilarityType in_dissimilarityType)
//Function input: none
//Function output: none
//Function objective: compute pieces Compatibilities based on 'in_dissimilarityType' dissimilarity type
/************************************************************************/
{
    //1. Allocate tensor
    const int32_t numOfRotatedPieces = GetNumOfRotatedPieces();
    m_piecesCompatibilities = PiecesTensorType(numOfRotatedPieces, numOfRotatedPieces, Orientation::eNumOfPossibleOrientations);
    m_piecesCompatibilities.setConstant(-1);

    //2. Compute comp
    ComputePieceCompatibilitiesBasedOnDissimilaritiesForDissimilarityType(m_piecesCompatibilities, m_piecesDissimilarities,
        in_dissimilarityType, m_kParamForMethod2CompatibilityComputation);
}                                                                        

/************************************************************************/
void PuzzleRL_Solver::ComputePieceCompatibilitiesBasedOnDissimilaritiesForDissimilarityType(
    PiecesTensorType& out_pieceCompatibilitiesTensor, const PiecesTensorType& in_pieceDissimilaritiesTensor, const DissimilarityType in_dissimilarityType,
    const double& in_kappaCoeff) const
/************************************************************************/
{
    int32_t kappaValue = std::max(2, static_cast<int32_t>(static_cast<double>(m_puzzlePieceObjectsPool.size()) * in_kappaCoeff));
    if (PuzzleType::eType2_UknownRotation == m_config.GetPuzzleType())
    {
        kappaValue *= 2;
    }

    std::atomic<double> minCompatibility = std::numeric_limits<double>::max();
    std::atomic<double> maxCompatibility = std::numeric_limits<double>::lowest();

    for (Orientation orien = Orientation::eUp; orien < Orientation::eNumOfPossibleOrientations; orien = static_cast<Orientation>(orien + 1))
    {
        for (const PieceObjectAndRot* firstRotatedPiece: m_rotatedPiecesPool)
        {
            //1. Get set of dissimilarity values
            std::multiset<double> set;

            for (const PieceObjectAndRot* secondRotatedPiece: m_rotatedPiecesPool)
            {
                if (!PieceObjectAndRot::ArePieceObjectsIdentical(*firstRotatedPiece, *secondRotatedPiece))
                {
                    const double currDissValue = in_pieceDissimilaritiesTensor(firstRotatedPiece->m_index, secondRotatedPiece->m_index, orien);
                    set.insert(currDissValue);
                }
            }

            //2. Get average within first k dissimilarities
            int32_t count = 1;
            double alpha_thDissimilarity = -1.0;
            double dissimilarityValuesSum = 0;

            for (const double& currDissimilarityVal : set)
            {
                dissimilarityValuesSum += currDissimilarityVal;

                if (count == kappaValue)
                {
                    alpha_thDissimilarity = (dissimilarityValuesSum / count);
                    break;
                }

                ++count;
            }
            Utilities::LogAndAbortIf(alpha_thDissimilarity == -1.0, "kthDissimilarity == -1.0");

            //3. Set values by 'kthDissimilarity'
            for (const PieceObjectAndRot* secondRotatedPiece: m_rotatedPiecesPool)
            {
                if (!PieceObjectAndRot::ArePieceObjectsIdentical(*firstRotatedPiece, *secondRotatedPiece))
                {
                    const int32_t currRank = GetOneBasedIndexInDissimilaritiesOrder(firstRotatedPiece,
                        orien, secondRotatedPiece, in_dissimilarityType);

                    //If should zero value
                    double val = 0.0;

                    const double currDissValue = in_pieceDissimilaritiesTensor(firstRotatedPiece->m_index, secondRotatedPiece->m_index, orien);
                    if (0.0 == alpha_thDissimilarity && 0.0 == currDissValue)
                    {
                        val = 1.0;
                    }
                    else if (currDissValue <= alpha_thDissimilarity)
                    {
                        const double naiveCompValue = 1.0 - currDissValue / alpha_thDissimilarity;
                        val = std::pow(naiveCompValue, currRank);
                    }
                    else
                    {
                        val = 0.0;
                    }

                    Utilities::LogAndAbortIf(std::isnan(val) || val < 0, "std::isnan(val) || val < 0");

                    Utilities::UpdateAtomicMinimum(minCompatibility, val);
                    Utilities::UpdateAtomicMaximum(maxCompatibility, val);

                    out_pieceCompatibilitiesTensor(firstRotatedPiece->m_index, secondRotatedPiece->m_index, orien) = val;
                }
            }
        }
    }
}

/************************************************************************/
void PuzzleRL_Solver::ApplyNormalizationAndSymmetrizationAndThresholding(PiecesTensorType& inout_pieceCompatibilitiesTensor) const
/************************************************************************/
{
    ApplyNormalizationAndSymmetrization(inout_pieceCompatibilitiesTensor);
    ApplyThresholding(inout_pieceCompatibilitiesTensor);
}

/************************************************************************/
void PuzzleRL_Solver::ApplyNormalizationAndSymmetrization(PiecesTensorType& inout_pieceCompatibilitiesTensor) const
/************************************************************************/
{
    std::atomic<double> minCompatibility = std::numeric_limits<double>::max();
    std::atomic<double> maxCompatibility = std::numeric_limits<double>::lowest();

    const PieceRelationVerifcationFuncType findMinMaxCompatibilities = 
       [&](const PieceObjectAndRot* in_firstRotatedPiece, const PieceObjectAndRot* in_secondRotatedPiece, const Orientation in_orien)
        {
            const double val = inout_pieceCompatibilitiesTensor(in_firstRotatedPiece->m_index, in_secondRotatedPiece->m_index, in_orien);
            Utilities::UpdateAtomicMinimum(minCompatibility, val);
            Utilities::UpdateAtomicMaximum(maxCompatibility, val);
        };

    ApplyVerificationFunctionForAllRotatedPiecesRelations(findMinMaxCompatibilities, false);

    ApplyNormalizationAndSymmetrization(inout_pieceCompatibilitiesTensor, minCompatibility, maxCompatibility);
}

/************************************************************************/
void PuzzleRL_Solver::ApplyNormalizationAndSymmetrization(PiecesTensorType& inout_pieceCompatibilitiesTensor, 
    const double& in_minCompatibility, const double& in_maxCompatibility) const
/************************************************************************/
{
    //Set normalizeCompatibilitiesFunction, which gets two rotated pieces and orientation and returns their normalized and symmetrized compatibility score,
    //and apply it on all values
    const PieceRelationComputationFuncType normalizeFunc = 
        [&](const PieceObjectAndRot* in_firstRotatedPiece, const PieceObjectAndRot* in_secondRotatedPiece, const Orientation in_orien)
    {
        const double compVal = inout_pieceCompatibilitiesTensor(in_firstRotatedPiece->m_index, in_secondRotatedPiece->m_index, in_orien);
        const double normalizedComp = (compVal - in_minCompatibility) / (in_maxCompatibility - in_minCompatibility);
        Utilities::LogAndAbortIf(normalizedComp < 0, "normalizedComp is smaller than 0");
        return normalizedComp;
    };

    const PieceRelationComputationFuncType normalizeCompatibilitiesFunction = 
       [&](const PieceObjectAndRot* in_firstRotatedPiece, const PieceObjectAndRot* in_secondRotatedPiece, const Orientation in_orien)
        {
            constexpr bool APPLY_MAX_MIN_NORMALIZATION = true;

            const Orientation oppositeOrien = RLSolverGeneralUtils::GetOppositeOrientation(in_orien);

            double compVal1 = normalizeFunc(in_firstRotatedPiece, in_secondRotatedPiece, in_orien);
            double compVal2 = normalizeFunc(in_secondRotatedPiece, in_firstRotatedPiece, oppositeOrien);

            const double compAverage = (compVal1 + compVal2) / 2;

            return compAverage;
        };

    Utilities::Log("min comp val before symmetrization is: " + std::to_string(in_minCompatibility));
    Utilities::Log("max comp val before symmetrization is: " + std::to_string(in_maxCompatibility));

    ApplyFunctionForAllRotatedPiecesRelations(normalizeCompatibilitiesFunction, inout_pieceCompatibilitiesTensor, true);
}

/************************************************************************/
void PuzzleRL_Solver::ApplyThresholding(PiecesTensorType& inout_pieceCompatibilitiesTensor) const
/************************************************************************/
{
    if (m_config.WasMinimumThresholdDefined())
    {
        const double minThreshold = m_config.GetMinimumThresholdForPieceCompatibility();

        const PieceRelationComputationFuncType thresholdCompFunction = 
            [&](const PieceObjectAndRot* in_firstRotatedPiece, const PieceObjectAndRot* in_secondRotatedPiece, const Orientation in_orien)
            {
                double val = inout_pieceCompatibilitiesTensor(in_firstRotatedPiece->m_index, in_secondRotatedPiece->m_index, in_orien);
                if (val < minThreshold)
                {
                    val = 0;
                }

                return val;
            };

        ApplyFunctionForAllRotatedPiecesRelations(thresholdCompFunction, inout_pieceCompatibilitiesTensor, true);
    }
}

/************************************************************************/
void PuzzleRL_Solver::VerifyCompatibilitySymmetry() const
//Function input: none
//Function output: none
//Function objective: Verify that our piece compatibility measure is symmetric (we just check 'm_piecesCompatibilities')
/************************************************************************/
{
    const PieceRelationVerifcationFuncType verifySymmetryFunction = 
       [&](const PieceObjectAndRot* in_firstRotatedPiece, const PieceObjectAndRot* in_secondRotatedPiece, const Orientation in_orien)
        {
            const Orientation oppositeOrien = RLSolverGeneralUtils::GetOppositeOrientation(in_orien);
            const double firstVal = m_piecesCompatibilities(in_firstRotatedPiece->m_index, in_secondRotatedPiece->m_index, in_orien);
            const double secondVal = m_piecesCompatibilities(in_secondRotatedPiece->m_index, in_firstRotatedPiece->m_index, oppositeOrien);

            Utilities::LogAndAbortIf(firstVal != secondVal, "In 'PuzzleRL_Solver::VerifyCompatibilitySymmetry()':firstVal != secondVal");
        };

    ApplyVerificationFunctionForAllRotatedPiecesRelations(verifySymmetryFunction, true);
}

/************************************************************************/
void PuzzleRL_Solver::ApplyFunctionForAllRotatedPiecesRelations(const PieceRelationComputationFuncType& in_function, 
    PiecesTensorType& inout_piecesTensor, const bool in_areValuesSymmetric) const
//Function input: in_function: function to be applied; inout_piecesTensor: pieces tensor to set the values in;
//  in_areValuesSymmetric: boolean indicating whether relations are symmetric
//Function output: none
//Function objective: apply 'in_function' for all possible "unique" combinations of rotated pieces
/************************************************************************/
{
    //We iterate all the combinations of non-rotated pieces (that's 'in_firstPuzzlePiece') and rotated pieces.
    //This is enough to set all values needed

    std::for_each(executionPolicy, m_puzzlePieceObjectsPool.begin(), m_puzzlePieceObjectsPool.end(),
        [&](const PuzzlePieceObject* in_firstPuzzlePiece) -> void
        {
            ApplyFunctionForAllRotatedPiecesRelations(in_firstPuzzlePiece, in_function, inout_piecesTensor, in_areValuesSymmetric);
        }
    );
}

/************************************************************************/
void PuzzleRL_Solver::ApplyFunctionForAllRotatedPiecesRelationsNonParallel(const PieceRelationComputationFuncType& in_function, 
    PiecesTensorType& inout_piecesTensor, const bool in_areValuesSymmetric)
//Function input: in_function: function to be applied; inout_piecesTensor: pieces tensor to set the values in;
//  in_areValuesSymmetric: boolean indicating whether relations are symmetric
//Function output: none
//Function objective: apply 'in_function' for all possible "unique" combinations of rotated pieces not in parallel
    //(this function may be used for debugging)
/************************************************************************/
{
    //We iterate all the combinations of non-rotated pieces (that's 'in_firstPuzzlePiece') and rotated pieces.
    //This is enough to set all values needed

    std::for_each(std::execution::seq, m_puzzlePieceObjectsPool.begin(), m_puzzlePieceObjectsPool.end(),
        [&](const PuzzlePieceObject* in_firstPuzzlePiece) -> void
        {
            ApplyFunctionForAllRotatedPiecesRelations(in_firstPuzzlePiece, in_function, inout_piecesTensor, in_areValuesSymmetric);
        }
    );
}

/************************************************************************/
void PuzzleRL_Solver::ApplyFunctionForAllRotatedPiecesRelations(const PuzzlePieceObject* in_puzzlePiece, const PieceRelationComputationFuncType& in_function, 
    PiecesTensorType& inout_piecesTensor, const bool in_areValuesSymmetric) const
//Function input: in_puzzlePiece: puzzle pieces in_function: function to be applied; inout_piecesTensor: pieces tensor to set the values in;
//  in_areValuesSymmetric: boolean indicating whether relations are symmetric
//Function output: none
//Function objective: apply 'in_function' for all possible "unique" combinations of rotated pieces
/************************************************************************/
{
    //We iterate all the combinations rotated pieces (that's secondRotatedPiece).
    //This is enough to set all values needed (see 'SetValuesToPiecesTensor' for explanation)

    for (const PieceObjectAndRot* secondRotatedPiece : m_rotatedPiecesPool)
    {
        if (*in_puzzlePiece == *secondRotatedPiece->m_pieceObject)
        {
            //We can break here for the symmetric case - this saves us half of the iterations
            if (in_areValuesSymmetric)
                break;
            //For the non-symmetric case, we'll just skip the iteration where both pieces are identical
            else
                continue;
        }

        const PieceObjectAndRot* firstPieceRotatedWith0Degrees = 
            GetRotatedPieceFromPieceNumberAndRotation(in_puzzlePiece->m_pieceNumber, ImageRotation::e_0_degrees);

        for (Orientation orien = Orientation::eUp; orien < Orientation::eNumOfPossibleOrientations; orien = static_cast<Orientation>(orien + 1))
        {
            //Get value from in_function, and call 'SetValuesToPiecesTensor()' to set it in 'inout_piecesTensor'
            const double value = in_function(firstPieceRotatedWith0Degrees, secondRotatedPiece, orien);
            SetValuesToPiecesTensor(firstPieceRotatedWith0Degrees, secondRotatedPiece, orien, value, inout_piecesTensor, in_areValuesSymmetric);
        }
    }
}

/************************************************************************/
void PuzzleRL_Solver::ApplyVerificationFunctionForAllRotatedPiecesRelations(const PieceRelationVerifcationFuncType& in_function, const bool in_areValuesSymmetric) const
//Function input: in_function: verification function to be applied; in_areValuesSymmetric: boolean indicating whether relations are symmetric
//Function output: none
//Function objective: apply 'in_function' for all possible "unique" combinations of rotated pieces
/************************************************************************/
{
    //We iterate all the combinations of non-rotated pieces (that's 'firstPuzzlePiece') and rotated pieces (that's secondRotatedPiece).
    //This is enough to get all values needed (see 'SetValuesToPiecesTensor' for explanation)

    std::for_each(executionPolicy, m_puzzlePieceObjectsPool.begin(), m_puzzlePieceObjectsPool.end(),
        [&](const PuzzlePieceObject* firstPuzzlePiece) -> void
        {
            for (const PieceObjectAndRot* secondRotatedPiece : m_rotatedPiecesPool)
            {
                if (*firstPuzzlePiece == *secondRotatedPiece->m_pieceObject)
                {
                    //We can break here for the symmetric case - this saves us half of the iterations
                    if (in_areValuesSymmetric)
                        break;
                    //For the non-symmetric case, we'll just skip the iteration where both pieces are identical
                    else
                        continue;
                }

                const PieceObjectAndRot* firstPieceRotatedWith0Degrees = 
                    GetRotatedPieceFromPieceNumberAndRotation(firstPuzzlePiece->m_pieceNumber, ImageRotation::e_0_degrees);

                for (Orientation orien = Orientation::eUp; orien < Orientation::eNumOfPossibleOrientations; orien = static_cast<Orientation>(orien + 1))
                    in_function(firstPieceRotatedWith0Degrees, secondRotatedPiece, orien);
            }
        }
    );
}

/************************************************************************/
void PuzzleRL_Solver::SetValuesToPiecesTensor(const PieceObjectAndRot* in_firstRotatedPiece, const PieceObjectAndRot* in_secondRotatedPiece, 
    const Orientation in_orien, const double in_val, PiecesTensorType& inout_piecesTensor, const bool in_isSymmetricRelation) const
//Function input: in_firstRotatedPiece: first rotated piece; in_secondRotatedPiece: second rotated piece; in_orien: orientation between pieces;
//  in_val: value to be set to tensor; inout_piecesTensor: tensor to set the value to; in_isSymmetricRelation: boolean indicating whether relation is symmetric
//Function output: none
//Function objective: set 'in_val' to the entry of (in_firstRotatedPiece, in_secondRotatedPiece, in_orien) in 'inout_piecesTensor',
//  and for type 2, also set the entry "rotated entries" with in_val too
/************************************************************************/
{
    //For all possible rotations (just one rotation for type-1 puzzles, and four rotation for type-2 puzzles), we set 'in_val' in inout_piecesTensor

    //For example, if (xi_0_deg, xj_90_deg, right) equals 'in_val', then there are three more entries that should get the value 'in_val':
    //a. (xi_90_deg, xj_180_deg, down)
    //b. (xi_180_deg, xj_270_deg, left)
    //c. (xi_270_deg, xj_0_deg, up)

    for (int32_t i = 0; i < m_numOfPossiblePieceRotations; ++i)
    {
        //1. Compute current orientation
        const Orientation currOrien = static_cast<Orientation>((static_cast<int32_t>(in_orien) + i) % ImageRotation::eNumOfImageRotations);

        //2. Compute rotation to apply to pieces, and rotate and get pieces
        const ImageRotation rotationToApply = static_cast<ImageRotation>(i);
        const PieceObjectAndRot* currFirstRotatedPiece = RotateAndGetPiece(in_firstRotatedPiece, rotationToApply);
        const PieceObjectAndRot* currSecondRotatedPiece = RotateAndGetPiece(in_secondRotatedPiece, rotationToApply);

        //3. Set 'in_val' in inout_piecesTensor
        inout_piecesTensor(currFirstRotatedPiece->m_index, currSecondRotatedPiece->m_index, currOrien) = in_val;

        //4. If the relation is symmetric - set 'in_val' to the symmetric entry
        if (in_isSymmetricRelation)
        {
            const Orientation oppositeOrien = RLSolverGeneralUtils::GetOppositeOrientation(currOrien);
            inout_piecesTensor(currSecondRotatedPiece->m_index, currFirstRotatedPiece->m_index, oppositeOrien) = in_val;
        }
    }
}

/************************************************************************/
double PuzzleRL_Solver::GetImprovedMGC_Dissimilarity(const PieceObjectAndRot* in_xi, const PieceObjectAndRot* in_xj, const Orientation in_orien) const
/************************************************************************/
{
    //'in_xj' should be in orientation 'in_orien' to 'in_xi' (for example, if in_orien=eRight, then in_xj is right to in_xi)

    //1. Set 'leftPiece' and 'rightPiece' (to simplify things so we always have left-right relation)
    const PieceObjectAndRot* leftPiece = nullptr;
    const PieceObjectAndRot* rightPiece = nullptr;

    switch (in_orien)
    {
    case eDown:
        {
            const PieceObjectAndRot* xi_RotatedBy90 = RotateAndGetPiece(in_xi, ImageRotation::e_90_degrees, true);
            const PieceObjectAndRot* xj_RotatedBy90 = RotateAndGetPiece(in_xj, ImageRotation::e_90_degrees, true);
            leftPiece = xj_RotatedBy90;
            rightPiece = xi_RotatedBy90;
        }
        break;

    case eUp:
        {
            const PieceObjectAndRot* xi_RotatedBy90 = RotateAndGetPiece(in_xi, ImageRotation::e_90_degrees, true);
            const PieceObjectAndRot* xj_RotatedBy90 = RotateAndGetPiece(in_xj, ImageRotation::e_90_degrees, true);
            leftPiece = xi_RotatedBy90;
            rightPiece = xj_RotatedBy90;
        }
        break;

    case eRight:
        leftPiece = in_xi;
        rightPiece = in_xj;
        break;

    case eLeft:
        leftPiece = in_xj;
        rightPiece = in_xi;
        break;

    default:
        Utilities::LogAndAbort("No correct orientation.");
    }

    double dlr = 0;
    double drl = 0;
    GetDlrAndDrlForImprovedMGC(leftPiece, rightPiece, dlr, drl);

    //9. Compute dissimilarity by dlr and drl
    double dissimilarity = dlr + drl;

    //10. Add derivatives info to dissimilarity
    double dlrDerivative = 0;
    double drlDerivative = 0;
    GetDlrDerivativeAndDrlDerivativeForImprovedMGC(leftPiece, rightPiece, dlrDerivative, drlDerivative);

    dissimilarity += dlrDerivative + drlDerivative;

    Utilities::LogAndAbortIf(std::isnan(dissimilarity), "std::isnan(dissimilarity): " 
        + m_puzzleRL_SolverOutputManager.GetPiecesOrientationStr(in_xi, in_xj, in_orien));

    return dissimilarity;
}

/************************************************************************/
void PuzzleRL_Solver::GetDlrAndDrlForImprovedMGC(const PieceObjectAndRot* in_leftPiece, const PieceObjectAndRot* in_rightPiece, double& out_dlr, double& out_drl) const
/************************************************************************/
{
    //1. Set constants 
    constexpr int32_t numOfDummyGradients = 8;

    const int32_t numOfPixels = m_puzzleImage.GetPieceSize();
    const size_t sizeWithDummyGradients = numOfPixels + numOfDummyGradients;

    Eigen::Matrix<double, numOfDummyGradients, 3> dummyGradientMatrix;
    dummyGradientMatrix <<  0, 0, 0,
                            0, 0, 1,
                            0, 1, 0,
                            0, 1, 1,
                            1, 0, 0,
                            1, 0, 1,
                            1, 1, 0,
                            1, 1, 1;

    //2. Compute gij_lr, gij_rl 
    Eigen::MatrixXd gij_lr = Eigen::MatrixXd::Constant(numOfPixels, numOfChannels, 0);
    Eigen::MatrixXd gji_rl = Eigen::MatrixXd::Constant(numOfPixels, numOfChannels, 0);
    
    const std::vector<PixelType> leftEdge_ofRightPiece = in_rightPiece->GetLeftEdgePixels();
    const std::vector<PixelType> rightEdge_ofLeftPiece = in_leftPiece->GetRightEdgePixels();

    for (int32_t p = 0; p < numOfPixels; ++p) 
    {
        for (int32_t color = 0; color < numOfChannels; ++color) 
        {
            gij_lr(p, color) = leftEdge_ofRightPiece[p][color] - rightEdge_ofLeftPiece[p][color];
            gji_rl(p, color) = -gij_lr(p, color);
        }
    }

    //3. Compute g_il, g_jr (including dummy gradients)
    const std::vector<PixelType> secondRightEdge_ofLeftPiece = in_leftPiece->GetSecondRightEdge();
    const std::vector<PixelType> secondLeftEdge_ofRightPiece = in_rightPiece->GetSecondLeftEdge();

    Eigen::MatrixXd g_il = Eigen::MatrixXd::Constant(sizeWithDummyGradients, numOfChannels, 0);
    Eigen::MatrixXd g_jr = Eigen::MatrixXd::Constant(sizeWithDummyGradients, numOfChannels, 0);

    //3.1. Add real data to g_il, g_jr 
    for (int32_t p = 0; p < numOfPixels; ++p) 
    {
        for (int32_t color = 0; color < numOfChannels; ++color) 
        {
            g_il(p, color) = rightEdge_ofLeftPiece[p][color] - secondRightEdge_ofLeftPiece[p][color];
            g_jr(p, color) = leftEdge_ofRightPiece[p][color] - secondLeftEdge_ofRightPiece[p][color];
        }
    }

    //3.2. Add dummy gradients to g_il, g_jr 
    for (int32_t p = numOfPixels; p < sizeWithDummyGradients; ++p) 
    {
        const auto currDummyGradient = dummyGradientMatrix.row(p - numOfPixels);
        g_il.row(p) = currDummyGradient;
        g_jr.row(p) = currDummyGradient;
    }

    //4. Compute leftInverseCov and rightInverseCov
    const Eigen::MatrixXd leftInverseCov = Utilities::ComputeInverseOfCovarianceMatrix(g_il);
    const Eigen::MatrixXd rightInverseCov = Utilities::ComputeInverseOfCovarianceMatrix(g_jr);

    //5. Compute eij_lr, eji_rl (eij_lr and eji_rl are only the difference in dlr and drl computation comparing to the regular MGC)
    Eigen::MatrixXd eij_lr = Eigen::MatrixXd::Constant(numOfPixels, numOfChannels, 0);
    Eigen::MatrixXd eji_rl = Eigen::MatrixXd::Constant(numOfPixels, numOfChannels, 0);

    for (int32_t p = 0; p < numOfPixels; ++p) 
    {
        for (int32_t color = 0; color < numOfChannels; ++color) 
        {
            eij_lr(p, color) = 0.5 * ((rightEdge_ofLeftPiece[p][color] - secondRightEdge_ofLeftPiece[p][color]) + 
                (secondLeftEdge_ofRightPiece[p][color] - leftEdge_ofRightPiece[p][color]));

            eji_rl(p, color) = -eij_lr(p, color);
        }
    }

    //6. Compute dlr
    double dlr = 0;
    for (int32_t p = 0; p < numOfPixels; ++p) 
    {
        const Eigen::Vector3d currRemainderVec = gij_lr.row(p) - eij_lr.row(p); 
        const Eigen::MatrixXd currValueIn1x1Matrix = currRemainderVec.transpose() * leftInverseCov * currRemainderVec;
        const double currValue = currValueIn1x1Matrix.sum();

        dlr += currValue;
    }
    out_dlr = dlr;

    //7. Compute drl
    double drl = 0;
    for (int32_t p = 0; p < numOfPixels; ++p) 
    {
        const Eigen::Vector3d currRemainderVec = gji_rl.row(p) - eji_rl.row(p); 
        const Eigen::MatrixXd currValueIn1x1Matrix = currRemainderVec.transpose() * rightInverseCov * currRemainderVec;
        const double currValue = currValueIn1x1Matrix.sum();

        drl += currValue;
    }
    out_drl = drl;
}

/************************************************************************/
void PuzzleRL_Solver::GetDlrDerivativeAndDrlDerivativeForImprovedMGC(const PieceObjectAndRot* in_leftPiece, const PieceObjectAndRot* in_rightPiece, 
    double&out_dlrDerivative, double& out_drlDerivative) const
/************************************************************************/
{
    //1. Set constants 
    constexpr int32_t numOfDummyGradients = 8;

    const int32_t numOfPixels = m_puzzleImage.GetPieceSize();
    const int32_t numOfAlongsideDerivatives = numOfPixels - 1;
    const size_t sizeWithDummyGradients = numOfPixels + numOfDummyGradients;

    Eigen::Matrix<double, numOfDummyGradients, 3> dummyGradientMatrix;
    dummyGradientMatrix <<  0, 0, 0,
                            0, 0, 1,
                            0, 1, 0,
                            0, 1, 1,
                            1, 0, 0,
                            1, 0, 1,
                            1, 1, 0,
                            1, 1, 1;

    //2. Compute gij_lr_derivative, gji_rl_derivative
    Eigen::MatrixXd gij_lr_derivative = Eigen::MatrixXd::Constant(numOfAlongsideDerivatives, numOfChannels, 0);
    Eigen::MatrixXd gji_rl_derivative = Eigen::MatrixXd::Constant(numOfAlongsideDerivatives, numOfChannels, 0);

    for (int32_t p = 1; p < numOfPixels - 1; ++p) 
    {
        const int32_t currRowMatrix = p - 1;
        const ExtendedPixelType delta_j_p_1 = ImageUtils::GetImprovedMGC_Delta(in_rightPiece->GetRotatedPiece(), p, 0);
        const ExtendedPixelType delta_i_p_P = ImageUtils::GetImprovedMGC_Delta(in_leftPiece->GetRotatedPiece(), p, numOfPixels - 1);

        for (int32_t color = 0; color < numOfChannels; ++color) 
        {
            gij_lr_derivative(currRowMatrix, color) = delta_j_p_1[color] - delta_i_p_P[color];
            gji_rl_derivative(currRowMatrix, color) = -gij_lr_derivative(currRowMatrix, color);
        }
    }

    //3. Compute g_il_derivative, g_jr_derivative (including dummy gradients)
    Eigen::MatrixXd g_il_derivative = Eigen::MatrixXd::Constant(sizeWithDummyGradients, numOfChannels, 0);
    Eigen::MatrixXd g_jr_derivative = Eigen::MatrixXd::Constant(sizeWithDummyGradients, numOfChannels, 0);

    //3.1. Add real data to g_il_derivative, g_jr_derivative 
    for (int32_t p = 1; p < numOfPixels; ++p) 
    {
        const int32_t currRowMatrix = p - 1;
        const ExtendedPixelType delta_i_p_P = ImageUtils::GetImprovedMGC_Delta(in_leftPiece->GetRotatedPiece(), p, numOfPixels - 1);
        const ExtendedPixelType delta_i_p_Pmin1 = ImageUtils::GetImprovedMGC_Delta(in_leftPiece->GetRotatedPiece(), p, numOfPixels - 2);
        const ExtendedPixelType delta_j_p_0 = ImageUtils::GetImprovedMGC_Delta(in_rightPiece->GetRotatedPiece(), p, 0);
        const ExtendedPixelType delta_j_p_1 = ImageUtils::GetImprovedMGC_Delta(in_rightPiece->GetRotatedPiece(), p, 1);

        for (int32_t color = 0; color < numOfChannels; ++color) 
        {
            g_il_derivative(currRowMatrix, color) = delta_i_p_P[color] - delta_i_p_Pmin1[color];
            g_jr_derivative(currRowMatrix, color) = delta_j_p_0[color] - delta_j_p_1[color];
        }
    }

    //3.2. Add dummy gradients to g_il, g_jr 
    for (int32_t p = numOfPixels; p < sizeWithDummyGradients; ++p) 
    {
        const int32_t currRowMatrix = p - 1;
        const auto currDummyGradient = dummyGradientMatrix.row(p - numOfPixels);
        g_il_derivative.row(currRowMatrix) = currDummyGradient;
        g_jr_derivative.row(currRowMatrix) = currDummyGradient;
    }

    //4. Compute leftInverseCov and rightInverseCov
    const Eigen::MatrixXd leftInverseCov = Utilities::ComputeInverseOfCovarianceMatrix(g_il_derivative);
    const Eigen::MatrixXd rightInverseCov = Utilities::ComputeInverseOfCovarianceMatrix(g_jr_derivative);

    //5. Compute eij_lr_derivative, eji_rl_derivative
    Eigen::MatrixXd eij_lr_derivative = Eigen::MatrixXd::Constant(numOfPixels, numOfChannels, 0);
    Eigen::MatrixXd eji_rl_derivative = Eigen::MatrixXd::Constant(numOfPixels, numOfChannels, 0);

    for (int32_t p = 1; p < numOfPixels; ++p) 
    {
        const int32_t currRowMatrix = p - 1;

        const ExtendedPixelType delta_i_p_P = ImageUtils::GetImprovedMGC_Delta(in_leftPiece->GetRotatedPiece(), p, numOfPixels - 1);
        const ExtendedPixelType delta_i_p_Pmin1 = ImageUtils::GetImprovedMGC_Delta(in_leftPiece->GetRotatedPiece(), p, numOfPixels - 2);
        const ExtendedPixelType delta_j_p_1 = ImageUtils::GetImprovedMGC_Delta(in_rightPiece->GetRotatedPiece(), p, 1);
        const ExtendedPixelType delta_j_p_0 = ImageUtils::GetImprovedMGC_Delta(in_rightPiece->GetRotatedPiece(), p, 0);

        for (int32_t color = 0; color < numOfChannels; ++color) 
        {
            eij_lr_derivative(currRowMatrix, color) = 0.5 * ((delta_i_p_P[color] - delta_i_p_Pmin1[color]) + 
                (delta_j_p_1[color] - delta_j_p_0[color]));

            eji_rl_derivative(currRowMatrix, color) = - eij_lr_derivative(currRowMatrix, color);
        }
    }

    //6. Compute dlrDerivative
    double dlrDerivative = 0;
    for (int32_t p = 0; p < numOfAlongsideDerivatives; ++p) 
    {
        const Eigen::Vector3d currRemainderVec = gij_lr_derivative.row(p) - eij_lr_derivative.row(p); 
        const Eigen::MatrixXd currValueIn1x1Matrix = currRemainderVec.transpose() * leftInverseCov * currRemainderVec;
        const double currValue = currValueIn1x1Matrix.sum();

        dlrDerivative += currValue;
    }
    out_dlrDerivative = dlrDerivative;

    //7. Compute drl
    double drlDerivative = 0;
    for (int32_t p = 0; p < numOfAlongsideDerivatives; ++p) 
    {
        const Eigen::Vector3d currRemainderVec = gji_rl_derivative.row(p) - eji_rl_derivative.row(p); 
        const Eigen::MatrixXd currValueIn1x1Matrix = currRemainderVec.transpose() * rightInverseCov * currRemainderVec;
        const double currValue = currValueIn1x1Matrix.sum();

        drlDerivative += currValue;
    }
    out_drlDerivative = drlDerivative;
}

/************************************************************************/
PuzzleRL_Solver::PieceDissimilarityFuncType PuzzleRL_Solver::GetDissimilarityFunction(const DissimilarityType in_dissType) const
/************************************************************************/
{
    PieceDissimilarityFuncType dissimilarityFunc = std::mem_fn(&PuzzleRL_Solver::GetImprovedMGC_Dissimilarity);

    switch (in_dissType)
    {
        case DissimilarityType::eImprovedMGC:
            dissimilarityFunc = std::mem_fn(&PuzzleRL_Solver::GetImprovedMGC_Dissimilarity);
            break;

        default:
            Utilities::LogAndAbort("Illegal dissimilarity type");
            break;
    }

    return dissimilarityFunc;
}

/************************************************************************/
int32_t PuzzleRL_Solver::GetOneBasedIndexInDissimilaritiesOrder(const PieceObjectAndRot* in_rotatedPiece, const Orientation in_orien,
    const PieceObjectAndRot* in_rotatedPieceToLookFor, const DissimilarityType in_dissimilarityType) const
//Function input: in_rotatedPiece: rotated piece object; in_orien: orientation; in_rotatedPieceToLookFor: rotated piece that we should look for;
//  in_dissimilarityType: dissimilarity
//Function output: the index of the dissimilarity of 'in_rotatedPieceToLookFor' among all the dissimilarities in relation 'in_orien' to 'in_rotatedPiece'
//Function objective: described in "Function output"
/************************************************************************/
{
    //1. Get dissimilarity info
    const DissimilarityInfo& dissimilarityInfo = GetDissimilarityInfo(in_rotatedPiece, in_orien, in_dissimilarityType);

    //2. Get the actual piece to look for (it's different than 'in_rotatedPieceToLookFor' due to optimization in 'GetDissimilarityInfo()')
    const PieceObjectAndRot* actualRotatedPieceToLookFor = GetPieceToLookForInDissimilarityInfo(in_rotatedPiece, in_rotatedPieceToLookFor);

    //3. Find the index of 'actualRotatedPieceToLookFor' in 'dissimilarityInfo.m_sortedOtherRotatedPieces'
    int32_t retVal = -1;
    for (int32_t i = 0; i < static_cast<int32_t>(dissimilarityInfo.m_sortedOtherRotatedPieces.size()); ++i)
    {
        const PieceObjectAndRot* currPieceObjAndRot = dissimilarityInfo.m_sortedOtherRotatedPieces[i].first;
        if (*currPieceObjAndRot == *actualRotatedPieceToLookFor)
        {
            retVal = dissimilarityInfo.m_sortedOtherRotatedPieces[i].second;
    
            break;
        }
    }

    Utilities::LogAndAbortIf(-1 == retVal, "'GetOneBasedIndexInDissimilaritiesOrder()' failed");

    return retVal;
}

/************************************************************************/
void PuzzleRL_Solver::SetDissimilarityInfo(const DissimilarityType in_dissimilarityType)
//Function input: in_rotatedPiece: rotated piece object; in_orien: orientation; in_dissimilarityType: dissimilarity
//Function output: the dissimilairy info of 'in_rotatedPiece' and orientation 'in_orien'
//Function objective: described in "Function output"
/************************************************************************/
{
    //1. Get pieces Dissimilarities tensor and the relevant dissimilarity info matrix
    DissimilarityInfoMatrix& dissInfoMatrix = m_dissimilarityInfoMatrixArr.at(static_cast<int32_t>(in_dissimilarityType));

    //2. Allocate 'dissInfoMatrix'
    dissInfoMatrix = DissimilarityInfoMatrix::Constant(m_puzzlePieceObjectsPool.size(), Orientation::eNumOfPossibleOrientations, DissimilarityInfo());

    //3. For each piece
    std::for_each(executionPolicy, m_puzzlePieceObjectsPool.begin(), m_puzzlePieceObjectsPool.end(),
    [&](const PuzzlePieceObject* currPieceObj) -> void
    {
        //Get 'currPieceObj' rotated in zero degress 
        const PieceObjectAndRot* currRotatedPiece = GetRotatedPieceFromPieceObjectAndRotation(currPieceObj, ImageRotation::e_0_degrees);

        //Compute and set the dissimilarity information of 'currRotatedPiece' in all orientations
        for (Orientation orien = Orientation::eUp; orien < Orientation::eNumOfPossibleOrientations; orien = static_cast<Orientation>(orien + 1))
        {
            DissimilarityInfo& currDissimilarityInfo = dissInfoMatrix(currPieceObj->m_pieceNumber, orien);
            ComputeDissimilarityInfo(currRotatedPiece, orien, m_piecesDissimilarities, currDissimilarityInfo);
        }
    });
}

/************************************************************************/
void PuzzleRL_Solver::ComputeDissimilarityInfo(const PieceObjectAndRot* in_rotatedPiece, const Orientation in_orien, 
    const PiecesTensorType& in_piecesDissimilaritiesTensor, DissimilarityInfo& out_dissimilarityInfo) const
/************************************************************************/
{
    std::vector<PieceObjectAndRotationValPair> vec;

    //1. Push all relevent pairs to vec
    for (const PieceObjectAndRot* currOtherRotatedPiece : m_rotatedPiecesPool)
    {
        if (!PieceObjectAndRot::ArePieceObjectsIdentical(*in_rotatedPiece, *currOtherRotatedPiece))
        {
            const double val = in_piecesDissimilaritiesTensor(in_rotatedPiece->m_index, currOtherRotatedPiece->m_index, in_orien);
            vec.emplace_back(currOtherRotatedPiece, val);
        }
    }

    //2. Sort vec by dissimilarity values
    std::sort(vec.begin(), vec.end(),
        [](const PieceObjectAndRotationValPair& in_lhs, const PieceObjectAndRotationValPair& in_rhs)
        {
            return in_lhs.second < in_rhs.second;
        });

    //3. Set 'out_dissimilarityInfo.m_sortedOtherRotatedPieces' first elements, and 'out_dissimilarityInfo.m_sortedDissimilarityValues'
    for (const PieceObjectAndRotationValPair& currPair : vec)
    {
        out_dissimilarityInfo.m_sortedOtherRotatedPieces.emplace_back(currPair.first, -1);
        out_dissimilarityInfo.m_sortedDissimilarityValues.push_back(currPair.second);
    }

    //4. Set 'out_dissimilarityInfo.m_sortedOtherRotatedPieces' second elements
    //We give the same rank for equal dissimilarity value.
    //For example, for the following dissimilarity values: 0.1, 0.1, 0.1, 0.2, 0.3, 0.4, 0.4, 0.8
    // - the ranking is: 1, 1, 1, 4, 5, 5, 6, 6, 7

    //Set the rank of the min dissimilarity piece (it's 1 and not zero since we used 1-based indices in phi computation)
    out_dissimilarityInfo.m_sortedOtherRotatedPieces[0].second = 1;
    
    for (int32_t i = 1; i < vec.size(); ++i)
    {
        const double prevDissVal = out_dissimilarityInfo.m_sortedDissimilarityValues[i - 1];
        const double currDissVal = out_dissimilarityInfo.m_sortedDissimilarityValues[i];

        //If the current value is the same as the previous one - we give them the same rank
        if (currDissVal == prevDissVal)
        {
            const int32_t prevRank = out_dissimilarityInfo.m_sortedOtherRotatedPieces[i - 1].second;
            out_dissimilarityInfo.m_sortedOtherRotatedPieces[i].second = prevRank;
        }
        //(if the current value is the bigger than the previous one - we given rank according to the ordering)
        else
        {
            out_dissimilarityInfo.m_sortedOtherRotatedPieces[i].second = i + 1;
        }
    }
}

/************************************************************************/
const DissimilarityInfo& PuzzleRL_Solver::GetDissimilarityInfo(const PieceObjectAndRot* in_rotatedPiece, const Orientation in_orien, 
    const DissimilarityType in_dissimilarityType) const
//Function input: in_rotatedPiece: rotated piece object; in_orien: orientation; in_dissimilarityType: dissimilarity
//Function output: the dissimilairy info of 'in_rotatedPiece' and orientation 'in_orien'
//Function objective: described in "Function output"
/************************************************************************/
{
    //The logic here stems from running time optimization. See more details in 'PuzzleRL_Solver::SetDissimilarityInfo()'

    //1. Get the rotation that should be apply on 'in_rotatedPiece', to make its rotation to be zero degrees
    const ImageRotation rotationToApplyForZeroRotation = ImageUtils::GetRotationDifference(ImageRotation::e_0_degrees, in_rotatedPiece->m_rotation);

    //2. Get the orientation relevant to the zero degrees rotated piece
    const Orientation actualOrien = static_cast<Orientation>((in_orien + rotationToApplyForZeroRotation) % Orientation::eNumOfPossibleOrientations);

    //3. Get dissimilarity info from 'm_dissimilarityInfoMatrixArr'
    const DissimilarityInfoMatrix& matrix = m_dissimilarityInfoMatrixArr.at(static_cast<int32_t>(in_dissimilarityType));
    const DissimilarityInfo& dissimilarityInfo = matrix(in_rotatedPiece->m_pieceObject->m_pieceNumber, actualOrien);
    
    return dissimilarityInfo;
}

/************************************************************************/
const PieceObjectAndRot* PuzzleRL_Solver::GetPieceToLookForInDissimilarityInfo(const PieceObjectAndRot* in_mainRotatedPiece, 
    const PieceObjectAndRot* in_rotatedPieceToLookFor) const
/************************************************************************/
{
    const ImageRotation mainPieceRotationDiff = ImageUtils::GetRotationDifference(ImageRotation::e_0_degrees, in_mainRotatedPiece->m_rotation);
    const PieceObjectAndRot* actualRotatedPieceToLookFor = RotateAndGetPiece(in_rotatedPieceToLookFor, mainPieceRotationDiff);

    return actualRotatedPieceToLookFor;
}

/************************************************************************/
void PuzzleRL_Solver::SetInitialLabeling()
//Function input: none
//Function output: none
//Function objective: set RL initial labeling, and set m_currentTopLabelsMatrix
/************************************************************************/
{
    SetDefaultInitLabeling();

    if (m_config.GetPuzzleType() == PuzzleType::eType2_UknownRotation)
        ManipulateInitialLabelingForType2Puzzles();

    m_currentTopLabelsMatrix = m_initLabeling->GetBinaryLabeling().GetLabelingMatrix().cast<int>();
}

/************************************************************************/
void PuzzleRL_Solver::SetDefaultInitLabeling() 
//Function input: none
//Function output: none
//Function objective: set initial uniform labeling
/************************************************************************/
{
    const size_t numOfObjects = m_puzzlePieceObjectsPool.size();
    const size_t numOfLabels = m_locationRotationLabelsPool.size();
    const double initProb = 1.0 / numOfLabels;

    m_initLabeling = new Labeling(numOfObjects, numOfLabels, initProb);
}

/************************************************************************/
void PuzzleRL_Solver::ManipulateInitialLabelingForType2Puzzles()
//Function input: none
//Function output: none
//Function objective: manipulate initial labeling so solution rotation is determined
/************************************************************************/
{
    // Manipulate labeling according to single piece
    SetLabelingManipulationForType2PuzzlesBySinglePieceVariables();
    ManipulateLabelingForType2PuzzlesBySinglePiece(*m_initLabeling, static_cast<int32_t>(m_puzzlePieceObjectsPool.size()));
}

/************************************************************************/
void PuzzleRL_Solver::SetLabelingManipulationForType2PuzzlesBySinglePieceVariables()
//Function input: none
//Function output: none
//Function objective: set all variables related to type 2 labeling manipulation by single piece
/************************************************************************/
{
    // Find and set single piece with max surrounding comp and set 'm_isOnlyOneRotationAllowed'
    m_firstRotationDeterminedPiece = GetSinglePieceWithMaxSurroundingComp();
}

/************************************************************************/
void PuzzleRL_Solver::ManipulateLabelingForType2PuzzlesBySinglePiece(Labeling& inout_labeling, const int32_t in_numOfAvailableLocations) const
//Function input: inout_labeling: labeling to be manipulated; in_numOfAvailableLocations: num of available locations in labeling
//Function output: none
//Function objective: manipulate labeling by single piece, so solution rotation is determined
/************************************************************************/
{
    //1. Do manipulation labeling according to 'm_firstRotationDeterminedPiece' and 'm_allowedRotationForFirstRotationDeterminedPiece'
    DoLabelingManipulationForType2PuzzlesBySinglePiece(inout_labeling, 
        m_firstRotationDeterminedPiece, m_allowedRotationForFirstRotationDeterminedPiece, in_numOfAvailableLocations);
}

/************************************************************************/
void PuzzleRL_Solver::DoLabelingManipulationForType2PuzzlesBySinglePiece(Labeling& inout_labeling, 
    const PuzzlePieceObject* in_selectedPiece, const ImageRotation& in_allowedRotation, const int32_t in_numOfAvailableLocations) const
//Function input: inout_labeling: labeling to manipulate; in_selectedRotatedPiece: piece according to which the solution rotation is determined;
    //in_allowedRotation: allowed rotation; in_numOfAvailableLocations: num of available locations in labeling
//Function output: none
//Function objective: manipulate labeling so solution rotation is determined
/************************************************************************/
{
    // Set initial labeling
    SetLabelingForPieceAndAllowedRotations(inout_labeling, in_selectedPiece, in_allowedRotation, in_numOfAvailableLocations);
}

/************************************************************************/
const PuzzlePieceObject* PuzzleRL_Solver::GetSinglePieceWithMaxSurroundingComp() const
//Function input: none
//Function output: single piece, such that this piece has maximum surrounding best comp values from its 4 sides
//Function objective: described in "Function output"
/************************************************************************/
{
    //1. Init variables
    std::set<const PuzzlePieceObject*> selectedPiecesSet;
    double maxSurroundingComp = -1;

    //2. For all pieces
    for (const PuzzlePieceObject* currPiece : m_puzzlePieceObjectsPool)
    {
        //2.1. Get 'currPiece' as rotated piece (it does not matter in which rotation)
        const PieceObjectAndRot* currRotatedPiece = GetRotatedPieceFromPieceObjectAndRotation(currPiece, e_0_degrees);

        if (!IsConstantPiece(currRotatedPiece))
        {
            //2.2. Find surrounding compatibilities of 'currPiece'
            double currSurroundingComp = 0.0;
            for (Orientation orien = Orientation::eUp; orien < Orientation::eNumOfPossibleOrientations; orien = static_cast<Orientation>(orien + 1))
            {
                const PieceObjectAndRot* mostCompatibleRotatedPieceInOrien = 
                    GetMostCompatiblePiece(currRotatedPiece, orien, m_piecesCompatibilities);

                if (nullptr != mostCompatibleRotatedPieceInOrien)
                {
                    const double compVal = m_piecesCompatibilities(currRotatedPiece->m_index, mostCompatibleRotatedPieceInOrien->m_index, orien);
                    currSurroundingComp += compVal;
                }
            }

            //2.3. Update 'selectedPiece' and 'maxSurroundingComp'
            if (currSurroundingComp > maxSurroundingComp)
            {
                selectedPiecesSet.clear();
                selectedPiecesSet.insert(currPiece);
                maxSurroundingComp = currSurroundingComp;
            }
            else if (currSurroundingComp == maxSurroundingComp)
            {
                selectedPiecesSet.insert(currPiece);
            }
        }        
    }

    const PuzzlePieceObject* selectedPiece = *selectedPiecesSet.begin();
    Utilities::Log("Determined rotation of piece " + selectedPiece->GetString() + ", with surrounding comp of " + std::to_string(maxSurroundingComp));

    return selectedPiece;
}

/************************************************************************/
void PuzzleRL_Solver::SetLabelingForPieceAndAllowedRotations(Labeling& inout_labeling, const PuzzlePieceObject* in_puzzlePieceObj,
    const ImageRotation& in_allowedRotation, const int32_t in_numOfAvailableLocations) const
//Function input: inout_labeling: labeling; in_puzzlePieceObj: puzzle piece; in_allowedRotation: allowed rotation for 'in_puzzlePieceObj';
    //in_numOfAvailableLocations: num of available locations in labeling
//Function output: none
//Function objective: manipulate the entries in 'inout_labeling' to allow 'in_puzzlePieceObj' only the rotations in 'in_allowedRotationsVec' 
/************************************************************************/
{
    const double probabilityInitVal = 1.0 / in_numOfAvailableLocations;

    for (const LocationRotationLabel* locationRotationLabel : m_locationRotationLabelsPool)
    {
        if (locationRotationLabel->m_rotation == in_allowedRotation)
            inout_labeling.SetLabelingValue(*in_puzzlePieceObj, *locationRotationLabel, probabilityInitVal);
        else
            inout_labeling.SetLabelingValue(*in_puzzlePieceObj, *locationRotationLabel, 0);
    }
}

/************************************************************************/
const PuzzlePieceObject* PuzzleRL_Solver::GetPuzzlePieceObjectByIndex(const int32_t& in_index) const
/************************************************************************/
{
    const PuzzlePieceObject* pieceObject = m_puzzlePieceObjectsPool[in_index];
    Utilities::LogAndAbortIf(in_index != m_puzzlePieceObjectsPool[in_index]->m_index, "");

    return pieceObject;
}

/************************************************************************/
const PuzzlePieceObject* PuzzleRL_Solver::GetPuzzlePieceObjectByPieceNumber(const int32_t& in_pieceNumber) const
/************************************************************************/
{
    const PuzzlePieceObject* retVal = nullptr;
    for (const PuzzlePieceObject* piece: m_puzzlePieceObjectsPool)
    {
        if (piece->m_pieceNumber == in_pieceNumber)
        {
            retVal = piece;
            break;
        }
    }
    Utilities::LogAndAbortIf(retVal == nullptr, "retVal == nullptr in GetPuzzlePieceObjectByPieceNumber");

    return retVal;
}

/************************************************************************/
const LocationRotationLabel* PuzzleRL_Solver::GetLocationRotationLabelByCoordAndRot(const Utilities::RowColInfo& in_coord, 
    const ImageRotation& in_rotation) const
/************************************************************************/
{
    const LocationRotationLabel* retVal = nullptr;
    for (const LocationRotationLabel* currLabel: m_locationRotationLabelsPool)
    {
        if (currLabel->m_row == in_coord.m_row && currLabel->m_column == in_coord.m_col && currLabel->m_rotation == in_rotation)
        {
            retVal = currLabel;
            break;
        }
    }

    Utilities::LogAndAbortIf(retVal == nullptr, "In 'PuzzleRL_Solver::GetLocationRotationLabelByCoordAndRot()': retVal == nullptr, in_coord is " + in_coord.GetString());
    return retVal;
}

/************************************************************************/
LocationRotationLabelsVector PuzzleRL_Solver::GetAllLocationRotationLabelsForLocation(const Utilities::RowColInfo& in_coord) const
/************************************************************************/
{
    return m_locationRotationLabelsMatrix(in_coord.m_row, in_coord.m_col); 
}

/************************************************************************/
const PieceObjectAndRot* PuzzleRL_Solver::GetRotatedPieceFromPieceNumberAndRotation(const PieceNumberAndRotation& in_pieceNumAndRotation) const
/************************************************************************/
{
    return GetRotatedPieceFromPieceNumberAndRotation(in_pieceNumAndRotation.m_pieceNumber, in_pieceNumAndRotation.m_rotation);
}

/************************************************************************/
const PieceObjectAndRot* PuzzleRL_Solver::GetRotatedPieceFromPieceNumberAndRotation(const int32_t in_pieceNumber, const ImageRotation in_rotation) const
/************************************************************************/
{
    return PieceObjectAndRot::GetRotatedPieceFromPieceNumberAndRotation(in_pieceNumber, in_rotation, m_rotatedPiecesPool);
}

/************************************************************************/
const PieceObjectAndRot* PuzzleRL_Solver::RotateAndGetPiece(const PieceObjectAndRot* in_rotatedPiece, const ImageRotation in_rotationToApply, 
    const bool in_mayGetPieceNotInPool) const
//Function input: in_rotatedPiece: rotated piece object; in_rotationToApply: rotation to apply to in_rotatedPiece;
//  in_mayGetPieceNotInPool: boolean indicating whether we may a rotated piece which is not in 'm_rotatedPiecesPool' (may happen only for type-1 puzzles) 
//Function output: 'PieceObjectAndRot' instance which is the rotated piece in in_rotatedPiece, rotated by additional 'in_rotationToApply' degrees
//Function objective: described in "Function output"
/************************************************************************/
{
    //1. Compute the desired rotation (the rotation of 'in_rotatedPiece' after adding 'in_rotationToApply')
    const ImageRotation desiredRotation = ImageUtils::SumSquareImageRotations(in_rotatedPiece->m_rotation, in_rotationToApply);

    //2. Look for desired the 'PieceObjectAndRot' instance in 'm_rotatedPiecesPool', and return it
    auto itr = std::find_if(m_rotatedPiecesPool.begin(), m_rotatedPiecesPool.end(), 
        [&](const PieceObjectAndRot* in_element)
    {
        return PieceObjectAndRot::ArePieceObjectsIdentical(*in_element, *in_rotatedPiece) && in_element->m_rotation == desiredRotation;
    });
    
    //3. If rotated piece not found
    if (itr == m_rotatedPiecesPool.end())
    {
        //3.1. If 'in_mayGetPieceNotInPool' is true - try to find the piece in ''m_rotatedPiecesExtraPool, and if it's not there - create it
        if (in_mayGetPieceNotInPool)
        {
            itr = std::find_if(m_rotatedPiecesExtraPool.begin(), m_rotatedPiecesExtraPool.end(), 
                [&](const PieceObjectAndRot* in_element)
            {
                return PieceObjectAndRot::ArePieceObjectsIdentical(*in_element, *in_rotatedPiece) && in_element->m_rotation == desiredRotation;
            });

            Utilities::LogAndAbortIf(itr == m_rotatedPiecesExtraPool.end(), "'PuzzleRL_Solver::RotateAndGetPiece()' failed");
        }
        //3.2. Else - crash
        else
        {
            Utilities::LogAndAbortIf(itr == m_rotatedPiecesPool.end(), "'PuzzleRL_Solver::RotateAndGetPiece()' failed");
        }
    }

    return *itr;
}
     
/************************************************************************/
void PuzzleRL_Solver::GetRotatedPieceAndCoordinateFromLabelingEntry(const RowColInfo& in_labelingEntry,
    const PieceObjectAndRot*& out_rotatedPiece, RowColInfo& out_coord) const
/************************************************************************/
{
    const PuzzlePieceObject* puzzlePiece = m_puzzlePieceObjectsPool[in_labelingEntry.m_row];
    const LocationRotationLabel* locationRotationLabel = m_locationRotationLabelsPool[in_labelingEntry.m_col];

    out_rotatedPiece = GetRotatedPieceFromPieceObjectAndRotation(puzzlePiece, locationRotationLabel->m_rotation);
    out_coord = RowColInfo(locationRotationLabel->m_row, locationRotationLabel->m_column);
}

/************************************************************************/
cv::Mat PuzzleRL_Solver::GetAssemblyImageFromBinaryLabeling(const Labeling& in_binaryLabeling) const
//Function input: in_binaryLabeling: binary labeling (values are 0 or 1 in labeling matrix)
//Function output: image built according to 'in_binaryLabeling'
//Function objective: described in "Function output"
/************************************************************************/
{
    //1. Verify that 'in_binaryLabeling' is a binary matrix
    Utilities::LogAndAbortIf(!in_binaryLabeling.IsBinaryLabeling(), 
        "!in_binaryLabeling.IsBinaryLabeling() in 'PuzzleRL_Solver::IsBinaryLabeling'");

    //2. Build assemblyImage, which is an image built according to 'in_binaryLabeling'
    cv::Mat assemblyImage = cv::Mat(m_puzzleImage.GetImage().rows, m_puzzleImage.GetImage().cols, ImageUtils::imagesType, ImageUtils::Colors::m_grayLab);

    for (const PuzzlePieceObject* puzzlePieceObject : m_puzzlePieceObjectsPool)
    {
        const LabelAndProbPair maxLabelAndProb = in_binaryLabeling.GetMaxLabelAndProb(*puzzlePieceObject);
        const LocationRotationLabel* locRotLabel = static_cast<const LocationRotationLabel*>(maxLabelAndProb.first);

        if (1 == maxLabelAndProb.second)
            SetRotatedPieceInLocation(puzzlePieceObject, locRotLabel, assemblyImage);
    }

    return assemblyImage;
}

/************************************************************************/
PieceNumbersAndRotationsMatrix PuzzleRL_Solver::GetAssignmentMatrixFromBinaryLabeling(const Labeling& in_binaryLabeling) const
//Function input: in_binaryLabeling: binary labeling (values are 0 or 1 in labeling matrix)
//Function output: assignment matrix according to 'in_binaryLabeling'
//Function objective: described in "Function output"
/************************************************************************/
{
    //1. Verify that 'in_binaryLabeling' is a binary matrix
    Utilities::LogAndAbortIf(!in_binaryLabeling.IsBinaryLabeling(), 
        "!in_binaryLabeling.IsBinaryLabeling() in 'PuzzleRL_Solver::IsBinaryLabeling'");

    //2. Build pieceAssignmentMatrix, which assignment matrix according to 'in_binaryLabeling'
    PieceNumbersAndRotationsMatrix pieceAssignmentMatrix = PieceNumbersAndRotationsMatrix(m_puzzleImage.GetNumOfRowPieces(), m_puzzleImage.GetNumOfColPieces());

    for (const PuzzlePieceObject* puzzlePieceObject : m_puzzlePieceObjectsPool)
    {
        const std::pair<const Label*, double> maxLabelAndProb = in_binaryLabeling.GetMaxLabelAndProb(*puzzlePieceObject);
        if (1.0 == maxLabelAndProb.second)
        {
            const LocationRotationLabel* locRotLabel = static_cast<const LocationRotationLabel*>(maxLabelAndProb.first);

            const PieceNumberAndRotation currPieceNumberAndRotation(puzzlePieceObject->m_pieceNumber, locRotLabel->m_rotation);

            pieceAssignmentMatrix(locRotLabel->m_row, locRotLabel->m_column) = currPieceNumberAndRotation;
        }
    }

    return pieceAssignmentMatrix;
}

/************************************************************************/
void PuzzleRL_Solver::SetRotatedPieceInLocation(const PuzzlePieceObject* in_puzzlePieceObject, const LocationRotationLabel* in_locRotLabel, cv::Mat& out_image) const
//Function input: in_puzzlePieceObject: puzzle piece object; in_locRotLabel: location-rotation label; out_image: image
//Function output: none
//Function objective: set the piece defined by 'in_puzzlePieceObject' after rotation defined by 'in_locRotLabel', in grid location defined by 'in_locRotLabel'
/************************************************************************/
{
    ImageUtils::SetRotatedPieceInLocation(in_puzzlePieceObject->GetPiece(), in_locRotLabel->m_rotation, 
        Utilities::RowColInfo(in_locRotLabel->m_row, in_locRotLabel->m_column), out_image);
}

/************************************************************************/
double PuzzleRL_Solver::ComputeSolutionDirectComparisonAndSetRotationSolution() const
//Function input: none
//Function output: number in the range [0,1] representing the direct comparison performance measure of the solution in m_solutionRotation
//Function objective: in addition to compute the direct comparison performance measure, this function sets m_solutionRotation to be the 
//  solution rotation that achieved the best direct comparison performance score
/************************************************************************/
{
    RowColInfoSet wrongAssignedPieceCoordsSet;
    ImageRotation solutionRotation = ImageRotation::eInvalidRotation; 
    const double directComparison = ComputeSolutionDirectComparison(m_finalPieceAssignmentMatrix, solutionRotation, wrongAssignedPieceCoordsSet);

    m_solutionRotation = solutionRotation;

    if (directComparison != 1)
    {
        const PieceNumbersAndRotationsMatrix rotatedSolution = 
            ImageUtils::RotatePuzzleAssignmentMatrix(m_finalPieceAssignmentMatrix, m_solutionRotation);
        m_puzzleRL_SolverOutputManager.LogWrongAssignmentsInfo(wrongAssignedPieceCoordsSet, rotatedSolution);
    }

    return directComparison;
}

/************************************************************************/
double PuzzleRL_Solver::ComputeSolutionDirectComparison(const PieceNumbersAndRotationsMatrix& in_assignmentMatrix,
    ImageRotation& out_solutionRotation, RowColInfoSet& out_wrongAssignedPieceCoordsSet) const
//Function input: in_assignmentMatrix: pieces assinnment matrix; out_solutionRotation: out solution rotation;
    //; out_wrongAssignedPieceCoordsSet: out wrong assigned pieces info
//Function output: number in the range [0,1] representing the direct comparison performance measure of the solution in m_solutionRotation
//Function objective: desribed in "Function output" 
/************************************************************************/
{
    double directComparison = std::numeric_limits<double>::lowest();
    
    const GroundTruthSolutionInfo groundTruthSolutionInfo = m_puzzleImage.GetGroundTruthSolutionInfo();

    Utilities::LogAndAbortIf(in_assignmentMatrix.rows() != groundTruthSolutionInfo.m_assignmentMatrix.rows() 
        || in_assignmentMatrix.cols() != groundTruthSolutionInfo.m_assignmentMatrix.cols(), "'ComputeSolutionDirectComparison()' failed");

    //For all relevant rotations
    const RotationsVec rotationsToCheck = ImageUtils::GetRotationsToCheck(m_puzzleImage.IsSquarePuzzle(), m_config.GetPuzzleType());

    for (const ImageRotation& currRotation : rotationsToCheck)
    {
        //1. Rotate the solution
        const PieceNumbersAndRotationsMatrix rotatedSolution = ImageUtils::RotatePuzzleAssignmentMatrix(in_assignmentMatrix, currRotation);

        //2. Get direct comparison result between the ground truth solution and rotatedSolution
        RowColInfoSet currWrongAssignedPieceCoordsSet;
        const double currDirectComparison = ImageUtils::ComputeDirectComparison(groundTruthSolutionInfo, rotatedSolution, m_puzzleImage.GetNumOfPieces(), 
            currWrongAssignedPieceCoordsSet);

        //3. Update direct comparison result and solution rotation, and break if achieved maximal direct comparison score 
        if (currDirectComparison > directComparison)
        {
            directComparison = currDirectComparison;
            out_solutionRotation = currRotation;
            out_wrongAssignedPieceCoordsSet = currWrongAssignedPieceCoordsSet;

            if (1.0 == directComparison)
                break;
        }
    }

    return directComparison;
}

/************************************************************************/
double PuzzleRL_Solver::ComputeSolutionNeighborComparison() const
//Function input: none
//Function output: number in the range [0,1] representing the neighbor comparison performance measure of the solution in m_solutionRotation
//Function objective: described in "Function output"
/************************************************************************/
{
    const GroundTruthSolutionInfo groundTruthSolutionMatrix = m_puzzleImage.GetGroundTruthSolutionInfo();

    Utilities::LogAndAbortIf(m_finalPieceAssignmentMatrix.rows() != groundTruthSolutionMatrix.m_assignmentMatrix.rows() 
        || m_finalPieceAssignmentMatrix.cols() != groundTruthSolutionMatrix.m_assignmentMatrix.cols(), "'ComputeSolutionNeighborComparison()' failed");

    //Compute rotated solution according to m_solutionRotation (m_solutionRotation should be already set)
    const PieceNumbersAndRotationsMatrix rotatedSolution = ImageUtils::RotatePuzzleAssignmentMatrix(m_finalPieceAssignmentMatrix, m_solutionRotation);

    const double neighborComparison = ImageUtils::ComputeNeighborComparison(groundTruthSolutionMatrix, rotatedSolution);

    return neighborComparison;
}

/************************************************************************/
bool PuzzleRL_Solver::AreBestBuddies(const PieceObjectAndRot* in_firstPiece, const PieceObjectAndRot* in_secondPiece, const Orientation in_orien,
    const PiecesTensorType& in_pieceCompatibilitiesTensor) const
//Function input: in_firstPiece: first rotated piece; in_secondPiece: second rotated piece; in_orien: orientation;
    //in_pieceCompatibilitiesTensor: piece comp tensor to check best buddies in
//Function output: boolean indicating whether (in_firstPiece, in_secondPiece, in_orien) are best buddies or not   
//Function objective: described in "Function output"
/************************************************************************/
{
    const PieceObjectAndRot* mostCompatiblePieceToFirst =
        GetMostCompatiblePiece(in_firstPiece, in_orien, in_pieceCompatibilitiesTensor);

    bool areBestBuddies = false;
    if (nullptr != mostCompatiblePieceToFirst && *mostCompatiblePieceToFirst == *in_secondPiece)
    {
        const PieceObjectAndRot* mostCompatiblePieceToSecond =
        GetMostCompatiblePiece(in_secondPiece, GetOppositeOrientation(in_orien), in_pieceCompatibilitiesTensor);

        if (nullptr != mostCompatiblePieceToSecond && *mostCompatiblePieceToSecond == *in_firstPiece)
        {
            areBestBuddies = true;    
        }
    }

    return areBestBuddies;
}

/************************************************************************/
const PieceObjectAndRot* PuzzleRL_Solver::GetMostCompatiblePiece(const PieceObjectAndRot* in_pieceObjAndRot, const Orientation in_orien, 
    const PiecesTensorType& in_pieceCompatibilitiesTensor) const
/************************************************************************/
{
    std::vector<PieceObjectAndRotationValPair> vec;

    //1. Push all relevent pairs to vec
    for (const PieceObjectAndRot* currRotatedPiece : m_rotatedPiecesPool)
    {
        if (!PieceObjectAndRot::ArePieceObjectsIdentical(*currRotatedPiece, *in_pieceObjAndRot))
        {
            const double val = in_pieceCompatibilitiesTensor(in_pieceObjAndRot->m_index, currRotatedPiece->m_index, in_orien);
            vec.emplace_back(currRotatedPiece, val);
        }
    }

    //2. Sort vec by dissimilarity values
    std::sort(vec.begin(), vec.end(),
        [](const PieceObjectAndRotationValPair& in_lhs, const PieceObjectAndRotationValPair& in_rhs)
        {
            return in_lhs.second > in_rhs.second;
        });

    //3. Compute propotion of the best compatibility values from the second best compatibility value
    const double propotionOfBestFromSecondBest = vec[0].second / vec[1].second;

    //4. If propotionOfBestFromSecondBest is bigger than 1, we have found our most compatible piece
    const PieceObjectAndRot* mostCompatiblePieceAndRot = nullptr;
    if (propotionOfBestFromSecondBest > 1)
        mostCompatiblePieceAndRot = vec[0].first;

    return mostCompatiblePieceAndRot;
}

/************************************************************************/
void PuzzleRL_Solver::CheckAndSetConstantPieces()
//Function input: none
//Function output: none
//Function objective: check, log and set information about constant pieces pieces 
/************************************************************************/
{
    //1. Define 'piecesMap', whose keys are all edge pixels (for dissimilarity computation) of a piece.
    //For each such key, the value is a set of pieces with these edge pixels
    using PiecesEdgesPixelsArray = std::array<std::vector<ExtendedPixelType>, Orientation::eNumOfPossibleOrientations>;

    struct PiecesEdgesPixelsArraysComparator
    {
        bool operator()(const PiecesEdgesPixelsArray& in_lhs, const PiecesEdgesPixelsArray& in_rhs) const
        {
            for (Orientation currOrien = eUp; currOrien < eNumOfPossibleOrientations; currOrien = static_cast<Orientation>(currOrien + 1))
            {
                const std::vector<ExtendedPixelType> currLhsPixelsVec = in_lhs[currOrien];
                const std::vector<ExtendedPixelType> currRhsPixelsVec = in_rhs[currOrien];

                Utilities::LogAndAbortIf(currLhsPixelsVec.size() != currRhsPixelsVec.size(), "currLhsPixelsVec.size() != currRhsPixelsVec.size()");

                for (int32_t i = 0; i < static_cast<int32_t>(currLhsPixelsVec.size()); ++i)
                {
                    for (int32_t color = 0; color < numOfChannels; ++color)
                    {
                        if (currLhsPixelsVec[i][color] != currRhsPixelsVec[i][color])
                            return currLhsPixelsVec[i][color] < currRhsPixelsVec[i][color];
                    }
                }
            }

            return false;
        }
    };

    struct RotatedPiecesComparator
    {
        bool operator()(const PieceObjectAndRot* in_lhs, const PieceObjectAndRot* in_rhs) const
        {
            if (in_lhs->m_pieceObject->m_pieceNumber != in_rhs->m_pieceObject->m_pieceNumber)
                return in_lhs->m_pieceObject->m_pieceNumber < in_rhs->m_pieceObject->m_pieceNumber;
            else if (in_lhs->m_rotation != in_rhs->m_rotation)
                return in_lhs->m_rotation < in_rhs->m_rotation;
            else
                return false;
        }
    };

    using LocalRotatedPiecesSet = std::set<const PieceObjectAndRot*, RotatedPiecesComparator>;

    std::map<PiecesEdgesPixelsArray, LocalRotatedPiecesSet, PiecesEdgesPixelsArraysComparator> piecesMap;

    //2. Insert values to 'piecesMap'
    for (const PieceObjectAndRot* currRotatedPiece : m_rotatedPiecesPool)
    {
        PiecesEdgesPixelsArray currPiecesEdgesPixelsArray;
        for (Orientation currOrien = eUp; currOrien < eNumOfPossibleOrientations; currOrien = static_cast<Orientation>(currOrien + 1))
            currPiecesEdgesPixelsArray[currOrien] = currRotatedPiece->GetEdgePixelsForDissimilarityComputation(currOrien);

        if (piecesMap.count(currPiecesEdgesPixelsArray) == 0)
        {
            const LocalRotatedPiecesSet set = {currRotatedPiece};
            piecesMap.emplace(currPiecesEdgesPixelsArray, set);
        }
        else
        {
            piecesMap.at(currPiecesEdgesPixelsArray).insert(currRotatedPiece);
        }
    }

    //3. Remove all unique pieces, and all sets of two pieces
    for (auto itr = piecesMap.begin(); itr != piecesMap.end();)
    {
        const LocalRotatedPiecesSet currPiecesSet = itr->second;
        //We treat pieces as constant pieces only if there more than 2 such constant pieces
        if (currPiecesSet.size() <= 2) 
            itr = piecesMap.erase(itr);
        else 
            ++itr;
    }

    //4. Remove set of the same piece (or two pieces), rotated in all four rotations
    if (PuzzleType::eType2_UknownRotation == m_config.GetPuzzleType())
    {
        const auto getNumberOfDifferentPiecesNumsInRotatedPiecesSet = [](const LocalRotatedPiecesSet& in_set)
        {
            std::set<int32_t> foundPieceNumbers;
            for (const PieceObjectAndRot* currRotatedPiece : in_set)
            {
                const int32_t currPieceNumber = currRotatedPiece->m_pieceObject->m_pieceNumber;
                if (!Utilities::IsItemInSet(currPieceNumber, foundPieceNumbers))
                {
                    foundPieceNumbers.insert(currPieceNumber);
                }
            }

            return foundPieceNumbers.size();
        };

        for (auto itr = piecesMap.begin(); itr != piecesMap.end();)
        {
            const LocalRotatedPiecesSet currPiecesSet = itr->second;
            if ((currPiecesSet.size() == eNumOfImageRotations && 1 == getNumberOfDifferentPiecesNumsInRotatedPiecesSet(currPiecesSet)) || 
                (currPiecesSet.size() == 2 * eNumOfImageRotations && 2 == getNumberOfDifferentPiecesNumsInRotatedPiecesSet(currPiecesSet))) 
                itr = piecesMap.erase(itr);
            else 
                ++itr;
        }
    }

    //5. Set 'm_constantPiecesVector'
    Utilities::LogAndAbortIf(!m_constantPiecesVector.empty(), "!m_constantPiecesVector.empty()");
    for (const std::pair<PiecesEdgesPixelsArray, LocalRotatedPiecesSet>& currPair : piecesMap)
    {
        const LocalRotatedPiecesSet currPiecesSet = currPair.second;
        m_constantPiecesVector.insert(m_constantPiecesVector.end(), currPiecesSet.begin(), currPiecesSet.end());
    }

    //6. Check that all four edges of constant pieces are identical and update 'm_constantPiecesVector' accordingly
    if (!m_constantPiecesVector.empty())
        CheckAllConstantPiecesEdgesAreIdenticalAndSetConstantPiecesVecToHaveUniquePieces();

    //7. Log constant pieces info
    m_puzzleRL_SolverOutputManager.LogConstantPiecesInfo();
}

/************************************************************************/
void PuzzleRL_Solver::CheckAllConstantPiecesEdgesAreIdenticalAndSetConstantPiecesVecToHaveUniquePieces()
/************************************************************************/
{
    //1. Check that each constant piece four edges are identical
    const PieceObjectAndRot* constantPiece = m_constantPiecesVector[0];
    const std::vector<ExtendedPixelType> topEdge = constantPiece->GetEdgePixelsForDissimilarityComputation(eUp);
    const std::vector<ExtendedPixelType> rightEdge = constantPiece->GetEdgePixelsForDissimilarityComputation(eRight);
    const std::vector<ExtendedPixelType> bottomEdge = constantPiece->GetEdgePixelsForDissimilarityComputation(eDown);
    const std::vector<ExtendedPixelType> leftEdge = constantPiece->GetEdgePixelsForDissimilarityComputation(eLeft);

    const auto areEqualEdges = [&](const std::vector<ExtendedPixelType>& in_lhs, const std::vector<ExtendedPixelType>& in_rhs)
    {
        Utilities::LogAndAbortIf(in_lhs.size() != in_rhs.size(), "currLhsPixelsVec.size() != currRhsPixelsVec.size()");

        for (int32_t i = 0; i < static_cast<int32_t>(in_lhs.size()); ++i)
        {
            for (int32_t color = 0; color < numOfChannels; ++color)
            {
                if (in_lhs[i][color] != in_lhs[i][color])
                    return false;
            }
        }
        return true;
    };

    const auto doesEdgeContainSinglePixel = [&](const std::vector<ExtendedPixelType>& in_edge)
    {
        const ExtendedPixelType firstPixel = in_edge[0];
        for (int32_t i = 0; i < static_cast<int32_t>(in_edge.size()); ++i)
        {
            for (int32_t color = 0; color < numOfChannels; ++color)
            {
                if (firstPixel[color] != in_edge[i][color])
                    return false;
            }
        }
        return true;
    };

    const bool areAllFourConstantPiecesEdgesIdentical =
        areEqualEdges(leftEdge, rightEdge) && areEqualEdges(leftEdge, topEdge) && areEqualEdges(leftEdge, bottomEdge);

    //2. In case that each constant piece four edges are identical
    if (areAllFourConstantPiecesEdgesIdentical)
    {
        //2.1. Check that each edge contains just one pixel type (as the four edge are identical, it si enough to check just for one edge) 
        const bool doEdgesContainSinglePixelType = doesEdgeContainSinglePixel(leftEdge);
        if (!doEdgesContainSinglePixelType)
        {
            Utilities::LogAndAbort("In 'PuzzleRL_Solver::CheckAllConstantPiecesEdgesAreIdenticalAndSetConstantPiecesVecToHaveUniquePieces()': \
                !doEdgesContainSinglePixelType");
        }

        //2.2. Update 'm_constantPiecesVector' so it will contain only the 0 rotation of each piece
        RotatedPiecesVector uniquePieceNumberConstantPiecesVector;
        for (const PieceObjectAndRot* currConstantPiece: m_constantPiecesVector)
        {
            const int32_t currPieceNumber = currConstantPiece->m_pieceObject->m_pieceNumber;
            if (currConstantPiece->m_rotation == m_constantPiecesRotation)
            {
                uniquePieceNumberConstantPiecesVector.push_back(currConstantPiece);
                m_constantPiecesObjectsVector.push_back(currConstantPiece->m_pieceObject);
            }
        }
        m_constantPiecesVector = uniquePieceNumberConstantPiecesVector;
    }
    else
    {
        Utilities::LogAndAbort("In 'PuzzleRL_Solver::CheckAllConstantPiecesEdgesAreIdenticalAndSetConstantPiecesVecToHaveUniquePieces()': \
            constant pieces edges are not identical. This should not happen.");
    }
}

/************************************************************************/
void PuzzleRL_Solver::ApplyConstantPiecesLogicToCompatibility(PiecesTensorType& inout_pieceCompatibilitiesTensor)
/************************************************************************/
{
    //1. Give random values
    std::random_device r;
    std::default_random_engine eng{r()};
    std::uniform_real_distribution<double> uniformDistribution(-4.0, 1.0);

    const PiecesTensorType oldPieceCompatibilitiesTensor = inout_pieceCompatibilitiesTensor; 

    const PieceRelationComputationFuncType ConstantPiecesLogicFunction = 
        [&](const PieceObjectAndRot* in_firstRotatedPiece, const PieceObjectAndRot* in_secondRotatedPiece, const Orientation in_orien)
        {
            double val = oldPieceCompatibilitiesTensor(in_firstRotatedPiece->m_index, in_secondRotatedPiece->m_index, in_orien);

            //firstConstantPiece is a constant piece which is not in_firstRotatedPiece (so we will be able to compute it compatibility with in_firstRotatedPiece)
            const PieceObjectAndRot* firstConstantPiece = m_constantPiecesVector[0]; 
            if (PieceObjectAndRot::ArePieceObjectsIdentical(*in_firstRotatedPiece, *m_constantPiecesVector[0]))
                firstConstantPiece = m_constantPiecesVector[1];

            const double compWithConstantPiece1 = oldPieceCompatibilitiesTensor(in_firstRotatedPiece->m_index, firstConstantPiece->m_index, in_orien);

            //secondConstantPiece is a constant piece which is not in_secondRotatedPiece (so we will be able to compute it compatibility with in_secondRotatedPiece)
            const PieceObjectAndRot* secondConstantPiece = m_constantPiecesVector[0]; 
            if (PieceObjectAndRot::ArePieceObjectsIdentical(*in_secondRotatedPiece, *m_constantPiecesVector[0]))
                secondConstantPiece = m_constantPiecesVector[1];

            const double compWithConstantPiece2 = oldPieceCompatibilitiesTensor(secondConstantPiece->m_index, in_secondRotatedPiece->m_index, in_orien);

            if (1.0 == compWithConstantPiece1 && 1.0 == compWithConstantPiece2)
            {
                val = uniformDistribution(eng);
                val = std::max(val, 0.0);
            }

            return val;
        };

    ApplyFunctionForAllRotatedPiecesRelations(ConstantPiecesLogicFunction, inout_pieceCompatibilitiesTensor, false);

    //2. For type 2 puzzles, zero all compatibility values for unused rotated constant pieces
    if (m_config.GetPuzzleType() == PuzzleType::eType2_UknownRotation)
    {
        const PieceRelationComputationFuncType zeroUsedConstantPiecesRotationFunction = 
        [&](const PieceObjectAndRot* in_firstRotatedPiece, const PieceObjectAndRot* in_secondRotatedPiece, const Orientation in_orien)
        {
            double val = -1.0;
            if ((IsConstantPiece(in_firstRotatedPiece->m_pieceObject) && in_firstRotatedPiece->m_rotation != m_constantPiecesRotation) ||
                (IsConstantPiece(in_secondRotatedPiece->m_pieceObject) && in_secondRotatedPiece->m_rotation != m_constantPiecesRotation))
            {
                val = 0; 
            }
            else
            {
                val = inout_pieceCompatibilitiesTensor(in_firstRotatedPiece->m_index, in_secondRotatedPiece->m_index, in_orien);
            }

            return val;
        };

        ApplyFunctionForAllRotatedPiecesRelations(zeroUsedConstantPiecesRotationFunction, inout_pieceCompatibilitiesTensor, true);
    }
}

/************************************************************************/
void PuzzleRL_Solver::SetAndCreateCurrentOutputAndIterationsFolders(const std::string& in_newOutputFolder) const
/************************************************************************/
{
    SetCurrentOutputFolder(in_newOutputFolder);
    FileSystemUtils::Create_Directory(m_currOutputFolder);
    FileSystemUtils::Create_Directory(RLSolverFileSystemUtils::GetIterationsFolderPath(m_currOutputFolder));
}

#if IS_SAVE_ALL_PAIRS_SUPPORT_COMPUTATION_METHOD()
/************************************************************************/
LabelsVector PuzzleRL_Solver::GetCorrelatedLabelsVec(const Label* in_label) const
//Function input: in_label: RL label
//Function output: all the labels which are correlated with 'in_label', which are all the neighbor labels of 'in_label'
//Function objective: described "Function output"
/************************************************************************/
{
    const LocationRotationLabel* locationRotationLabel = static_cast<const LocationRotationLabel*>(in_label);
    return m_neighborsLabelsMatrix(locationRotationLabel->m_row, locationRotationLabel->m_column);
}

/************************************************************************/
void PuzzleRL_Solver::UpdateNeighborLabelsMatrix(const int32_t in_row, const int32_t in_col, const LabelsVector& in_labelsVec)
//Function input: in_row: row index; in_col: col index; in_labelsVec: vector of labels representing the location (in_row, in_col)
//Function output: none
//Function objective: update all the neighbors of location (in_row, in_col) with 'in_labelsVec'
/************************************************************************/
{
    //Has top neighbor
    if (in_row - 1 >= 0)
    {
        LabelsVector& topNeighborVec = m_neighborsLabelsMatrix(in_row - 1, in_col);
        topNeighborVec.insert(topNeighborVec.end(), in_labelsVec.begin(), in_labelsVec.end());
    }

    //Has bottom neighbore
    if (in_row + 1 < m_puzzleImage.GetNumOfRowPieces())
    {
        LabelsVector& bottomNeighborVec = m_neighborsLabelsMatrix(in_row + 1, in_col);
        bottomNeighborVec.insert(bottomNeighborVec.end(), in_labelsVec.begin(), in_labelsVec.end());
    }

    //Has left neighbor
    if (in_col - 1 >= 0)
    {
        LabelsVector& leftNeighborVec = m_neighborsLabelsMatrix(in_row, in_col - 1);
        leftNeighborVec.insert(leftNeighborVec.end(), in_labelsVec.begin(), in_labelsVec.end());
    }

    //Has right neighbore
    if (in_col + 1 < m_puzzleImage.GetNumOfColPieces())
    {
        LabelsVector& rightNeighborVec = m_neighborsLabelsMatrix(in_row, in_col + 1);
        rightNeighborVec.insert(rightNeighborVec.end(), in_labelsVec.begin(), in_labelsVec.end());
    }
}
#endif

/************************************************************************/
void PuzzleRL_Solver::DoPuzzles(const std::string& in_configXML_Path)
//Function input: in_configXML_Path: configuration XML path
//Function output: none
//Function objective: run puzzle/s according to in_configXML_Path
/************************************************************************/
{
    //1. Technical filesystem tasks
    FileSystemUtils::SetCurrentPath(PUZZLE_RL_SOLVER_DIR);
    PuzzleRL_Solver::ResetAllRL_SolverRunTimeFiles();
    if (SHOULD_LOG_SCREEN_OUTPUT_TO_FILE)
        g_logger.SetAnotherStream(RLSolverFileSystemUtils::GetScreenOutputFilePath());

    //2. Read XML 
    PuzzleRL_SolverConfiguration configs(in_configXML_Path);
    configs.ReadXML();

    //3. Run multiple or single puzzle
    const bool runAllInFolderIsOn = configs.IsOnRunAllFolderMode();
    const int32_t numOfRuns = configs.GetNumOfRuns();
    if (runAllInFolderIsOn || numOfRuns > 1)
        PuzzleRL_Solver::DoMultiplePuzzles(configs, runAllInFolderIsOn, numOfRuns);
    else
        PuzzleRL_Solver::DoSinglePuzzle(CONFIG_XML_PATH.string());
}

/************************************************************************/
void PuzzleRL_Solver::DoMultiplePuzzles(const PuzzleRL_SolverConfiguration& in_configs, const bool in_runAllInFolderIsOn, const int32_t in_definedNumOfRuns)
//Function input: in_configs: run configuration; in_runAllInFolderIsOn: boolean indicating whether we should run all images in a specific folder;
//  in_definedNumOfRuns: integer indicating if we should run each image few times
//Function output: none
//Function objective: run multiple puzzle solvers
/************************************************************************/
{
    PuzzleRL_SolverOutputManager::m_areMultipleRuns = true;

    std::vector<std::string> sourcesVec;

    //1. Get images path strings and related data
    if (in_runAllInFolderIsOn && in_definedNumOfRuns > 1)
    {
        const std::string sourceFolderPath = DATA_FOLDER_PATH.string() + in_configs.GetRunFolderName();
        const std::vector<std::string> folderSourcesVec = FileSystemUtils::GetAllFilesInDirectory(sourceFolderPath);
        for (const std::string& sourceStr: folderSourcesVec)
        {
            for (int32_t i = 0; i < in_definedNumOfRuns; ++i)
            {
                sourcesVec.push_back(sourceStr);
            }
        }                   
    }
    else if (in_runAllInFolderIsOn)
    {
        const std::string sourceFolderPath = DATA_FOLDER_PATH.string() + in_configs.GetRunFolderName();
        sourcesVec = FileSystemUtils::GetAllFilesInDirectory(sourceFolderPath);
    }
    else if (in_definedNumOfRuns > 1)
    {
        const std::string singleSourcePath = DATA_FOLDER_PATH.string() + in_configs.GetSourceName();
        sourcesVec = std::vector<std::string>(in_definedNumOfRuns, singleSourcePath);
    }
    else
    {
        Utilities::LogAndAbort("'DoMultiplePuzzles()' should not have been called");
    }

    //2. Run puzzle solver for all images
    std::vector<RunData> runDataVector;
    const int32_t actualNumOfRuns = static_cast<int32_t>(sourcesVec.size());
    for (int32_t i = 1; i <= actualNumOfRuns; ++i)
    {
        const RunData runData = PuzzleRL_Solver::DoSinglePuzzle(CONFIG_XML_PATH.string(), i, sourcesVec[i - 1]);
        runDataVector.push_back(runData);
    }

    //3. Log output
    PuzzleRL_SolverOutputManager::LogMultiplePuzzlesOutput(runDataVector);
}

/************************************************************************/
RunData PuzzleRL_Solver::DoSinglePuzzle(const std::string& in_configXML_Path, const int32_t in_runNumber,
    const std::string& in_sourcePath)
//Function input: in_configXML_Path: configuration XML path, in_runNumber: the run number (used for cases of more than one run); 
//  in_imagePath: source image path
//Function output: RunData struct, containing run information
//Function objective: run single puzzle and return its run data
/************************************************************************/
{
    //1. Read xml and reset output files
    PuzzleRL_SolverConfiguration configs(in_configXML_Path);
    configs.ReadXML();
    ResetCurrentRunFiles(in_runNumber);

    std::string actualSourcePath;
    if (in_sourcePath.empty())
        actualSourcePath = configs.GetSourcePath();
    else
        actualSourcePath = in_sourcePath;

    //2. Create puzzle image
    const PuzzleImage puzzleImage = CreatePuzzleFromImage(actualSourcePath, configs, in_runNumber);

    //3. Run solver
    RunData finalRunData;
    const bool isNonSquareType2Puzzle = PuzzleType::eType2_UknownRotation == configs.GetPuzzleType() && !puzzleImage.IsSquarePuzzle();

    g_logger.BruteLog("\n##############   " + FileSystemUtils::GetFileNameFromPath(actualSourcePath) + "   ##############\n");

    //4.1. If it's non-square type 2 puzzle - we want to run it twice with different rotations
    if (isNonSquareType2Puzzle)
    {
        finalRunData = RunTwoType2PuzzlesWithDifferentRotations(puzzleImage, configs, actualSourcePath, in_runNumber);
    }
    //4.2. Otherwise, just do single run
    else
    {
        //4.2.1. Create solver and initialize its output folder
        PuzzleRL_Solver solver(puzzleImage, configs, actualSourcePath, ImageRotation::eInvalidRotation, in_runNumber);
        solver.SetCurrentOutputFolder(FileSystemUtils::GetRunOutputFolderPath(in_runNumber));

        //4.2.2. Run solver
        const RL_puzzleSolverAlgorithmRunningInfo algRunningInfo = RunPuzzleSolver(solver);

        //4.2.3. Compute run data and log output
        finalRunData = ComputeSolverRunData(solver, algRunningInfo);

        std::string outputTextStr = solver.m_puzzleRL_SolverOutputManager.GetSinglePuzzleOutput(finalRunData, algRunningInfo.m_initTime, algRunningInfo.m_finalAlc);
        g_logger.BruteLog(outputTextStr);

        PuzzleRL_SolverOutputManager::AddToMultipleRunsSummary(outputTextStr);
    }

    return finalRunData;
}

/************************************************************************/
RL_puzzleSolverAlgorithmRunningInfo PuzzleRL_Solver::RunPuzzleSolver(PuzzleRL_Solver& inout_solver)
//Function input: inout_solver: puzzle solver
//Function output: algorithm running info
//Function objective: run single puzzle and return its algorithm running info
/************************************************************************/
{
    //1. Set variables
    const Utilities::Timer timer;

    //2. Init solver
    inout_solver.Init();
    const long long initTime = timer.GetSecondsPassed();
    g_logger << "Init in " << initTime << " seconds" << std::endl;

    //3. Call relaxation labeling algorithm
    RL_puzzleSolverAlgorithmRunningInfo algRunningInfo(inout_solver.m_allPossibleObjects.size());
    algRunningInfo.m_initTime = initTime;
    inout_solver.SolveProblem(&algRunningInfo);
    algRunningInfo.m_totalTime = timer.GetTime();;

    return algRunningInfo;
}

/************************************************************************/
RunData PuzzleRL_Solver::ComputeSolverRunData(const PuzzleRL_Solver& in_solver, const RL_puzzleSolverAlgorithmRunningInfo& in_algRunningInfo)
//Function input: in_solver: puzzle solver; in_algRunningInfo: solver running info
//Function output: solver run data
//Function objective: described in "Function output"
/************************************************************************/
{
    RunData runData;
    runData.m_runNumber = in_solver.m_runNumber;
    runData.m_numOfIterations = in_algRunningInfo.m_iterationNum;
    runData.m_duration = in_algRunningInfo.m_totalTime;
    runData.m_averageTimePerIteration = in_algRunningInfo.m_averageTimePerIteration;
    runData.m_firstLevelTransDecision = in_algRunningInfo.m_firstLevelTransDecision;
    runData.m_secondLevelTransDecision = in_algRunningInfo.m_secondLevelTransDecision;

    runData.m_directComparison = 100 * in_solver.ComputeSolutionDirectComparisonAndSetRotationSolution();
    runData.m_neighborComparison = 100 * in_solver.ComputeSolutionNeighborComparison();
    runData.m_wasPuzzleCorrectlySolved = (100 == runData.m_directComparison); 

    runData.m_isAssignmentFeasible = in_solver.m_isSolutionAssignmentFeasible; 
    runData.m_assignedPiecesPercentage = 100 * 
        (static_cast<double>(in_solver.m_numOfAssignedPiecesInSolution) / static_cast<double>(in_solver.m_puzzleImage.GetNumOfPieces())); 

    runData.m_isType2Puzzle = PuzzleType::eType2_UknownRotation == in_solver.m_config.GetPuzzleType();

    return runData;
}

/************************************************************************/
RunData PuzzleRL_Solver::RunTwoType2PuzzlesWithDifferentRotations(const PuzzleImage& in_puzzleImage,
    const PuzzleRL_SolverConfiguration& in_config, const std::string& in_sourceImagePath, const int32_t in_runNumber)
//Function input: in_puzzleImage: puzzle image; in_config: puzzle configuration; in_sourceImagePath: source image path; in_runNumber: run number 
//Function output: "winnning "solver run data
//Function objective: run type 2 non-square puzzle twice
/************************************************************************/
{
    Utilities::LogAndAbortIf(in_puzzleImage.IsSquarePuzzle() || PuzzleType::eType2_UknownRotation != in_config.GetPuzzleType(),
        "In 'PuzzleRL_Solver::RunTwoType2PuzzlesWithDifferentRotations()': invalid input");

    //1. Get run output folder
    const std::string runOutputFolderPath = FileSystemUtils::GetRunOutputFolderPath(in_runNumber);

    //2. Run first option (with rotation1)
    const ImageRotation rotation1 = ImageRotation::e_0_degrees;
    const std::string rotationStr1 = "rotation " + GetRotationString(rotation1, false);
    RL_puzzleSolverAlgorithmRunningInfo algRunningInfo1;
    cv::Mat solutionImage1;
    std::string solverTextOutput1;
    RunData runData1;
    {
        PuzzleRL_Solver solver1(in_puzzleImage, in_config, in_sourceImagePath, rotation1, in_runNumber);
        const std::string solver1OutputFolder = FileSystemUtils::GetPathInFolder(runOutputFolderPath, rotationStr1 + "/"); 
        solver1.SetAndCreateCurrentOutputAndIterationsFolders(solver1OutputFolder);

        const RL_puzzleSolverAlgorithmRunningInfo algRunningInfo = RunPuzzleSolver(solver1);
        algRunningInfo1.Set(algRunningInfo);
        runData1 = ComputeSolverRunData(solver1, algRunningInfo1);
        solverTextOutput1 = solver1.m_puzzleRL_SolverOutputManager.GetSinglePuzzleOutput(runData1, algRunningInfo1.m_initTime, algRunningInfo1.m_finalAlc);
        solutionImage1 = solver1.GetAssemblyImageFromBinaryLabeling(*(solver1.m_pSolution));
    }

    Utilities::Log(PuzzleRL_SolverOutputManager::Get1stLevelTitleString("Done with " + rotationStr1));

    //3. Run second option (with rotation2)
    const ImageRotation rotation2 = ImageRotation::e_90_degrees;
    const std::string rotationStr2 = "rotation " + GetRotationString(rotation2, false);
    RL_puzzleSolverAlgorithmRunningInfo algRunningInfo2;
    cv::Mat solutionImage2;
    std::string solverTextOutput2;
    RunData runData2;
    {
        PuzzleRL_Solver solver2(in_puzzleImage, in_config, in_sourceImagePath, rotation2, in_runNumber);
        const std::string solver2OutputFolder = FileSystemUtils::GetPathInFolder(runOutputFolderPath, rotationStr2 + "/"); 
        solver2.SetAndCreateCurrentOutputAndIterationsFolders(solver2OutputFolder);

        const RL_puzzleSolverAlgorithmRunningInfo algRunningInfo = RunPuzzleSolver(solver2);
        algRunningInfo2.Set(algRunningInfo);
        runData2 = ComputeSolverRunData(solver2, algRunningInfo2);
        solverTextOutput2 = solver2.m_puzzleRL_SolverOutputManager.GetSinglePuzzleOutput(runData2, algRunningInfo2.m_initTime, algRunningInfo2.m_finalAlc);
        solutionImage2 = solver2.GetAssemblyImageFromBinaryLabeling(*(solver2.m_pSolution));
    }

    Utilities::Log(PuzzleRL_SolverOutputManager::Get1stLevelTitleString("Done with " + rotationStr2));

    //4. Set winning rotation
    std::string winningRotationStr;
    const cv::Mat* winningSolutionImage = nullptr;
    const RL_puzzleSolverAlgorithmRunningInfo* winningAlgInfo = nullptr;
    RunData* winningRunData = nullptr;
    std::string* winningSolverTextOutput = nullptr;

    std::string losingRotationStr;
    const RL_puzzleSolverAlgorithmRunningInfo* losingAlgInfo = nullptr;
    std::string* losingSolverTextOutput = nullptr;
    
    if (algRunningInfo1.m_finalAlc > algRunningInfo2.m_finalAlc)
    {
        winningRotationStr = rotationStr1;
        winningSolutionImage = &solutionImage1;
        winningAlgInfo = &algRunningInfo1;
        winningRunData = &runData1;
        winningSolverTextOutput = &solverTextOutput1;

        losingRotationStr = rotationStr2;
        losingAlgInfo = &algRunningInfo2;
        losingSolverTextOutput = &solverTextOutput2;
    }
    else if (algRunningInfo1.m_finalAlc < algRunningInfo2.m_finalAlc)
    {
        winningRotationStr = rotationStr2;
        winningSolutionImage = &solutionImage2;
        winningAlgInfo = &algRunningInfo2;
        winningRunData = &runData2;
        winningSolverTextOutput = &solverTextOutput2;

        losingRotationStr = rotationStr1;
        losingAlgInfo = &algRunningInfo1;
        losingSolverTextOutput = &solverTextOutput1;
    }
    else
    {
        Utilities::LogAndAbort("algRunningInfo1.m_finalAlc == algRunningInfo2.m_finalAlc");
    }

    //6. Log winning rotation data
    g_logger.BruteLog("Winning rotation is " + winningRotationStr + "\n");
    g_logger.BruteLog("since " + winningRotationStr + " > " +  losingRotationStr+
        " (" + std::to_string(winningAlgInfo->m_finalAlc) + " > " + std::to_string(losingAlgInfo->m_finalAlc) + ")\n");
    g_logger.BruteLog("Winning rotation runtime is " + winningRotationStr + "\n");

    ImageUtils::WriteImage(RLSolverFileSystemUtils::GetSolutionImagePath(runOutputFolderPath), *winningSolutionImage);

    g_logger.BruteLog("\nLosing run info:\n" + *losingSolverTextOutput);
    g_logger.BruteLog("Winning run info:\n" + *winningSolverTextOutput);

    //7. Add time of two runs
    winningRunData->m_type2RunTimeForTwoRotations = runData1.m_duration + runData2.m_duration;
    const std::string timeForTwoRunsStr = Utilities::Timer::GetTimeString(winningRunData->m_type2RunTimeForTwoRotations);
    g_logger.BruteLog("Two runs took: " + timeForTwoRunsStr + "\n");

    PuzzleRL_SolverOutputManager::AddToMultipleRunsSummary(*winningSolverTextOutput);

    return *winningRunData;
}

/************************************************************************/
void PuzzleRL_Solver::ResetAllRL_SolverRunTimeFiles()
//Function input: none
//Function output: none
//Function objective: delete all content from OUTPUT_DIRECTORY_PATH directory
/************************************************************************/
{
    FileSystemUtils::DeleteDirectoryRecursively(OUTPUT_DIRECTORY_PATH.string());
    FileSystemUtils::Create_Directory(OUTPUT_DIRECTORY_PATH.string());
}

/************************************************************************/
void PuzzleRL_Solver::ResetCurrentRunFiles(const int32_t in_runNumber)
//Function input: in_runNumber: run number
//Function output: none
//Function objective: reset and create all directories for run number 'in_runNumber'
/************************************************************************/
{
    const std::string runOutputFolderPath = FileSystemUtils::GetRunOutputFolderPath(in_runNumber); 
    if (-1 != in_runNumber)
        FileSystemUtils::Create_Directory(runOutputFolderPath);

    FileSystemUtils::Create_Directory(RLSolverFileSystemUtils::GetIterationsFolderPath(runOutputFolderPath));
}

/************************************************************************/
PuzzleImage PuzzleRL_Solver::CreatePuzzleFromImage(const std::string& in_srcPath, 
    const PuzzleRL_SolverConfiguration& in_config, const int32_t in_runNumber)
//Function input: in_srcPath: path of source image; in_config: run configuration; in_runNumber: run number
//Function output: PuzzleImage instance
//Function objective: create PuzzleImage instance from the image in in_srcPath
/************************************************************************/
{
    //1. Set variables and abort if needed
    const ImageInputData& imageInputData = in_config.GetImageInputData();

    //2. Read image
    cv::Mat image = ImageUtils::ReadImage(in_srcPath);

    //3. Generate puzzle (puzzleImage contains all the puzzle information)
    const PuzzleImage puzzleImage = PuzzleRL_Solver::GeneratePuzzleImage(image, imageInputData.GetPieceSize(), in_runNumber,
        in_config.GetPuzzleType());

    return puzzleImage;
}

/************************************************************************/
PuzzleImage PuzzleRL_Solver::GeneratePuzzleImage(const cv::Mat& in_image, const int32_t in_pieceSize, const int32_t in_runNumber,
    const PuzzleType in_puzzleType)
//Function input: in_image: source image; in_pieceSize: piece size in pixel (for example, if in_pieceSize=3, then each piece is a 3x3 pixels square)
//  in_runNumber: run number; in_puzzleType: puzzle type;
//Function output: PuzzleImage instance
//Function objective: generate puzzle from in_image
/************************************************************************/
{
    //1. Truncate image
    cv::Mat truncatedImage = PuzzleRL_Solver::TruncateImageToFitPieceSize(in_image, in_pieceSize);

    //3. Create puzzle image
    PuzzleImage puzzleImage = PuzzleImage::CreateAndInitPuzzleImage(truncatedImage, in_pieceSize, in_puzzleType, PRINT_PIECE_NUMBERS_IN_IMAGE);

    //4. Save the original and puzzle image
    ImageUtils::WriteImage(RLSolverFileSystemUtils::GetOriginalImagePath(in_runNumber), puzzleImage.GetOriginalImage());
    
    ImageUtils::WriteImage(RLSolverFileSystemUtils::GetPuzzleImagePath(in_runNumber), puzzleImage.GetImage());

    return puzzleImage;
}

/********************************************************************/
cv::Mat PuzzleRL_Solver::TruncateImageToFitPieceSize(const cv::Mat& in_image, const int32_t in_pieceSize)
//Function input: in_image: source image; in_pieceSize: piece size in pixels (for example, if in_pieceSize=3, then each piece is a 3x3 pixels square)
//Function output: truncated image
//Function objective: truncate image so it fits exactly to piece size
/************************************************************************/
{
    //1. Set numbers of rows and column to fit in_pieceSize
    const int numOfRowsAfterTruncation = in_image.rows - (in_image.rows % in_pieceSize);
    const int numOfColsAfterTruncation = in_image.cols - (in_image.cols % in_pieceSize);

    //2. Truncate image and log
    const cv::Mat truncatedImage = ImageUtils::TruncatedImage(in_image, numOfRowsAfterTruncation, numOfColsAfterTruncation);
    
    if (truncatedImage.rows != in_image.rows || truncatedImage.cols != in_image.cols)
    {
        g_logger << "Image was truncated from size of " << in_image.rows << "x" << in_image.cols << 
            " to size of " << truncatedImage.rows << "x" << truncatedImage.cols << std::endl;
    }

    return truncatedImage;
}

/************************************************************************/
int32_t PuzzleRL_Solver::GetMaxRL_Iterations(const PuzzleType in_puzzleType, const PuzzleImage& in_puzzleImage)
//Function input: in_puzzleType: puzzle type; in_puzzleImage: puzzle image
//Function output: integer which represents the max number of relaxation labeling iterations allowed
//Function objective: described in "Function output"
/************************************************************************/
{
    int32_t maxNumberOfIterations = -1;
    
    if (PuzzleType::eType1_NoRotation == in_puzzleType)
        maxNumberOfIterations = in_puzzleImage.GetNumOfPieces() * 3;
    else
        maxNumberOfIterations = in_puzzleImage.GetNumOfPieces() * 6; 

    maxNumberOfIterations = std::max(1000, maxNumberOfIterations);

    return maxNumberOfIterations;
}
