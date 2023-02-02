#pragma once

#include "PuzzleRL_SolverOutputManager.h"
#include "RelaxationLabelingSolverBase.h"
#include "PuzzlePieceObject.h"
#include "LocationRotationLabel.h"
#include "PuzzleImage.h"
#include "PuzzleRL_SolverConfiguration.h"
#include "PiecesBlockTranslationManager.h"


class PuzzleRL_Solver : public RelaxationLabelingSolverBase
{
    friend class PuzzleRL_SolverOutputManager;
    friend class PiecesBlockTranslationManager;
    
//public member functions
public:
    PuzzleRL_Solver(const PuzzleImage& in_puzzleImage, const PuzzleRL_SolverConfiguration& in_config, const std::string& in_sourceImagePath, 
        const ImageRotation& in_pieceRotationForType2Puzzles = ImageRotation::eInvalidRotation, const int32_t in_runNumber = -1);
    ~PuzzleRL_Solver() override;

//Overriden public member functions
public:
    void Init() override;

//Overriden protected member functions
protected:
    Labeling GetInitLabeling() const override {return *m_initLabeling;}
    void OutputLabelingDuringAlg(const Labeling& in_labeling, const int32_t in_iterationNum) const override;
    void OutputSolutionLabeling(const Labeling& in_labeling) const override;
    bool HasSolutionChangedInIteration(const Labeling& in_newLabeling) const override;
    void PostMakeLabelingConsistent(const Labeling& in_labeling) override;
    bool ShouldDoRandomAnchoringAndThenBreak(const std::set<int32_t>& in_anchoredRowsSet, std::string& out_breakAfterAnchoringStr) const override;
    RowColInfoSet GetSetOfEntriesToBeAnchored(const Labeling& in_labeling, const AnchoringData& in_anchoringData) const override;
    Labeling DoAnchoring(const Labeling& in_labeling, const RowColInfoSet& in_allEntriesToBeAnchoredSet) const override;
    bool DoTranslationAndPossiblyBranchAlg(Labeling& inout_labeling, const RowColInfoSet& in_newJustAnchoredEntries, 
        RL_algorithmRunningInfo& inout_algRunningInfo, bool& out_shouldBreakAfterTranslation) const override;
    bool AreTranslationsDone() const override {return m_piecesBlockTranslationManager.AreTranslationsDone();}
    void DoPostConvergenceAnchoring(Labeling& inout_labeling, AnchoringData& inout_anchoringData) const override;
    void AnchorNonAnchoredObjects(Labeling& inout_labeling, AnchoringData& inout_anchoringData) const override;
    RowColInfo GetMaxEntryInNonAnchoredRow(const Labeling& in_labeling, const AnchoringData& in_anchoringData) const override;
    bool VerifyAnchoringLegality(const RowColInfo& in_newEntryToBeAnchored, const AnchoringData& in_anchoringData) const override;

//Protected member functions and data
protected:
    void DeleteObjectsPool();
    void DeleteLabelsPool();
    void DeletePieceObjectsAndRotationsObjects();

    void InitAllObjectsPool();
    void InitAllLabelsPool();
    void InitPieceAndRotationsData();

    double ComputeSolutionDirectComparisonAndSetRotationSolution() const;
    double ComputeSolutionDirectComparison(const PieceNumbersAndRotationsMatrix& in_assignmentMatrix,
        ImageRotation& out_solutionRotation, RowColInfoSet& out_wrongAssignedPieceCoordsSet) const;
    double ComputeSolutionNeighborComparison() const;
  
    void InitSolverCompatibility();

    double GetSolverCompatibilityValue(const PuzzlePieceObject* in_firstPiece, const LocationRotationLabel* in_firstLocRot,
        const PuzzlePieceObject* in_secondPiece, const LocationRotationLabel* in_secondLocRot) const;
    double GetSolverCompatibilityValueWithPiecesComp(const PuzzlePieceObject* in_firstPiece, const LocationRotationLabel* in_firstLocRot,
        const PuzzlePieceObject* in_secondPiece, const LocationRotationLabel* in_secondLocRot, const PiecesTensorType& in_piecesComp) const;
   
    void ComputePieceDissimilarities();
    void ComputePieceDissimilaritiesForDissimilarityType(const DissimilarityType in_dissimilarityType);
    void ComputePieceCompatibilitiesBasedOnDissimilarities();
    void ComputePieceCompatibilitiesBasedOnDissimilaritiesForDissimilarityType(const DissimilarityType in_dissimilarityType);

    void ComputePieceCompatibilitiesBasedOnDissimilaritiesForDissimilarityType(PiecesTensorType& out_pieceCompatibilitiesTensor, 
        const PiecesTensorType& in_pieceDissimilaritiesTensor, const DissimilarityType in_dissimilarityType,
        const double& in_kappaCoeff) const;

    void ApplyNormalizationAndSymmetrizationAndThresholding(PiecesTensorType& inout_pieceCompatibilitiesTensor) const;

    void ApplyNormalizationAndSymmetrization(PiecesTensorType& inout_pieceCompatibilitiesTensor) const;
    void ApplyNormalizationAndSymmetrization(PiecesTensorType& inout_pieceCompatibilitiesTensor, 
        const double& in_minCompatibility, const double& in_maxCompatibility) const;

    void ApplyThresholding(PiecesTensorType& inout_pieceCompatibilitiesTensor) const;

    void VerifyCompatibilitySymmetry() const;

    using PieceRelationComputationFuncType = std::function<double(const PieceObjectAndRot*, const PieceObjectAndRot*, const Orientation)>;
    using PieceRelationVerifcationFuncType = std::function<void(const PieceObjectAndRot*, const PieceObjectAndRot*, const Orientation)>;
    void ApplyFunctionForAllRotatedPiecesRelations(const PieceRelationComputationFuncType& in_function, PiecesTensorType& inout_piecesTensor, 
        const bool in_areValuesSymmetric) const;
    void ApplyFunctionForAllRotatedPiecesRelationsNonParallel(const PieceRelationComputationFuncType& in_function, PiecesTensorType& inout_piecesTensor, 
        const bool in_areValuesSymmetric);
    void ApplyFunctionForAllRotatedPiecesRelations(const PuzzlePieceObject* in_puzzlePiece, const PieceRelationComputationFuncType& in_function, 
        PiecesTensorType& inout_piecesTensor, const bool in_areValuesSymmetric) const;

    void ApplyVerificationFunctionForAllRotatedPiecesRelations(const PieceRelationVerifcationFuncType& in_function, const bool in_areValuesSymmetric) const;

    void SetValuesToPiecesTensor(const PieceObjectAndRot* in_firstRotatedPiece, const PieceObjectAndRot* in_secondRotatedPiece,
        const Orientation in_orien, const double in_val, PiecesTensorType& inout_piecesTensor, const bool in_isSymmetricRelation) const;

    using PieceDissimilarityFuncType = std::_Mem_fn<double(PuzzleRL_Solver::*)(const PieceObjectAndRot*, const PieceObjectAndRot*, const Orientation) const>;

    double GetImprovedMGC_Dissimilarity(const PieceObjectAndRot* in_xi, const PieceObjectAndRot* in_xj, const Orientation in_orien) const;
    void GetDlrAndDrlForImprovedMGC(const PieceObjectAndRot* in_leftPiece, const PieceObjectAndRot* in_rightPiece, double& out_dlr, double& out_drl) const;
    void GetDlrDerivativeAndDrlDerivativeForImprovedMGC(const PieceObjectAndRot* in_leftPiece, const PieceObjectAndRot* in_rightPiece, 
        double& out_dlrDerivative, double& out_drlDerivative) const;
    PieceDissimilarityFuncType GetDissimilarityFunction(const DissimilarityType in_dissType) const;

    int32_t GetOneBasedIndexInDissimilaritiesOrder(const PieceObjectAndRot* in_rotatedPiece, const Orientation in_orien,
        const PieceObjectAndRot* in_rotatedPieceToLookFor, const DissimilarityType in_dissimilarityType) const;

    void SetDissimilarityInfo(const DissimilarityType in_dissimilarityType);
    void ComputeDissimilarityInfo(const PieceObjectAndRot* in_rotatedPiece, const Orientation in_orien, 
        const PiecesTensorType& in_piecesDissimilaritiesTensor, DissimilarityInfo& out_dissimilarityInfo) const;
    const DissimilarityInfo& GetDissimilarityInfo(const PieceObjectAndRot* in_rotatedPiece, const Orientation in_orien, const DissimilarityType in_dissimilarityType) const;
    const PieceObjectAndRot* GetPieceToLookForInDissimilarityInfo(const PieceObjectAndRot* in_pieceObjAndRot, const PieceObjectAndRot* in_pieceObjAndRotToLookFor) const;

    void SetInitialLabeling();
    void SetDefaultInitLabeling();

    void ManipulateInitialLabelingForType2Puzzles();
    void SetLabelingManipulationForType2PuzzlesBySinglePieceVariables();

    void ManipulateLabelingForType2PuzzlesBySinglePiece(Labeling& inout_labeling, const int32_t in_numOfAvailableLocations) const;

    void DoLabelingManipulationForType2PuzzlesBySinglePiece(Labeling& inout_labeling, 
        const PuzzlePieceObject* in_selectedPiece, const ImageRotation& in_allowedRotation, const int32_t in_numOfAvailableLocations) const;

    const PuzzlePieceObject* GetSinglePieceWithMaxSurroundingComp() const;

    void SetLabelingForPieceAndAllowedRotations(Labeling& inout_labeling, const PuzzlePieceObject* in_PuzzlePieceObj, 
        const ImageRotation& in_allowedRotation, const int32_t in_numOfAvailableLocations) const;

    const PuzzlePieceObject* GetPuzzlePieceObjectByIndex(const int32_t& in_index) const;
    const PuzzlePieceObject* GetPuzzlePieceObjectByPieceNumber(const int32_t& in_pieceNumber) const;
    const LocationRotationLabel* GetLocationRotationLabelByCoordAndRot(const Utilities::RowColInfo& in_coord,
        const ImageRotation& in_rotation) const;
    LocationRotationLabelsVector GetAllLocationRotationLabelsForLocation(const Utilities::RowColInfo& in_coord) const;

    const PieceObjectAndRot* GetRotatedPieceFromPieceNumberAndRotation(const PieceNumberAndRotation& in_pieceNumAndRotation) const;
    const PieceObjectAndRot* GetRotatedPieceFromPieceNumberAndRotation(const int32_t in_pieceNumber, const ImageRotation in_rotation) const;
    const PieceObjectAndRot* GetRotatedPieceFromPieceObjectAndRotation(const PuzzlePieceObject* in_pieceObj, const ImageRotation in_rotation) const
        {return GetRotatedPieceFromPieceNumberAndRotation(in_pieceObj->m_pieceNumber, in_rotation);}
    
    int32_t GetRotatedPieceIndex(const PuzzlePieceObject* in_puzzlePieceObj, const ImageRotation in_rotation) const
        {return m_rotatedPieceIndexesMatrix(in_puzzlePieceObj->m_index, static_cast<int32_t>(in_rotation));}

    const PieceObjectAndRot* RotateAndGetPiece(const PieceObjectAndRot* in_rotatedPiece, const ImageRotation in_rotationToApply, const bool in_mayGetPieceNotInPool = false) const;
    int32_t GetNumOfRotatedPieces() const { return static_cast<int32_t>(m_rotatedPieceIndexesMatrix.rows() * m_rotatedPieceIndexesMatrix.cols()); }

    void GetRotatedPieceAndCoordinateFromLabelingEntry(const RowColInfo& in_labelingEntry, const PieceObjectAndRot*& out_rotatedPiece, 
        RowColInfo& out_coord) const;

    cv::Mat GetAssemblyImageFromBinaryLabeling(const Labeling& in_binaryLabeling) const;
    PieceNumbersAndRotationsMatrix GetAssignmentMatrixFromBinaryLabeling(const Labeling& in_binaryLabeling) const;
    void SetRotatedPieceInLocation(const PuzzlePieceObject* in_puzzlePieceObject, const LocationRotationLabel* in_locRotLabel, cv::Mat& out_image) const;

    bool AreBestBuddies(const PieceObjectAndRot* in_firstPiece, const PieceObjectAndRot* in_secondPiece, const Orientation in_orien,
        const PiecesTensorType& in_pieceCompatibilitiesTensor) const;

    const PieceObjectAndRot* GetMostCompatiblePiece(const PieceObjectAndRot* in_pieceObjAndRot, const Orientation in_orien, const PiecesTensorType& in_pieceCompatibilitiesTensor) const;

    void CheckAndSetConstantPieces();
    void CheckAllConstantPiecesEdgesAreIdenticalAndSetConstantPiecesVecToHaveUniquePieces();
    bool IsConstantPiece(const PieceObjectAndRot* in_rotatedPiece) const {return Utilities::IsItemInVector(in_rotatedPiece, m_constantPiecesVector);};
    bool IsConstantPiece(const PuzzlePieceObject* in_piece) const {return Utilities::IsItemInVector(in_piece, m_constantPiecesObjectsVector);};
    void ApplyConstantPiecesLogicToCompatibility(PiecesTensorType& in_pieceCompatibilitiesTensor);

    bool AreAllNonAnchoredPiecesConstantPieces(const std::set<int32_t>& in_anchoredRowsSet) const;
    BooleansMatrix GetAnchoredPiecesMatrix(const AnchoringData& in_anchoringData) const;
    const PieceObjectAndRot* GetAnchoredRotatedPieceInCoord(const RowColInfo& in_coord, const AnchoringData& in_anchoringData) const;

    void SortAnchoringCandidates(std::vector<LabelingEntryInfo>& inout_candidateEntriesForAnchoring, const AnchoringData& in_anchoringData) const;

    void BranchDueToTranslationDilemma(Labeling& inout_labeling, RL_algorithmRunningInfo& inout_algRunningInfo, 
                                       const TranslationMode& in_transDilemma) const;
    void ApplyTranslationDecisionAndPrePostActions(const TranslationDecision& in_transDecision, const TranslationMode& in_transDilemma, Labeling& inout_labeling, 
        RL_puzzleSolverAlgorithmRunningInfo& inout_algRunningInfo, double &out_finalALC) const;
    void ApplyTranslationDecision(const TranslationDecision& in_transDecision, const TranslationMode& in_transDilemma,
        Labeling& inout_labeling, RL_puzzleSolverAlgorithmRunningInfo& inout_puzzleAlgRunningInfo) const;
    void MakeLabelingConsistentWithTranslationDecision(Labeling& inout_labeling, const TranslationDecision& in_transDecision, 
        const TranslationMode& in_transDilemma, RL_algorithmRunningInfo& inout_algRunningInfo) const;
    void SetCurrOutputFolderBeforeTranslationBranching(const TranslationDecision& in_transDecision, const TranslationMode& in_transDilemma) const;

    double GetAverageLocalConsistencyWithNonManipulatedCompatibility(const Labeling& in_labeling) const;
    SupportType GetCompatibilitySupportWithNonManipulatedCompatibility(const Labeling& in_labeling) const;

    std::string GetSolutionImagePath() const {return RLSolverFileSystemUtils::GetSolutionImagePath(m_currOutputFolder);}
    std::string GetIterationImageFileName(const int32_t& in_iterationNum) const
        {return RLSolverFileSystemUtils::GetIterationImageFileName(in_iterationNum, m_currOutputFolder);}

    void SetAndCreateCurrentOutputAndIterationsFolders(const std::string& in_newOutputFolder) const;

#if IS_SAVE_ALL_PAIRS_SUPPORT_COMPUTATION_METHOD()
    LabelsVector GetCorrelatedLabelsVec(const Label* in_label) const override;
    void UpdateNeighborLabelsMatrix(const int32_t in_row, const int32_t in_col, const LabelsVector& in_labelsVec);

    //m_neighborsLabelsMatrix stores the neighbor labels of each entry in the puzzle (for example, in entry (i, j) m_neighborsLabelsMatrix will store 
    //all the labels that are neighbor with (i, j))
    using LabelVectorsMatrix = Eigen::Matrix<LabelsVector, Eigen::Dynamic, Eigen::Dynamic>;
    LabelVectorsMatrix m_neighborsLabelsMatrix;
#endif

    Labeling* m_initLabeling = nullptr;

    LocationRotationLabelsVector m_locationRotationLabelsPool;
    PuzzlePieceObjectsVector m_puzzlePieceObjectsPool;

    const PuzzleImage m_puzzleImage;
    const PuzzleRL_SolverConfiguration m_config;
    const std::string m_sourceImagePath;

    mutable ImageRotation m_solutionRotation = ImageRotation::eInvalidRotation;
    PieceNumbersAndRotationsMatrix m_finalPieceAssignmentMatrix;
    bool m_isSolutionAssignmentFeasible = false;
    int32_t m_numOfAssignedPiecesInSolution = 0;

    mutable Eigen::MatrixXi m_currentTopLabelsMatrix;
    int32_t m_numOfPossiblePieceRotations = 1;

    Eigen::MatrixXi m_rotatedPieceIndexesMatrix;
    PuzzleRL_SolverOutputManager m_puzzleRL_SolverOutputManager;

    PiecesBlockTranslationManager m_piecesBlockTranslationManager;
    
    RotatedPiecesVector m_rotatedPiecesPool;
    RotatedPiecesVector m_rotatedPiecesExtraPool;

    using LocationRotationLabelsVectorMatrix = Eigen::Matrix<LocationRotationLabelsVector, Eigen::Dynamic, Eigen::Dynamic>;
    LocationRotationLabelsVectorMatrix m_locationRotationLabelsMatrix;

    //'m_constantPiecesVector' contains pieces with identical pixel vectors on all four edges, but the "middle" pixels are not necessary identical among them
    //Thus, there might be scenarios in which the binary performance measure (the "solved perfectly" measure) will be false, but direct comparison will be 100%
    //(it happens since the "solved perfectly" measure is computed by pixelwise comparison of ground truth and solution image)
    //For type 2 puzzles, 'm_constantPiecesVector' contains only one rotation of each constant piece
    RotatedPiecesVector m_constantPiecesVector;
    PuzzlePieceObjectsVector m_constantPiecesObjectsVector;
    ImageRotation m_constantPiecesRotation = e_0_degrees;

    const PuzzlePieceObject* m_firstRotationDeterminedPiece = nullptr;
    ImageRotation m_allowedRotationForFirstRotationDeterminedPiece;

    //m_dissimilarityInfoMatrixArr is used to store dissimilarity info, which makes some running time optimizations possible
    using DissimilarityInfoMatrix = Eigen::Matrix<DissimilarityInfo, Eigen::Dynamic, Eigen::Dynamic>;
    mutable std::array<DissimilarityInfoMatrix, static_cast<int32_t>(DissimilarityType::eNumOfDissimilarityTypes)> m_dissimilarityInfoMatrixArr;

    double GetCompatibilityValue(const Object* in_object1, const Label* in_label1,
        const Object* in_object2, const Label* in_label2) const override
    {
        const double compVal = GetSolverCompatibilityValue(static_cast<const PuzzlePieceObject*>(in_object1),
            static_cast<const LocationRotationLabel*>(in_label1), static_cast<const PuzzlePieceObject*>(in_object2),
            static_cast<const LocationRotationLabel*>(in_label2));

        return compVal;
    }

    PiecesTensorType m_piecesDissimilarities;
    PiecesTensorType m_piecesCompatibilities;
    PiecesTensorType m_piecesCompatibilitiesNonManipulated;
                                                     
    const double m_kParamForMethod2CompatibilityComputation = 0.03;

//Public static functions
public:
    static void DoPuzzles(const std::string& in_configXML_Path);
    static RunData DoSinglePuzzle(const std::string& in_configXML_Path, const int32_t in_runNumber = -1, const std::string& in_imagePath = "");
    static void DoMultiplePuzzles(const PuzzleRL_SolverConfiguration& in_configs, const bool in_runAllInFolderIsOn, const int32_t in_definedNumOfRuns);

    static RL_puzzleSolverAlgorithmRunningInfo RunPuzzleSolver(PuzzleRL_Solver& inout_solver);
    static RunData ComputeSolverRunData(const PuzzleRL_Solver& in_solver, const RL_puzzleSolverAlgorithmRunningInfo& in_algRunningInfo);

    static RunData RunTwoType2PuzzlesWithDifferentRotations(const PuzzleImage& in_puzzleImage, 
        const PuzzleRL_SolverConfiguration& in_config, const std::string& in_sourceImagePath, const int32_t in_runNumber);

//Protected static functions
protected:
    static void ResetAllRL_SolverRunTimeFiles();
    static void ResetCurrentRunFiles(const int32_t in_runNumber);

    static PuzzleImage CreatePuzzleFromImage(const std::string& in_srcPath, const PuzzleRL_SolverConfiguration& in_config, const int32_t in_runNumber);

    static PuzzleImage GeneratePuzzleImage(const cv::Mat& in_image, const int32_t in_pieceSize, const int32_t in_runNumber,
        const PuzzleType in_puzzleType);

    static cv::Mat TruncateImageToFitPieceSize(const cv::Mat& in_image, const int32_t in_pieceSize);
    static int32_t GetMaxRL_Iterations(const PuzzleType in_puzzleType, const PuzzleImage& in_puzzleImage);
};