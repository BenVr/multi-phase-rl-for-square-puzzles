#include "PuzzleRL_SolverOutputManager.h"
#include "PuzzleRL_Solver.h"
#include "FileSystemUtils.h"

extern Utilities::Logger g_logger;

bool PuzzleRL_SolverOutputManager::m_areMultipleRuns = false;

/************************************************************************/
void PuzzleRL_SolverOutputManager::OutputAllPiecesSeparately() const
/************************************************************************/
{
    const std::string piecesFolderPath = RLSolverFileSystemUtils::GetPiecesFolderPath(m_puzzleSolver->m_runNumber);
    if (!FileSystemUtils::DoesFileExist(piecesFolderPath))
    {
        FileSystemUtils::Create_Directory(piecesFolderPath);

        for (const PuzzlePieceObject* puzzlePieceObject : m_puzzleSolver->m_puzzlePieceObjectsPool)
        {
            std::stringstream fileName;
            fileName << piecesFolderPath << puzzlePieceObject->GetStringForPiecePrint();
            ImageUtils::WriteImage(fileName.str(), puzzlePieceObject->GetPiece());
        }
    }
}

/************************************************************************/
std::string PuzzleRL_SolverOutputManager::GetSinglePuzzleOutput(const RunData& in_runData, const long long& in_timeInit, const double in_finalALC) const
//Function input: in_runData: run data; in_solver: solver used in this run; in_timeInit: time for algorithm init; in_finalALC: final alc value
//Function output: text string with single run data
//Function objective: described in "Function output"
/************************************************************************/
{
    std::stringstream str;

    const PuzzleImage& puzzleImage = m_puzzleSolver->m_puzzleImage;
    const PuzzleRL_SolverConfiguration& config = m_puzzleSolver->m_config;

    //1. log performance measures
    if (in_runData.m_wasPuzzleCorrectlySolved)
        str << std::endl << "Puzzle was correctly solved: ";
    else
        str << std::endl << "Puzzle was NOT correctly solved: ";

    str << "direct comparison is " << in_runData.m_directComparison << "%" <<
       ", neighbor comparison is " << in_runData.m_neighborComparison << "%" << std::endl;

    //2. Log solution feasibility
    if (in_runData.m_isAssignmentFeasible)
        str << "Solution is feasible";
    else
        str << "Solution is not feasible (" << in_runData.m_assignedPiecesPercentage << "% of pieces are assigned)";

    //3. Log final average local consistency
    str << ". Final ALC is " << in_finalALC << std::endl;

    //4. Log translation info
    if (TranslationDecision::eInvalidTranslationDecision != in_runData.m_firstLevelTransDecision)
    {
        str << "Result achieved with the following translation decisions: " << GetTranslationDecisionString(in_runData.m_firstLevelTransDecision);

        if (TranslationDecision::eInvalidTranslationDecision != in_runData.m_secondLevelTransDecision)
            str << ", and then " << GetTranslationDecisionString(in_runData.m_secondLevelTransDecision);
        else
            str << ", with no second translation decision";

        str << "." << std::endl; 
    }
    else
    {
        str << "No translation decisions taken." << std::endl; 
    }

    //5. Log compatibility type, puzzle type, num of piece, piece size, puzzle dimension
    str << GetCompatibilityTypeString() << "." << std::endl
        << RLSolverGeneralUtils::GetPuzzleTypeString(config.GetPuzzleType()) << ": "
        << puzzleImage.GetNumOfPieces()
        << " pieces puzzle (piece size is " << puzzleImage.GetPieceSize() << "x" << puzzleImage.GetPieceSize()
        << ", puzzle dimension is " << puzzleImage.GetNumOfRowPieces() << "x" << puzzleImage.GetNumOfColPieces() << ")." << std::endl;

    //6. Log init time, total run time, and iterations info
    str << "Init in " << in_timeInit << " seconds, total " << Utilities::Timer::GetTimeString(in_runData.m_duration)
        << ", in iteration #" << in_runData.m_numOfIterations
        << ", " << in_runData.m_averageTimePerIteration << " milliseconds per iteration." << std::endl;

    //7. Log execution policy and allocation type   
    str << "With " << (IS_PARALLEL_EXECUTION_POLICY ? "parallel" : "sequential") << " execution policy, ";
    str << "with " << (DO_NOT_COMPUTE_UNNECESSARY_SUPPORT_VALUES ? "efficient computation of support values" : "computation of all support values");

    //8. Log max RL iterations and RL threshold
    str << ", max number of RL iterations is " << m_puzzleSolver->m_maxRLIterations <<  
        ". RL threshold is " << RelaxationLabelingSolverBaseConstants::epsilon << std::endl;

    //9. Log original image path
    str << "Image name is: '" << FileSystemUtils::GetFileNameFromPath(m_puzzleSolver->m_sourceImagePath) << "'";
    if (m_areMultipleRuns)
        str << " (run " << m_puzzleSolver->m_runNumber<< ")";
    str << "." << std::endl;

    //10. Log dissimilarity type and compatibility function into
    str << "Dissimilarity is " << RLSolverGeneralUtils::GetDissimilarityTypeString(DISSIMILARITY_TYPE);

    str << ", compatibility func is computed with k = " << m_puzzleSolver->m_kParamForMethod2CompatibilityComputation;

    str << "." << std::endl;

    //11. Log permutation labeling and duplicated labels info
    if (m_puzzleSolver->m_pSolution->IsPermutationLabeling(false))
        str << "Final labeling is permutation labeling" << std::endl;
    else
        str << "Final labeling IS NOT permutation labeling" << std::endl;

    str << "##############" << std::endl << std::endl;

    return str.str();
}


/************************************************************************/
void PuzzleRL_SolverOutputManager::LogMultiplePuzzlesOutput(const std::vector<RunData>& in_runDataVector)
//Function input: in_runDataVec: vector of RunData structs
//Function output: none
//Function objective: log data of multiple runs
/************************************************************************/
{
    const size_t totalNumOfRuns = in_runDataVector.size();

    size_t numOfPuzzlesSuccessfullySolved = 0;
    size_t numOfPuzzlesFeasiblySolved = 0;
    std::string failedRunsStr;
    std::string nonFeasiblySolvedRunsStr;

    for (const RunData& currRunData: in_runDataVector)
    {
        if (currRunData.m_wasPuzzleCorrectlySolved)
            ++numOfPuzzlesSuccessfullySolved;
        else
            failedRunsStr += (failedRunsStr.empty()? "" : ", ") + std::to_string(currRunData.m_runNumber);

        if (currRunData.m_isAssignmentFeasible)
            ++numOfPuzzlesFeasiblySolved;
        else
            nonFeasiblySolvedRunsStr += (nonFeasiblySolvedRunsStr.empty()? "" : ", ") + std::to_string(currRunData.m_runNumber);
    }

    std::stringstream textStr;
    textStr << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
    textStr << numOfPuzzlesSuccessfullySolved << "/" << totalNumOfRuns << " puzzles were successfully solved";
    if (numOfPuzzlesSuccessfullySolved < totalNumOfRuns)
        textStr << " (the following runs have failed: " << failedRunsStr << ")";
    textStr << std::endl;

    textStr << numOfPuzzlesFeasiblySolved << "/" << totalNumOfRuns << " puzzles were legally solved";
    if (numOfPuzzlesSuccessfullySolved < totalNumOfRuns)
        textStr << " (the following runs produced non-feasible solutions: " << nonFeasiblySolvedRunsStr << ")";
    textStr << std::endl;

    const RunData averageRunData = RunData::ComputeAverageRunData(in_runDataVector);

    textStr << std::endl << "Average run data: num of iteration is " << averageRunData.m_numOfIterations <<
        ", time is " << Utilities::Timer::GetTimeString(averageRunData.m_duration) <<
        ", direct comparison is: " << averageRunData.m_directComparison << "%" <<
        ", neighbor comparison is: " << averageRunData.m_neighborComparison << "%" <<
        ", " << std::endl << "assigned piece percentage is: " << averageRunData.m_assignedPiecesPercentage << "%." << std::endl;

    if (averageRunData.m_isType2Puzzle)
    {
        textStr << "average time for two type 2 runs is " << Utilities::Timer::GetTimeString(averageRunData.m_type2RunTimeForTwoRotations) << std::endl;
    }

    textStr << std::endl << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;

    g_logger.BruteLog(textStr.str());

    AddToMultipleRunsSummary(textStr.str());
}

/************************************************************************/
std::string PuzzleRL_SolverOutputManager::GetPiecesOrientationStr(const PieceObjectAndRot* in_firstRotatedPiece,
    const PieceObjectAndRot* in_secondRotatedPiece, const Orientation in_orien, const double& in_val) const
    /************************************************************************/
{
    std::stringstream str;

    str << "first: '" << in_firstRotatedPiece->GetString() << "', ";
    str << "second: '" << in_secondRotatedPiece->GetString() << "', ";
    str << "orien: '" << RLSolverGeneralUtils::GetOrientationString(in_orien) << "'";
    if (-1 != in_val)
        str << ". " << "val is: " << in_val << ((in_val == 1) ? " (max)" : "");

    str << std::endl;

    return str.str();
}

/************************************************************************/
void PuzzleRL_SolverOutputManager::LogConstantPiecesInfo() const
/************************************************************************/
{
    std::stringstream str;
    if (m_puzzleSolver->m_constantPiecesVector.empty())
    {
        str << "No constant pieces";
    }
    else
    {
        std::string constantPiecesStr;
        for (const PieceObjectAndRot* currRotatedPiece : m_puzzleSolver->m_constantPiecesVector)
        {
            if (!constantPiecesStr.empty())
                constantPiecesStr += ", ";

            constantPiecesStr += currRotatedPiece->GetString();
        }
        str << "The following " << m_puzzleSolver->m_constantPiecesVector.size() << " pieces are totally identical (constant pieces): " << 
            constantPiecesStr << std::endl;
    }

    Utilities::Log(str.str());
}

/************************************************************************/
void PuzzleRL_SolverOutputManager::LogWrongAssignmentsInfo(const RowColInfoSet& in_wrongAssignedPieceCoordsSet, 
    const PieceNumbersAndRotationsMatrix& in_rotatedSolution) const
//Function input: in_wrongAssignedPieceCoordsSet: set of wrong coordinates in solution; in_rotatedSolution: solution
//Function output: none
//Function objective: log info about wrong assignments in 'in_rotatedSolution'
/************************************************************************/
{
    //1. Print info to txt file
    std::stringstream str;

    str << "Number of wrong assignments is " << in_wrongAssignedPieceCoordsSet.size() << std::endl << std::endl;

    const PieceNumbersAndRotationsMatrix groundTruth = m_puzzleSolver->m_puzzleImage.GetGroundTruthSolutionInfo().m_assignmentMatrix;

    for (const RowColInfo& currCoord: in_wrongAssignedPieceCoordsSet)
    {
        const PieceNumberAndRotation assignedPiece = in_rotatedSolution(currCoord.m_row, currCoord.m_col);
        const PieceNumberAndRotation groundTruthPiece = groundTruth(currCoord.m_row, currCoord.m_col);

        str << "In coordinate " << currCoord.GetString() << 
            ": assigned piece is " << assignedPiece.GetString() << 
            (m_puzzleSolver->IsConstantPiece(m_puzzleSolver->GetRotatedPieceFromPieceNumberAndRotation(assignedPiece))? "(constant piece)" : "") <<
            ". Should be " << groundTruthPiece.GetString() <<
            (m_puzzleSolver->IsConstantPiece(m_puzzleSolver->GetRotatedPieceFromPieceNumberAndRotation(groundTruthPiece))? "(constant piece)" : "") << std::endl;
    }

    FileSystemUtils::WriteToFile(m_puzzleSolver->GetPathInRunFolder("wrongAssignments.txt"), str);

    //2. Visualize info in image
    PrintWrongAssignmentsInRotatedSolutionVisualization(in_wrongAssignedPieceCoordsSet, in_rotatedSolution);
}

/************************************************************************/
std::string PuzzleRL_SolverOutputManager::GetCompatibilityTypeString() const
/************************************************************************/
{
    std::stringstream str;

    str << "real solver compatibility";

    if (m_puzzleSolver->m_config.WasMinimumThresholdDefined())
        str << ", WITH min threshold of " << m_puzzleSolver->m_config.GetMinimumThresholdForPieceCompatibility();
    else
        str << ", NO min threshold";

    return str.str();
}

/************************************************************************/
void PuzzleRL_SolverOutputManager::PrintWrongAssignmentsInRotatedSolutionVisualization(const RowColInfoSet& in_wrongAssignedPieceCoordsSet, 
    const PieceNumbersAndRotationsMatrix& in_rotatedSolution) const
//Function input: in_wrongAssignedPieceCoordsSet: set of wrong coordinates in solution; in_rotatedSolution: solution
//Function output: none
//Function objective: print visualization of wrong assignment in 'in_rotatedSolution'
/************************************************************************/
{
    //1. Set wrong assignments sign
    const PixelType orangeBGR = PixelType(0, 165, 255); 
    const PixelType orangeLab = ImageUtils::ConvertPixelFromBGR_ToLAB(orangeBGR);

    const int32_t pieceSize = m_puzzleSolver->m_puzzleImage.GetPieceSize();
    cv::Mat wrongAssignmentSign = cv::Mat::zeros(pieceSize, pieceSize, ImageUtils::imagesType);
    wrongAssignmentSign.setTo(orangeLab);

    //2. Set constant pieces sign
    const PixelType purpleBGR = PixelType(102, 0, 102); 
    const PixelType purpleLab = ImageUtils::ConvertPixelFromBGR_ToLAB(purpleBGR);

    cv::Mat constantPiecesSign = cv::Mat::zeros(pieceSize, pieceSize, ImageUtils::imagesType);
    constantPiecesSign.setTo(purpleLab);

    //3. Get solution image and ground truth info
    const GroundTruthSolutionInfo groundTruthSolutionInfo = m_puzzleSolver->m_puzzleImage.GetGroundTruthSolutionInfo();
    cv::Mat rotatedSolutionImage = ImageUtils::GetImageMatrixFromAssignementAndPieces(in_rotatedSolution, pieceSize,
            groundTruthSolutionInfo.m_piecesWithNoTextMap);

    constexpr bool showConstantPiecesDifferently = true;

    //4. Set wrong assignments with 'wrongAssignmentSign' 
    for (const RowColInfo currCoord: in_wrongAssignedPieceCoordsSet)
    {
        const PieceNumberAndRotation currPieceNumAndRotInSol = in_rotatedSolution(currCoord.m_row, currCoord.m_col);

        if constexpr (showConstantPiecesDifferently)
        {
            if (m_puzzleSolver->IsConstantPiece(m_puzzleSolver->GetPuzzlePieceObjectByPieceNumber(currPieceNumAndRotInSol.m_pieceNumber)))
                ImageUtils::SetPieceToImageCoordinates(constantPiecesSign, currCoord, rotatedSolutionImage);
            else
                ImageUtils::SetPieceToImageCoordinates(wrongAssignmentSign, currCoord, rotatedSolutionImage);
        }
        else
        {
            ImageUtils::SetPieceToImageCoordinates(wrongAssignmentSign, currCoord, rotatedSolutionImage);
        }
    }

    //5. Add pieces grid to image
    ImageUtils::AddPiecesGridToImage(rotatedSolutionImage, pieceSize);

    //6. Write image
    ImageUtils::WriteImage(m_puzzleSolver->GetPathInRunFolder("WrongAssignmentsVisualization.png"), rotatedSolutionImage);
}

/************************************************************************/
void PuzzleRL_SolverOutputManager::AddToMultipleRunsSummary(const std::string& in_strForTxt)
/************************************************************************/
{
    if (m_areMultipleRuns)
    {                          
        FileSystemUtils::AppendToFile(RLSolverFileSystemUtils::GetMultipleRunsTextSummaryFilePath(), std::stringstream(in_strForTxt));
    }
}

/************************************************************************/
std::string PuzzleRL_SolverOutputManager::GetStartTranslationDilemmaString(const int32_t& in_level, 
    const std::string& in_transAction, const std::string& in_transDilemma)
//Function input: in_level: level; in_transAction: translation action; in_transDilemma: translation dilemma
//Function output: get start translation dilemma title string 
//Function objective: described in "Function output"
/************************************************************************/
{
    const std::string title = "Doing " + in_transAction + " in " + in_transDilemma;
    const std::string startTranslationTitle = GetLevelTitleString(in_level, title); 

    return "\n" + startTranslationTitle + "\n";
}

/************************************************************************/
std::string PuzzleRL_SolverOutputManager::GetDoneTranslationDilemmaString(const int32_t& in_level, 
    const std::string& in_transAction, const std::string& in_transDilemma)
//Function input: in_level: level; in_transAction: translation action; in_transDilemma: translation dilemma
//Function output: get done translation dilemma title string 
//Function objective: described in "Function output"
/************************************************************************/
{
    const std::string title = "Done " + in_transAction + " in " + in_transDilemma;
    const std::string doneTranslationTitle = GetLevelTitleString(in_level, title); 

    return "\n" + doneTranslationTitle + "\n";
}

/************************************************************************/
std::string PuzzleRL_SolverOutputManager::GetLevelTitleString(const int32_t& in_level, const std::string& in_title)
//Function input: in_level: level; in_title: title
//Function output: none
//Function objective: get title string according to 'in_level'
/************************************************************************/
{
    if (1 == in_level)
    {
        return Get1stLevelTitleString("1. " + in_title);
    }
    else if (2 == in_level)
    {
        return Get2ndLevelTitleString("2. " + in_title);
    }
    else if (3 == in_level)
    {
        return Get3rdLevelTitleString("3. " + in_title);
    }
    else
    {
        Utilities::LogAndAbort("In 'PuzzleRL_SolverOutputManager::GetLevelTitleString()': shouldn't get here");
        return "";
    }
}