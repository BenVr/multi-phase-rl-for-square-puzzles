#include "PiecesBlockTranslationManager.h"
#include "PuzzleRL_Solver.h"

/************************************************************************/
TranslationMode PiecesBlockTranslationManager::DoTranslation(Labeling& inout_labeling, const RowColInfoSet& in_newJustAnchoredEntries, 
    RowColInfoSet& inout_anchoredEntriesSet) const
//Function input: inout_labeling: labeling; in_newJustAnchoredEntries: set of new just anchored entries
    //(sent to this function just for efficiency purposes); inout_anchoredEntriesSet: already anchored entries set
//Function output: boolean indicating whether translation was done or not 
//Function objective: if needed, do vertical and horizontal translation of the pieces block represented by 'inout_labeling'
/************************************************************************/
{
    //We can apply only one of vertical and horizontal in each call
    TranslationMode translationMode = DoVerticalTranslation(inout_labeling, in_newJustAnchoredEntries, inout_anchoredEntriesSet);

    if (TranslationMode::eDidNoTranslation == translationMode)
        translationMode = DoHorizontalTranslation(inout_labeling, in_newJustAnchoredEntries, inout_anchoredEntriesSet);

    return translationMode;
}

/************************************************************************/
void PiecesBlockTranslationManager::TranslatePiecesBlock(Labeling& inout_labeling, const TranslationDecision& in_transDecision, 
    AnchoringData& inout_anchoringData) const
//Function input: inout_labeling: labeling; in_transDecision: translation decision; inout_anchoringData: anchoring data
//Function output: none 
//Function objective: translate the pieces block represented by 'inout_labeling' according to 'in_transDecision'
/************************************************************************/
{
    switch (in_transDecision)
    {
        case TranslationDecision::eNoTranslation:
            break;

        case TranslationDecision::eDoTranslationUp:
            MovePiecesBlockUp(inout_labeling, inout_anchoringData.m_anchoredEntriesSet);
            break;

        case TranslationDecision::eDoTranslationDown:
            MovePiecesBlockDown(inout_labeling, inout_anchoringData.m_anchoredEntriesSet);
            break;

        case TranslationDecision::eDoTranslationLeft:
            MovePiecesBlockLeft(inout_labeling, inout_anchoringData.m_anchoredEntriesSet);
            break;

        case TranslationDecision::eDoTranslationRight:
            MovePiecesBlockRight(inout_labeling, inout_anchoringData.m_anchoredEntriesSet);
            break;

        default:
            Utilities::LogAndAbort("shouldn't get here");
            break;
    }
}

/************************************************************************/
void PiecesBlockTranslationManager::AllowTranslation(const TranslationMode& in_transDilemma) const
/************************************************************************/
{
    switch (in_transDilemma)
    {
    case TranslationMode::eVerticalTranslationUpDilemma:
    case TranslationMode::eVerticalTranslationDownDilemma:
        m_doneWithVerticalTranslation = false;
        break;

    case TranslationMode::eHorizontalTranslationLeftDilemma:
    case TranslationMode::eHorizontalTranslationRightDilemma:
        m_doneWithHorizontalTranslation = false;
        break;

    default:
        Utilities::LogAndAbort("In 'PiecesBlockTranslationManager::AllowTranslation()': shouldn't get here");
    }
}

/************************************************************************/
void PiecesBlockTranslationManager::DisallowTranslation(const TranslationMode& in_transDilemma) const
/************************************************************************/
{
    switch (in_transDilemma)
    {
    case TranslationMode::eVerticalTranslationUpDilemma:
    case TranslationMode::eVerticalTranslationDownDilemma:
        m_doneWithVerticalTranslation = true;
        break;

    case TranslationMode::eHorizontalTranslationLeftDilemma:
    case TranslationMode::eHorizontalTranslationRightDilemma:
        m_doneWithHorizontalTranslation = true;
        break;

    default:
        Utilities::LogAndAbort("In 'PiecesBlockTranslationManager::AllowTranslation()': shouldn't get here");
    }
}

/************************************************************************/
void PiecesBlockTranslationManager::SetEfficiencyParams(const std::set<RowColInfo>& in_anchoredEntriesSet) const
/************************************************************************/
{
    m_mostTopRowOfAnchoredPiece = GetMostTopRowIndexFromEntriesSet(in_anchoredEntriesSet);
    m_mostBottomRowOfAnchoredPiece = GetMostBottomRowIndexFromEntriesSet(in_anchoredEntriesSet);
    m_mostLeftColOfAnchoredPiece = GetMostLeftColIndexFromEntriesSet(in_anchoredEntriesSet);
    m_mostRightColOfAnchoredPiece = GetMostRightColIndexFromEntriesSet(in_anchoredEntriesSet);
}

/************************************************************************/
TranslationMode PiecesBlockTranslationManager::DoVerticalTranslation(Labeling& inout_labeling, const RowColInfoSet& in_newJustAnchoredEntries, 
    RowColInfoSet& inout_anchoredEntriesSet) const
//Function input: inout_labeling: labeling; in_newJustAnchoredEntries: set of new just anchored entries;
    //inout_anchoredEntriesSet: already anchored entries set
//Function output: boolean indicating whether vertical translation was done or not 
//Function objective: if needed, do vertical translation of the pieces block represented by 'inout_labeling'
/************************************************************************/
{
    if (m_doneWithVerticalTranslation)
        return TranslationMode::eDidNoTranslation;

    //1. Get variables
    const int32_t mostTopRowOfNewAnchoredPiece = GetMostTopRowIndexFromEntriesSet(in_newJustAnchoredEntries);
    const int32_t mostBottomRowOfNewAnchoredPiece = GetMostBottomRowIndexFromEntriesSet(in_newJustAnchoredEntries);

    if (mostTopRowOfNewAnchoredPiece == m_mostTopRowOfAnchoredPiece && mostBottomRowOfNewAnchoredPiece == m_mostBottomRowOfAnchoredPiece)
        return TranslationMode::eDidNoTranslation;

    Utilities::Log("In DoVerticalTranslation");
    m_mostTopRowOfAnchoredPiece = std::min(m_mostTopRowOfAnchoredPiece, mostTopRowOfNewAnchoredPiece);
    m_mostBottomRowOfAnchoredPiece = std::max(m_mostBottomRowOfAnchoredPiece, mostBottomRowOfNewAnchoredPiece);

    const int32_t numOfRowsPieces = m_puzzleSolver->m_puzzleImage.GetNumOfRowPieces();

    TranslationMode retVal = TranslationMode::eDidNoTranslation;

    //2. Do translation
    if (0 == m_mostTopRowOfAnchoredPiece && (numOfRowsPieces - 1) == m_mostBottomRowOfAnchoredPiece)
    {
        //Do nothing - pieces are spread over all rows
    }
    else if (0 == m_mostTopRowOfAnchoredPiece)
    {
        //If moving the block down will get it touching bottom edge
        if ((numOfRowsPieces - 1) == (m_mostBottomRowOfAnchoredPiece + 1))
        {
            retVal = TranslationMode::eVerticalTranslationDownDilemma;
        }
        else
        {
            MovePiecesBlockDown(inout_labeling, inout_anchoredEntriesSet);
            retVal = TranslationMode::eDidTranslation;
        }
    }
    else if ((numOfRowsPieces - 1) == m_mostBottomRowOfAnchoredPiece)
    {
        //If moving the block up will get it touching top edge
        if (1 == m_mostTopRowOfAnchoredPiece)
        {
            retVal = TranslationMode::eVerticalTranslationUpDilemma;
        }
        else
        {
            MovePiecesBlockUp(inout_labeling, inout_anchoredEntriesSet);
            retVal = TranslationMode::eDidTranslation;
        }
    }

    return retVal;
}

/************************************************************************/
TranslationMode PiecesBlockTranslationManager::DoHorizontalTranslation(Labeling& inout_labeling, const RowColInfoSet& in_newJustAnchoredEntries,
    RowColInfoSet& inout_anchoredEntriesSet) const
//Function input: inout_labeling: labeling; in_newJustAnchoredEntries: set of new just anchored entries;
    //inout_anchoredEntriesSet: already anchored entries set
//Function output: boolean indicating whether horizontal translation was done or not 
//Function objective: if needed, do horizontal translation of the pieces block represented by 'inout_labeling'
/************************************************************************/
{
    if (m_doneWithHorizontalTranslation)
        return TranslationMode::eDidNoTranslation;

    //1. Get variables
    const int32_t mostLeftColOfNewAnchoredPiece = GetMostLeftColIndexFromEntriesSet(in_newJustAnchoredEntries);
    const int32_t mostRightColOfNewAnchoredPiece = GetMostRightColIndexFromEntriesSet(in_newJustAnchoredEntries);

    if (mostLeftColOfNewAnchoredPiece == m_mostLeftColOfAnchoredPiece && mostRightColOfNewAnchoredPiece == m_mostRightColOfAnchoredPiece)
        return TranslationMode::eDidNoTranslation;

    m_mostLeftColOfAnchoredPiece = std::min(m_mostLeftColOfAnchoredPiece, mostLeftColOfNewAnchoredPiece);
    m_mostRightColOfAnchoredPiece = std::max(m_mostRightColOfAnchoredPiece, mostRightColOfNewAnchoredPiece);

    const int32_t numOfColPieces = m_puzzleSolver->m_puzzleImage.GetNumOfColPieces();

    TranslationMode retVal = TranslationMode::eDidNoTranslation;

    //2. Do translation
    if (0 == m_mostLeftColOfAnchoredPiece && (numOfColPieces - 1) == m_mostRightColOfAnchoredPiece)
    {
        //Do nothing - pieces are spread over all columns
    }
    else if (0 == m_mostLeftColOfAnchoredPiece)
    {
        //If moving the block right will get it touching right edge
        if ((numOfColPieces - 1) == (m_mostRightColOfAnchoredPiece + 1))
        {
            retVal = TranslationMode::eHorizontalTranslationRightDilemma;
        }
        else
        {
            MovePiecesBlockRight(inout_labeling, inout_anchoredEntriesSet);
            retVal = TranslationMode::eDidTranslation;
        }
    }
    else if ((numOfColPieces - 1) == m_mostRightColOfAnchoredPiece)
    {
        //If moving the block left will get it touching left edge
        if (1 == m_mostLeftColOfAnchoredPiece)
        {
            retVal = TranslationMode::eHorizontalTranslationLeftDilemma;
        }
        else
        {
            MovePiecesBlockLeft(inout_labeling, inout_anchoredEntriesSet);
            retVal = TranslationMode::eDidTranslation;
        }
    }

    return retVal;
}

/************************************************************************/
int32_t PiecesBlockTranslationManager::GetMostTopRowIndexOfAnAnchoredPiece(const Labeling& in_labeling) const
//Function input: in_labeling: labeling;
//Function output: index of the row that is the most top one in the pieces block represented by 'inout_labeling' 
//Function objective: described in "Function output"
/************************************************************************/
{
    int32_t topRowIndex = m_puzzleSolver->m_puzzleImage.GetNumOfRowPieces();

    for (const LocationRotationLabel* currLocationRotationLabel : m_puzzleSolver->m_locationRotationLabelsPool)
    {
        if (in_labeling.IsColBinaryWithSumOne(currLocationRotationLabel->m_index))
        {
            if (currLocationRotationLabel->m_row < topRowIndex)
            {
                topRowIndex = currLocationRotationLabel->m_row;
            }
        }
    }

    return topRowIndex;
}

/************************************************************************/
int32_t PiecesBlockTranslationManager::GetMostBottomRowIndexOfAnAnchoredPiece(const Labeling& in_labeling) const
//Function input: in_labeling: labeling;
//Function output: index of the row that is the most bottom one in the pieces block represented by 'inout_labeling' 
//Function objective: described in "Function output"
/************************************************************************/
{
    int32_t bottomRowIndex = -1;

    for (const LocationRotationLabel* currLocationRotationLabel : m_puzzleSolver->m_locationRotationLabelsPool)
    {
        if (in_labeling.IsColBinaryWithSumOne(currLocationRotationLabel->m_index))
        {
            if (currLocationRotationLabel->m_row > bottomRowIndex)
            {
                bottomRowIndex = currLocationRotationLabel->m_row;
            }
        }
    }

    return bottomRowIndex;
}

/************************************************************************/
int32_t PiecesBlockTranslationManager::GetMostLeftColIndexOfAnAnchoredPiece(const Labeling& in_labeling) const
//Function input: in_labeling: labeling;
//Function output: index of the column that is the most left one in the pieces block represented by 'inout_labeling' 
//Function objective: described in "Function output"
/************************************************************************/
{
    int32_t leftColIndex = m_puzzleSolver->m_puzzleImage.GetNumOfColPieces();

    for (const LocationRotationLabel* currLocationRotationLabel : m_puzzleSolver->m_locationRotationLabelsPool)
    {
        if (in_labeling.IsColBinaryWithSumOne(currLocationRotationLabel->m_index))
        {
            if (currLocationRotationLabel->m_column < leftColIndex)
            {
                leftColIndex = currLocationRotationLabel->m_column;
            }
        }
    }

    return leftColIndex;
}

/************************************************************************/
int32_t PiecesBlockTranslationManager::GetMostRightColIndexOfAnAnchoredPiece(const Labeling& in_labeling) const
//Function input: in_labeling: labeling;
//Function output: index of the column that is the most right one in the pieces block represented by 'inout_labeling' 
//Function objective: described in "Function output"
/************************************************************************/
{
    int32_t rightColIndex = -1;

    for (const LocationRotationLabel* currLocationRotationLabel : m_puzzleSolver->m_locationRotationLabelsPool)
    {
        if (in_labeling.IsColBinaryWithSumOne(currLocationRotationLabel->m_index))
        {
            if (currLocationRotationLabel->m_column > rightColIndex)
            {
                rightColIndex = currLocationRotationLabel->m_column;
            }
        }
    }

    return rightColIndex;
}

/************************************************************************/
int32_t PiecesBlockTranslationManager::GetMostTopRowIndexFromEntriesSet(const RowColInfoSet& in_anchoredEntriesSet) const
//Function input: in_anchoredEntriesSet: anchored entry set
//Function output: index of the row that is the most top one in the pieces block represented by 'in_anchoredEntriesSet'' 
//Function objective: described in "Function output"
/************************************************************************/
{
    int32_t topRowIndex = m_puzzleSolver->m_puzzleImage.GetNumOfRowPieces();

    for (const RowColInfo currEntry: in_anchoredEntriesSet)
    {
        const LocationRotationLabel* currLocationRotationLabel =  m_puzzleSolver->m_locationRotationLabelsPool[currEntry.m_col];
        if (currLocationRotationLabel->m_row < topRowIndex)
        {
            topRowIndex = currLocationRotationLabel->m_row;
        }
    }

    return topRowIndex;
}

/************************************************************************/
int32_t PiecesBlockTranslationManager::GetMostBottomRowIndexFromEntriesSet(const RowColInfoSet& in_anchoredEntriesSet) const
//Function input: in_anchoredEntriesSet: anchored entry set
//Function output: index of the row that is the most bottom one in the pieces block represented by 'in_anchoredEntriesSet'' 
//Function objective: described in "Function output"
/************************************************************************/
{
    int32_t bottomRowIndex = -1;

    for (const RowColInfo currEntry: in_anchoredEntriesSet)
    {
        const LocationRotationLabel* currLocationRotationLabel =  m_puzzleSolver->m_locationRotationLabelsPool[currEntry.m_col];
        if (currLocationRotationLabel->m_row > bottomRowIndex)
        {
            bottomRowIndex = currLocationRotationLabel->m_row;
        }
    }

    return bottomRowIndex;
}

/************************************************************************/
int32_t PiecesBlockTranslationManager::GetMostLeftColIndexFromEntriesSet(const RowColInfoSet& in_anchoredEntriesSet) const
//Function input: in_anchoredEntriesSet: anchored entry set
//Function output: index of the column that is the most left one in the pieces block represented by 'in_anchoredEntriesSet'' 
//Function objective: described in "Function output"
/************************************************************************/
{
    int32_t leftColIndex = m_puzzleSolver->m_puzzleImage.GetNumOfColPieces();

    for (const RowColInfo currEntry: in_anchoredEntriesSet)
    {
        const LocationRotationLabel* currLocationRotationLabel =  m_puzzleSolver->m_locationRotationLabelsPool[currEntry.m_col];
        if (currLocationRotationLabel->m_column < leftColIndex)
        {
            leftColIndex = currLocationRotationLabel->m_column;
        }
    }

    return leftColIndex;
}

/************************************************************************/
int32_t PiecesBlockTranslationManager::GetMostRightColIndexFromEntriesSet(const RowColInfoSet& in_anchoredEntriesSet) const
//Function input: in_anchoredEntriesSet: anchored entry set
//Function output: index of the column that is the most right one in the pieces block represented by 'in_anchoredEntriesSet'' 
//Function objective: described in "Function output"
/************************************************************************/
{
    int32_t rightColIndex = -1;

    for (const RowColInfo currEntry: in_anchoredEntriesSet)
    {
        const LocationRotationLabel* currLocationRotationLabel =  m_puzzleSolver->m_locationRotationLabelsPool[currEntry.m_col];
        if (currLocationRotationLabel->m_column > rightColIndex)
        {
            rightColIndex = currLocationRotationLabel->m_column;
        }
    }

    return rightColIndex;
}

/************************************************************************/
void PiecesBlockTranslationManager::MovePiecesBlockUp(Labeling& inout_labeling, RowColInfoSet& inout_anchoredEntriesSet) const
//Function input: inout_labeling: labeling; inout_anchoredEntriesSet: already anchored entries set
//Function output: none 
//Function objective: move the pieces block in 'inout_labeling' according one row up
/************************************************************************/
{
    const Utilities::RowColInfo blockOffset(-1, 0);
    MovePiecesBlock(inout_labeling, blockOffset, inout_anchoredEntriesSet);
    Utilities::Log("MovePiecesBlockUp");
    m_mostTopRowOfAnchoredPiece = GetMostTopRowIndexOfAnAnchoredPiece(inout_labeling);
    m_mostBottomRowOfAnchoredPiece = GetMostBottomRowIndexOfAnAnchoredPiece(inout_labeling);
}

/************************************************************************/
void PiecesBlockTranslationManager::MovePiecesBlockDown(Labeling& inout_labeling, RowColInfoSet& inout_anchoredEntriesSet) const
//Function input: inout_labeling: labeling; inout_anchoredEntriesSet: already anchored entries set
//Function output: none 
//Function objective: move the pieces block in 'inout_labeling' according one row down
/************************************************************************/
{
    const Utilities::RowColInfo blockOffset(1, 0);
    MovePiecesBlock(inout_labeling, blockOffset, inout_anchoredEntriesSet);
    Utilities::Log("MovePiecesBlockDown");
    m_mostTopRowOfAnchoredPiece = GetMostTopRowIndexOfAnAnchoredPiece(inout_labeling);
    m_mostBottomRowOfAnchoredPiece = GetMostBottomRowIndexOfAnAnchoredPiece(inout_labeling);
}

/************************************************************************/
void PiecesBlockTranslationManager::MovePiecesBlockLeft(Labeling& inout_labeling, RowColInfoSet& inout_anchoredEntriesSet) const
//Function input: inout_labeling: labeling; inout_anchoredEntriesSet: already anchored entries set
//Function output: none 
//Function objective: move the pieces block in 'inout_labeling' according one column left
/************************************************************************/
{
    const Utilities::RowColInfo blockOffset(0, -1);
    MovePiecesBlock(inout_labeling, blockOffset, inout_anchoredEntriesSet);
    Utilities::Log("MovePiecesBlockLeft");
    m_mostLeftColOfAnchoredPiece = GetMostLeftColIndexOfAnAnchoredPiece(inout_labeling);
    m_mostRightColOfAnchoredPiece = GetMostRightColIndexOfAnAnchoredPiece(inout_labeling);
}

/************************************************************************/
void PiecesBlockTranslationManager::MovePiecesBlockRight(Labeling& inout_labeling, RowColInfoSet& inout_anchoredEntriesSet) const
//Function input: inout_labeling: labeling; inout_anchoredEntriesSet: already anchored entries set
//Function output: none 
//Function objective: move the pieces block in 'inout_labeling' according one column right
/************************************************************************/
{
    const Utilities::RowColInfo blockOffset(0, 1);
    MovePiecesBlock(inout_labeling, blockOffset, inout_anchoredEntriesSet);
    Utilities::Log("MovePiecesBlockRight");
    m_mostLeftColOfAnchoredPiece = GetMostLeftColIndexOfAnAnchoredPiece(inout_labeling);
    m_mostRightColOfAnchoredPiece = GetMostRightColIndexOfAnAnchoredPiece(inout_labeling);
}

/************************************************************************/
void PiecesBlockTranslationManager::MovePiecesBlock(Labeling& inout_labeling, const Utilities::RowColInfo& in_blockOffset,
    RowColInfoSet& inout_anchoredEntriesSet) const
//Function input: inout_labeling: labeling; in_blockOffset: offset according to which the pieces block will be moved;
    //inout_anchoredEntriesSet: already anchored entries set 
//Function output: none 
//Function objective: move the pieces block in 'inout_labeling' according to 'in_blockOffset'
/************************************************************************/
{
    //1. Set the new anchored pieces and locations info, when all pieces are moved according to 'in_blockOffset'
    RowColInfoSet newAnchoredEntriesSet;
    for (const RowColInfo& currEntry: inout_anchoredEntriesSet)
    {
        //2.1. Set piece new location
        const LocationRotationLabel* currLabel = m_puzzleSolver->m_locationRotationLabelsPool[currEntry.m_col];
        const Utilities::RowColInfo newLocationCoord(currLabel->m_row + in_blockOffset.m_row, currLabel->m_column + in_blockOffset.m_col);

        //2.2. Find the label of 'newLocationCoord' and insert info to 'newAnchoredObjectsAndLabelsSet' 
        const LocationRotationLabel* newLabel = m_puzzleSolver->GetLocationRotationLabelByCoordAndRot(newLocationCoord, currLabel->m_rotation);

        //2.3. Insert new anchored entry to 'newAnchoredEntriesSet'
        newAnchoredEntriesSet.emplace(currEntry.m_row, newLabel->m_index);
    }

    //2. Anchor according to 'newAnchoredObjectsAndLabelsSet'
    inout_labeling = m_puzzleSolver->DoAnchoring(inout_labeling, newAnchoredEntriesSet);

    //3. Set 'inout_anchoredEntriesSet'
    inout_anchoredEntriesSet = newAnchoredEntriesSet;
}