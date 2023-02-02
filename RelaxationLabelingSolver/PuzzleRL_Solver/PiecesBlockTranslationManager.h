#pragma once

#include "Labeling.h"
#include "PuzzlePieceObject.h"

class PuzzleRL_Solver;

class PiecesBlockTranslationManager
{
public:
    PiecesBlockTranslationManager(const PuzzleRL_Solver* const in_puzzleRL_Solver) : m_puzzleSolver(in_puzzleRL_Solver) {}

    TranslationMode DoTranslation(Labeling& inout_labeling, const RowColInfoSet& in_newJustAnchoredEntries, RowColInfoSet& inout_anchoredEntriesSet) const;
    void TranslatePiecesBlock(Labeling& inout_labeling, const TranslationDecision& in_transDecision, AnchoringData& inout_anchoringData) const;

    void AllowTranslation(const TranslationMode& in_transDilemma) const;
    void DisallowTranslation(const TranslationMode& in_transDilemma) const;
    void SetEfficiencyParams(const std::set<RowColInfo>& in_anchoredEntriesSet) const;
      
    bool AreTranslationsDone() const {return m_doneWithVerticalTranslation && m_doneWithHorizontalTranslation;}

protected:
    TranslationMode DoVerticalTranslation(Labeling& inout_labeling, const RowColInfoSet& in_newJustAnchoredEntries, RowColInfoSet& inout_anchoredEntriesSet) const;
    TranslationMode DoHorizontalTranslation(Labeling& inout_labeling, const RowColInfoSet& in_newJustAnchoredEntries, RowColInfoSet& inout_anchoredEntriesSet) const;

    int32_t GetMostTopRowIndexOfAnAnchoredPiece(const Labeling& in_labeling) const;
    int32_t GetMostBottomRowIndexOfAnAnchoredPiece(const Labeling& in_labeling) const;
    int32_t GetMostLeftColIndexOfAnAnchoredPiece(const Labeling& in_labeling) const;
    int32_t GetMostRightColIndexOfAnAnchoredPiece(const Labeling& in_labeling) const;

    int32_t GetMostTopRowIndexFromEntriesSet(const RowColInfoSet& in_anchoredEntriesSet) const;
    int32_t GetMostBottomRowIndexFromEntriesSet(const RowColInfoSet& in_anchoredEntriesSet) const;
    int32_t GetMostLeftColIndexFromEntriesSet(const RowColInfoSet& in_anchoredEntriesSet) const;
    int32_t GetMostRightColIndexFromEntriesSet(const RowColInfoSet& in_anchoredEntriesSet) const;

    void MovePiecesBlockUp(Labeling& inout_labeling, RowColInfoSet& inout_anchoredEntriesSet) const;
    void MovePiecesBlockDown(Labeling& inout_labeling, RowColInfoSet& inout_anchoredEntriesSet) const;
    void MovePiecesBlockLeft(Labeling& inout_labeling, RowColInfoSet& inout_anchoredEntriesSet) const;
    void MovePiecesBlockRight(Labeling& inout_labeling, RowColInfoSet& inout_anchoredEntriesSet) const;
    void MovePiecesBlock(Labeling& inout_labeling, const Utilities::RowColInfo& in_blockOffset, RowColInfoSet& inout_anchoredEntriesSet) const;

    using ObjectLabelPair = std::pair<const Object*, const Label*>;
    using ObjectLabelPairsSet = std::set<ObjectLabelPair>;

    const PuzzleRL_Solver* const m_puzzleSolver;

    mutable int32_t m_mostTopRowOfAnchoredPiece = std::numeric_limits<int32_t>::max();
    mutable int32_t m_mostBottomRowOfAnchoredPiece = std::numeric_limits<int32_t>::lowest();
    mutable int32_t m_mostLeftColOfAnchoredPiece = std::numeric_limits<int32_t>::max();
    mutable int32_t m_mostRightColOfAnchoredPiece = std::numeric_limits<int32_t>::lowest();

    mutable bool m_doneWithVerticalTranslation = false;
    mutable bool m_doneWithHorizontalTranslation = false;
};