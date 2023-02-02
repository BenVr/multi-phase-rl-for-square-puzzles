#pragma once
#include "PuzzleRL_SolverConstants.h"
#include "PuzzleRL_SolverInputInfo.h"
#include "XML_Configuration.h"

class PuzzleRL_SolverConfiguration : public XML_Configuration
{
public:
    PuzzleRL_SolverConfiguration(const std::string& in_configXML_Path);

    void ReadXMLSpecific(const XMLElement* in_configurationsElement) override;
    void ReadImageInput(const XMLElement* in_configurationsElement);

    ImageInputData GetImageInputData() const {return m_imageInputData;}
    PuzzleType GetPuzzleType() const { return m_puzzleType; }
    bool WasMinimumThresholdDefined() const { return m_shouldThresholdPieceCompatibility; }
    
    double GetMinimumThresholdForPieceCompatibility() const { return m_minThresholdForPieceCompatibility; }

    bool IsOnRunAllFolderMode() const;
    std::string GetRunFolderName() const;
    std::string GetSourceName() const;
    std::string GetSourcePath() const;

protected:
    void ReadSolverParameters(const XMLElement* in_configurationsElement);

    ImageInputData m_imageInputData;

    PuzzleType m_puzzleType = PuzzleType::eType1_NoRotation;
    
    bool m_shouldThresholdPieceCompatibility = false;
    double m_minThresholdForPieceCompatibility = 0.0;
};
