#include "PuzzleRL_SolverConfiguration.h"
#include "Utilities.h"

/************************************************************************/
PuzzleRL_SolverConfiguration::PuzzleRL_SolverConfiguration(const std::string& in_configXML_Path) :
XML_Configuration(in_configXML_Path)
/************************************************************************/
{
}

/************************************************************************/
void PuzzleRL_SolverConfiguration::ReadXMLSpecific(const XMLElement* in_configurationsElement)
/************************************************************************/
{
    //1. Read image input
    ReadImageInput(in_configurationsElement);
    
    //2. Read "PuzzleType"
    int32_t readPuzzleType = 0;
    in_configurationsElement->FirstChildElement("PuzzleType")->QueryIntText(&readPuzzleType);
    m_puzzleType = static_cast<PuzzleType>(readPuzzleType);
    Utilities::LogAndAbortIf((PuzzleType::eType1_NoRotation != m_puzzleType) && (PuzzleType::eType2_UknownRotation != m_puzzleType),
        "Wrong puzzle type defined in xml");

    //4. Read real solver parameters
    ReadSolverParameters(in_configurationsElement);
}

/************************************************************************/
void PuzzleRL_SolverConfiguration::ReadImageInput(const XMLElement* in_configurationsElement)
/************************************************************************/
{
    const XMLElement* imageInputElement = in_configurationsElement->FirstChildElement("ImageInput");

    if (imageInputElement)
    {
        m_imageInputData.ReadImageInput(imageInputElement);
    }
}

/************************************************************************/
bool PuzzleRL_SolverConfiguration::IsOnRunAllFolderMode() const
/************************************************************************/
{
    const std::string runAllFolderName = m_imageInputData.GetRunAllFolderName();
    return !runAllFolderName.empty();
}

/************************************************************************/
std::string PuzzleRL_SolverConfiguration::GetRunFolderName() const
/************************************************************************/
{
    return m_imageInputData.GetRunAllFolderName();
}

/************************************************************************/
std::string PuzzleRL_SolverConfiguration::GetSourceName() const
/************************************************************************/
{
    return  m_imageInputData.GetSourceImageName();;
}

/************************************************************************/
std::string PuzzleRL_SolverConfiguration::GetSourcePath() const
/************************************************************************/
{
    std::string sourceName = DATA_FOLDER_PATH.string() + m_imageInputData.GetSourceImageName();

    return sourceName;
}

/************************************************************************/
void PuzzleRL_SolverConfiguration::ReadSolverParameters(const XMLElement* in_configurationsElement)
/************************************************************************/
{
    //1. Read 'SolverParameters' element
    const XMLElement* solverParametersElement = in_configurationsElement->FirstChildElement("SolverParameters");

    if (nullptr != solverParametersElement)
    {
        //3. Read 'MinThresholdForPieceCompatibility' element
        const XMLElement* minThresholdElement = solverParametersElement->FirstChildElement("MinThresholdForPieceCompatibility");
        if (nullptr != minThresholdElement)
        {
            m_shouldThresholdPieceCompatibility = true;
            m_minThresholdForPieceCompatibility = minThresholdElement->DoubleAttribute("Value");
        }
    }
}