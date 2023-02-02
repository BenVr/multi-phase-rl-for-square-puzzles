#include "PuzzleRL_SolverInputInfo.h"
#include "RelaxationLabelingSolverBaseUtils.h"

/************************************************************************/
void ImageInputData::ReadImageInput(const XMLElement* in_imageInputElement)
//Function input: in_imageInputElement: image input xml element
//Function output: none
//Function objective: read image input mode data
/************************************************************************/
{
    //1. Read "PieceSize"
    in_imageInputElement->FirstChildElement("PieceSize")->QueryIntText(&m_pieceSize);

    //2. Read "SourceImageName" or "RunAllFolderName"
    ReadSourceImageOrRunAllFolder(in_imageInputElement);
}

/************************************************************************/
void ImageInputData::ReadSourceImageOrRunAllFolder(const XMLElement* in_imageInputElement)
/************************************************************************/
{
    const XMLElement* sourceImageNameElement = in_imageInputElement->FirstChildElement("SourceImageName");
    const XMLElement* runAllFolderNameElement = in_imageInputElement->FirstChildElement("RunAllFolderName");

    Utilities::LogAndAbortIf((sourceImageNameElement && runAllFolderNameElement) || (!sourceImageNameElement && !runAllFolderNameElement),
        "Exactly one of 'SourceImageName' and 'RunAllFolderName' should be defined");

    if (sourceImageNameElement)
        m_sourceImageName = in_imageInputElement->FirstChildElement("SourceImageName")->GetText();
    else if (runAllFolderNameElement)
        m_runAllFolderName = in_imageInputElement->FirstChildElement("RunAllFolderName")->GetText();
}