#include "XML_Configuration.h"
#include "Utilities.h"
#include "RelaxationLabelingSolverBaseUtils.h"

/************************************************************************/
XML_Configuration::XML_Configuration(const std::string& in_configXML_Path) : 
m_xmlPath(in_configXML_Path)
/************************************************************************/
{
}

/************************************************************************/
void XML_Configuration::ReadXML()
/************************************************************************/
{
    tinyxml2::XMLDocument doc;
    doc.LoadFile(m_xmlPath.c_str());

    const XMLElement* configurationsElement = doc.FirstChildElement("Configurations");

    ReadRunParameters(configurationsElement);
    ReadTechnicalParameters(configurationsElement);
}

/************************************************************************/
void XML_Configuration::ReadRunParameters(const XMLElement* in_configurationsElement)
/************************************************************************/
{
    const XMLElement* runParametersElement = in_configurationsElement->FirstChildElement("RunParameters");
    ReadNumOfRuns(runParametersElement);

    ReadXMLSpecific(runParametersElement);
}

/************************************************************************/
void XML_Configuration::ReadTechnicalParameters(const XMLElement* in_configurationsElement)
/************************************************************************/
{
    //1. Init 'technicalParametersElement'
    const XMLElement* technicalParametersElement = in_configurationsElement->FirstChildElement("TechnicalParameters");

    //2. Read 'IterationsFrequencyOfImagePrintsDuringAlg' element
    const XMLElement* iterationsFrequencyOfImagePrintsDuringAlgElement = technicalParametersElement->FirstChildElement("IterationsFrequencyOfImagePrintsDuringAlg");
    if (nullptr != iterationsFrequencyOfImagePrintsDuringAlgElement)
        iterationsFrequencyOfImagePrintsDuringAlgElement->QueryIntText(&TechnicalParameters::m_iterationsFrequencyOfImagePrintsDuringAlg);
}

/************************************************************************/
void XML_Configuration::ReadNumOfRuns(const XMLElement* in_element)
/************************************************************************/
{
    const XMLElement* numOfRunsElement = in_element->FirstChildElement("NumOfRuns");
    if (nullptr != numOfRunsElement)
        numOfRunsElement->QueryIntText(&m_numOfRuns);
}