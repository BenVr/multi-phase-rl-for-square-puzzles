#pragma once

#include <string>
#include <tinyxml2.h>
#include "RelaxationLabelingSolverBaseConstants.h"

using namespace tinyxml2;

class XML_Configuration
{
public:
    XML_Configuration(const std::string& in_configXML_Path);
    virtual ~XML_Configuration() = default;

    virtual void ReadXMLSpecific(const XMLElement* in_configurationsElement) = 0;
    
    void ReadXML();
    int32_t GetNumOfRuns() const { return m_numOfRuns; };
    
protected:
    void ReadRunParameters(const XMLElement* in_configurationsElement);
    void ReadTechnicalParameters(const XMLElement* in_configurationsElement);

    void ReadNumOfRuns(const XMLElement* in_element);

    const std::string& m_xmlPath;
    int32_t m_numOfRuns = 1;
};