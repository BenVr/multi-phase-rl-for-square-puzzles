#pragma once

#include <string>
#include <tinyxml2.h>

using namespace tinyxml2;

namespace InputInfo
{
    class ImageInputData
    {
    public:
        void ReadImageInput(const XMLElement* in_imageInputElement);

        int32_t GetPieceSize() const { return m_pieceSize;}
        std::string GetSourceImageName() const { return m_sourceImageName;}
        std::string GetRunAllFolderName() const { return m_runAllFolderName;}

    protected:
        void ReadSourceImageOrRunAllFolder(const XMLElement* in_imageInputElement);

        int32_t m_pieceSize = 0;
        std::string m_sourceImageName;
        std::string m_runAllFolderName;
    };
};

using ImageInputData = InputInfo::ImageInputData;
