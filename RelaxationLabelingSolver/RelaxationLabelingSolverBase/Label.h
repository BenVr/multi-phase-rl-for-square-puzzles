#pragma once

#include <sstream>
#include "Utilities.h"

class Label
{
public:
    Label() : m_index(m_index_counter.GetNextIndex()) {}
    virtual ~Label() = default;

    bool operator<(const Label& in_other) const { return m_index < in_other.m_index; };
    bool operator==(const Label& in_other) const { return m_index == in_other.m_index; };
    bool operator!=(const Label& in_other) const { return !(*this == in_other); };

    virtual std::string GetString() const = 0;

    const int32_t m_index;

    static void ZeroIndexCounter() { Label::m_index_counter.Zero(); }

protected:
    static Utilities::IndexCounter m_index_counter;
};

using LabelsVector = std::vector<const Label*>;