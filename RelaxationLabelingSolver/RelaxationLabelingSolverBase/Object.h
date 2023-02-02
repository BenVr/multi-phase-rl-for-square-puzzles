#pragma once

#include <sstream>
#include "Utilities.h"

class Object
{
public:
    Object() : m_index(m_index_counter.GetNextIndex()) {}
    virtual ~Object() = default;

    bool operator<(const Object& in_other) const { return m_index < in_other.m_index; };
    bool operator==(const Object& in_other) const { return m_index == in_other.m_index; };
    bool operator!=(const Object& in_other) const { return !(*this == in_other); };

    virtual std::string  GetString() const = 0;

    const int32_t m_index;

    static void ZeroIndexCounter() { Object::m_index_counter.Zero(); }

protected:
    static Utilities::IndexCounter m_index_counter;
};

using ObjectsVector = std::vector<const Object*>;
