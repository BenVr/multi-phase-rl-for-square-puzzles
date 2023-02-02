#pragma once

#include <vector>
#include <set>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <atomic>

#include <Eigen/Core>

enum Orientation
{
    eInvalidOrientation = -1,
    eUp = 0, 
    eRight,
    eDown,
    eLeft,
    eNumOfPossibleOrientations = 4
};

class XML_Configuration;

class TechnicalParameters
{
    friend class XML_Configuration;

public:
    static int32_t GetIterationsFrequencyOfImagePrintsDuringAlg() {return m_iterationsFrequencyOfImagePrintsDuringAlg;}

protected:
    static int32_t m_iterationsFrequencyOfImagePrintsDuringAlg;
};

using BooleansMatrix = Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>;

namespace Utilities {
    
    class IndexCounter
    {
    public:
        IndexCounter() = default;
        int32_t GetNextIndex() {return m_index++;}
        void Zero() { m_index = 0; };
    
    protected:
        int32_t m_index = 0;
    };

    class Timer
    {
    public:
        Timer() : m_start(std::chrono::steady_clock::now()) {}
        long long GetSecondsPassed() const
        {
           return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - m_start).count();
        }

        long long GetMilliSecondsPassed() const
        {
            return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - m_start).count();
        }

        const std::chrono::nanoseconds GetTime() const
        {
            std::chrono::nanoseconds duration = std::chrono::steady_clock::now() - m_start;
            return duration;
        }

        static const std::string GetTimeString(const std::chrono::nanoseconds& in_duration)
        {
            //This function was copied from Stack Overflow
            std::chrono::nanoseconds duration = in_duration;
            const std::chrono::hours hh = std::chrono::duration_cast<std::chrono::hours>(duration);
            duration -= hh;
            const std::chrono::minutes mm = std::chrono::duration_cast<std::chrono::minutes>(duration);
            duration -= mm;
            const std::chrono::seconds ss = std::chrono::duration_cast<std::chrono::seconds>(duration);
            duration -= ss;
            const std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);

            std::stringstream stream;
            stream << std::setfill('0') << std::setw(2) << hh.count() << ':' <<
                std::setfill('0') << std::setw(2) << mm.count() << ':' <<
                std::setfill('0') << std::setw(2) << ss.count() << '.' <<
                std::setfill('0') << std::setw(3) << ms.count();

            return stream.str();
        }

    protected:
        const std::chrono::steady_clock::time_point m_start;
    };

    static constexpr bool WITH_SCREEN_LOGS = true;

    class Logger
    {
    public:
        Logger(std::ostream& in_ostream = std::cout) : m_ostream(in_ostream) {}

        void SetAnotherStream(const std::string& in_filePath)
        {
            static std::ofstream fileStream(in_filePath);
            m_pOtherOstream = &fileStream;
        }

        template<typename T>
        Logger& operator<<(const T& data)
        {
            if constexpr (WITH_SCREEN_LOGS)
            {
                m_ostream << data;
                m_ostream.flush();
            }

            if (m_pOtherOstream)
            {
                (*m_pOtherOstream) << data;
                m_pOtherOstream->flush();
            }

            return *this;
        }

        //This is the function signature of std::endl
        typedef std::basic_ostream<char, std::char_traits<char> > CoutType;
        typedef CoutType& (*StandardEndLine)(CoutType&);

        //Define an operator<< to take in std::endl
        Logger& operator<<(StandardEndLine manip)
        {
            if constexpr (WITH_SCREEN_LOGS)
            {
                manip(m_ostream);
            }

            if (m_pOtherOstream)
                manip(*m_pOtherOstream);

            return *this;
        }

        template<typename T>
        void BruteLog(const T& data) const
        {
            m_ostream << data;
            m_ostream.flush();

            if (m_pOtherOstream)
            {
                (*m_pOtherOstream) << data;
                m_pOtherOstream->flush();
            }
        }

        void BruteLog(StandardEndLine manip) const
        {
            manip(m_ostream);

            if (m_pOtherOstream)
                manip(*m_pOtherOstream);
        }

    protected:
        std::ostream& m_ostream;
        std::ostream* m_pOtherOstream = nullptr;
    };

    struct RowColInfo
    {
        RowColInfo(const int32_t in_row, const int32_t in_col) : m_row(in_row), m_col(in_col) {}

        bool operator<(const RowColInfo& in_other) const;
        bool operator==(const RowColInfo& in_other) const {return m_row == in_other.m_row && m_col == in_other.m_col;}
        
        bool IsValid() const {return m_row >= 0 && m_col >= 0;}
        bool IsValid(const int32_t in_numRows, const int32_t in_numCols) const {return IsValid() && m_row < in_numRows && m_col < in_numCols;}
        std::string GetString() const {return "row: " + std::to_string(m_row) + ", col: " + std::to_string(m_col);}

        int32_t m_row;
        int32_t m_col;

        static constexpr int32_t invalidIndex = -1;

        static RowColInfo GetRowAndColFrom1DIndex(const size_t in_1D_ZeroBasedIndex, const size_t in_totalNumOfCols)
        {
            return GetRowAndColFrom1DIndex(static_cast<int32_t>(in_1D_ZeroBasedIndex), static_cast<int32_t>(in_totalNumOfCols));
        }

        static RowColInfo GetRowAndColFrom1DIndex(const int32_t in_1D_ZeroBasedIndex, const int32_t in_totalNumOfCols)
        {
            const int32_t row = in_1D_ZeroBasedIndex / in_totalNumOfCols;
            const int32_t col = in_1D_ZeroBasedIndex % in_totalNumOfCols;
            return RowColInfo(row, col);
        }

        static bool AreNeighbors(const RowColInfo& in_lhs, const RowColInfo& in_rhs);
        static RowColInfo NeighborCoordByOrien(const int32_t in_row, const int32_t in_col, const Orientation in_orien);

        static RowColInfo GetTopNeighborCoord(const int32_t in_row, const int32_t in_col) {return RowColInfo(in_row - 1, in_col);}
        static RowColInfo GetBottomNeighborCoord(const int32_t in_row, const int32_t in_col) {return RowColInfo(in_row + 1, in_col);}
        static RowColInfo GetLeftNeighborCoord(const int32_t in_row, const int32_t in_col) {return RowColInfo(in_row, in_col - 1);}
        static RowColInfo GetRightNeighborCoord(const int32_t in_row, const int32_t in_col) {return RowColInfo(in_row, in_col + 1);}

        static const RowColInfo m_invalidRowColInfo;
    };

    using GraphPoint = std::pair<int, double>; 
    using GraphData = std::vector<GraphPoint>;

    void LogAndAbort(const std::string& in_str);
    void LogAndAbortIf(const bool in_condition, const std::string& in_str);
    void Log(const std::string& in_str);

    double RoundWithPrecision(const double in_val, const int32_t in_numOfDigits);
    bool IsDoublyStochasticMatrixWithPrecision(const Eigen::MatrixXd& in_matrix, const int32_t in_numOfDigits, 
        const bool in_shouldAbort, const bool in_shouldLog = true);

    Eigen::MatrixXd ComputeInverseOfCovarianceMatrix(const Eigen::MatrixXd& in_samplesMatrix);

    inline void UpdateAtomicMinimum(std::atomic<double>& in_minValue, const double& in_newValue)
    {
        double prevValue = in_minValue;
        while(prevValue > in_newValue &&
                !in_minValue.compare_exchange_weak(prevValue, in_newValue))
            {}
    }

    inline void UpdateAtomicMaximum(std::atomic<double>& in_maxValue, const double& in_newValue)
    {
        double prevValue = in_maxValue;
        while(prevValue < in_newValue &&
                !in_maxValue.compare_exchange_weak(prevValue, in_newValue))
            {}
    }

    template<typename T>
    bool IsItemInSet(const T& in_item, const std::set<T>& in_set) {return in_set.end() != std::find(in_set.begin(), in_set.end(), in_item);};

    template<typename T>
    bool IsItemInVector(const T& in_item, const std::vector<T>& in_vector) {return in_vector.end() != std::find(in_vector.begin(), in_vector.end(), in_item);};
}

using RowColInfo = Utilities::RowColInfo;
using RowColInfoSet = std::set<RowColInfo>;
