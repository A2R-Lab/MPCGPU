#pragma once
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mpcgpu {
// Strict rectangular CSV input. Reject bad data before any CUDA allocation.
template<class T>
std::vector<T> readTrajectory(const std::string& path, size_t width) {
    std::ifstream input(path);
    if (!input) throw std::invalid_argument("Cannot open trajectory: " + path);
    std::vector<T> result;
    std::string line;
    size_t row = 0;
    while (std::getline(input, line)) {
        ++row;
        std::stringstream stream(line);
        std::string field;
        size_t columns = 0;
        while (std::getline(stream, field, ',')) {
            std::stringstream cell(field);
            T value;
            if (!(cell >> value) || !std::isfinite(value))
                throw std::invalid_argument("Invalid trajectory value: " + path + ":" + std::to_string(row));
            cell >> std::ws;
            if (!cell.eof()) throw std::invalid_argument("Trailing text in trajectory: " + path);
            result.push_back(value);
            ++columns;
        }
        if (columns != width || (!line.empty() && line.back() == ','))
            throw std::invalid_argument("Wrong trajectory row width: " + path + ":" + std::to_string(row));
    }
    if (result.empty()) throw std::invalid_argument("Empty trajectory: " + path);
    return result;
}

template<class T> struct Reference {
    std::vector<T> ee, xu;
    size_t steps;
    Reference(const std::string& prefix, size_t knots, size_t states, size_t controls)
        : ee(readTrajectory<T>(prefix+"_eepos.traj",6)),
          xu(readTrajectory<T>(prefix+"_traj.csv",states+controls)), steps(ee.size()/6) {
        if (!knots || steps < knots || xu.size()/(states+controls) != steps)
            throw std::invalid_argument("Reference trajectories must have matching rows and cover the horizon");
    }
};
} // namespace mpcgpu
