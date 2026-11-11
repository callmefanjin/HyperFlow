#pragma once
#include "solver.hpp"
#include <memory>
#include <string>

// One independently loaded Cantera phase and one CVODE context per worker.
// A batch belongs to one caller; do not invoke it concurrently/nested.
class ChemistryBatch {
public:
    explicit ChemistryBatch(const std::string& mechanism, int threads=0);
    void advance(Field& field, double dt, size_t count);
    int threads() const { return workers.size(); }
    int lastTeamSize() const { return lastTeam; }
private:
    std::vector<std::unique_ptr<Chemistry>> workers;
    int lastTeam=1;
};
