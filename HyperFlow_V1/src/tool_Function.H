#ifndef TOOL_FUNCTION_H
#define TOOL_FUNCTION_H

#include <Eigen/Dense>

#include <string>
#include <vector>

namespace Cantera
{
class ThermoPhase;
}

struct InputData;

/** HDF5 snapshot metadata */
struct StateSnapshotMeta
{
    std::string kind = "unsteady";
    double time = 0.0;
    bool converged = true;
    int steady_iter = 0;
    double steady_error = 0.0;
};

/**
 * Write field to HDF5 (conserved U + primitives; restart / post-processing).
 * Datasets: x, A, U, rho, p, u, T, Mach, e, Y, dA_dx, dlnA_dx
 */
void writeStateHDF5(const std::string & filename,
                    const StateSnapshotMeta & meta,
                    const InputData & input,
                    const Eigen::VectorXd & xface,
                    const Eigen::MatrixXd & A_array,
                    const Eigen::MatrixXd & U,
                    Cantera::ThermoPhase & gas);

/** HDF5 snapshot returned by readStateHDF5 */
struct StateSnapshot
{
    StateSnapshotMeta meta;
    int ncell = 0;
    int nface = 0;
    int nVar = 0;
    int nSpecies = 0;
    std::string mechanism;
    std::vector<std::string> species_names;
    Eigen::VectorXd xface;
    Eigen::MatrixXd A_array;
    Eigen::MatrixXd U;
    Eigen::MatrixXd rho;
    Eigen::MatrixXd p;
    Eigen::MatrixXd u;
    Eigen::MatrixXd T;
    Eigen::MatrixXd Mach;
    Eigen::MatrixXd e;
    Eigen::MatrixXd Y;
    Eigen::MatrixXd dA_dx;
    Eigen::MatrixXd dlnA_dx;
};

/** Read snapshot; datasets x, A, U are required */
StateSnapshot readStateHDF5(const std::string & filename);

inline constexpr const char kConfigJson[] = "config.json";
inline constexpr const char kInitialH5[] = "initial.h5";

struct CasePaths
{
    std::string case_dir;
    std::string config_json;
    std::string out_dir;
};

/** Resolve case_dir → config.json and out/ (creates out/) */
CasePaths resolveCaseFiles(const std::string & case_dir);

#endif
