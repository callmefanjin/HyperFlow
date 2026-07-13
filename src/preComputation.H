#ifndef PRE_COMPUTATION_H
#define PRE_COMPUTATION_H

#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <string>

/**
 * Runtime case data: A(x), time step, mechanism, inlet/outlet BC.
 * Geometry and initial U come from external HDF5 (no built-in duct generator).
 */
struct InputData
{
    std::string mechanism;
    bool chemistry = true;

    int ncell = 0;
    int nface = 0;
    int nVar = 0;
    int nSpecies = 0;

    double t_final = 0.0;
    double dt = 0.0;

    Eigen::MatrixXd A_array;   // 1 × nface
    Eigen::MatrixXd dA_dx;     // 1 × nface
    Eigen::MatrixXd dlnA_dx;   // 1 × nface
    Eigen::VectorXd xface;     // nface

    /** Inlet reservoir for Riemann BC */
    std::string inletMoleFraction;
    double pressure_inlet = 0.0;
    double temperature_inlet = 0.0;
    double velocity_inlet = 0.0;

    /** Outlet ambient for subsonic back-pressure Riemann BC */
    std::string outletMoleFraction;
    double pressure_environment = 0.0;
    double temperature_environment = 0.0;
};

/**
 * Load config.json + initial.h5 into InputData; return initial U.
 * initial.h5 must provide datasets x, A, U.
 */
InputData loadCase(const std::string & case_dir,
                   const nlohmann::json & j,
                   Eigen::MatrixXd & U0);

/** Form dA/dx and dlnA/dx from x and A (central; one-sided at ends) */
void computeAreaDerivatives(InputData & input);

#endif
