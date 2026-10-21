#include "preComputation.H"
#include "tool_Function.H"

#include <cantera/core.h>

#include <cmath>
#include <stdexcept>
#include <string>

using namespace Eigen;
using namespace Cantera;
using namespace std;

void computeAreaDerivatives(InputData & input)
{
    const int n = input.nface;
    if (n < 2) {
        throw runtime_error("nface must be >= 2");
    }
    input.dA_dx = MatrixXd::Zero(1, n);
    input.dlnA_dx = MatrixXd::Zero(1, n);

    input.dA_dx(0, 0) =
        (input.A_array(0, 1) - input.A_array(0, 0)) / (input.xface(1) - input.xface(0));
    for (int i = 1; i < n - 1; ++i) {
        input.dA_dx(0, i) =
            (input.A_array(0, i + 1) - input.A_array(0, i - 1))
            / (input.xface(i + 1) - input.xface(i - 1));
    }
    input.dA_dx(0, n - 1) =
        (input.A_array(0, n - 1) - input.A_array(0, n - 2))
        / (input.xface(n - 1) - input.xface(n - 2));

    for (int i = 0; i < n; ++i) {
        const double A = input.A_array(0, i);
        if (A <= 0.0) {
            throw runtime_error("A must be positive at all faces");
        }
        input.dlnA_dx(0, i) = input.dA_dx(0, i) / A;
    }
}

InputData loadCase(const string & case_dir,
                   const nlohmann::json & j,
                   MatrixXd & U0)
{
    InputData input;
    input.mechanism = j.at("mechanism").get<string>();
    input.chemistry = j.value("chemistry", true);
    input.dt = j.at("dt").get<double>();
    input.t_final = j.at("t_final").get<double>();

    const auto & inlet = j.at("boundary").at("inlet");
    input.pressure_inlet = inlet.at("p").get<double>();
    input.temperature_inlet = inlet.at("T").get<double>();
    input.velocity_inlet = inlet.at("u").get<double>();
    input.inletMoleFraction = inlet.at("X").get<string>();

    const auto & outlet = j.at("boundary").at("outlet");
    input.pressure_environment = outlet.at("p").get<double>();
    input.temperature_environment = outlet.value("T", inlet.at("T").get<double>());
    input.outletMoleFraction = outlet.value("X", input.inletMoleFraction);

    const string initial_name = j.value("initial", string("initial.h5"));
    const string initial_path = case_dir + "/" + initial_name;
    const StateSnapshot snap = readStateHDF5(initial_path);

    if (snap.xface.size() == 0 || snap.A_array.size() == 0 || snap.U.size() == 0) {
        throw runtime_error(
            "initial HDF5 must contain datasets x, A, U: " + initial_path);
    }

    input.xface = snap.xface;
    input.nface = static_cast<int>(snap.xface.size());
    input.ncell = input.nface - 1;
    input.A_array = snap.A_array;
    if (input.A_array.rows() != 1) {
        // accept column vector stored as nface×1
        if (input.A_array.cols() == 1 && input.A_array.rows() == input.nface) {
            input.A_array = input.A_array.transpose();
        } else if (input.A_array.rows() == 1 && input.A_array.cols() == input.nface) {
            // ok
        } else {
            throw runtime_error("A shape incompatible with x");
        }
    }
    if (input.A_array.cols() != input.nface) {
        throw runtime_error("A length != nface");
    }

    U0 = snap.U;
    auto sol = newSolution(input.mechanism);
    auto gas = sol->thermo();
    input.nSpecies = static_cast<int>(gas->nSpecies());
    input.nVar = 3 + input.nSpecies;

    if (U0.rows() != input.nVar || U0.cols() != input.nface) {
        throw runtime_error(
            "U shape mismatch: got (" + to_string(U0.rows()) + "," + to_string(U0.cols())
            + "), expected (" + to_string(input.nVar) + "," + to_string(input.nface)
            + ") for mechanism " + input.mechanism);
    }

    computeAreaDerivatives(input);
    return input;
}
