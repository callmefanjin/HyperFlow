#include "profileFunction.H"

#include <cmath>
#include <stdexcept>
#include <utility>

using namespace Eigen;

namespace {

double injectionRamp(const double x, const double injection_location,
                     const double transition_length)
{
    if (x <= injection_location) {
        return 0.0;
    }
    if (x >= injection_location + transition_length) {
        return 1.0;
    }
    const double xi = (x - injection_location) / transition_length;
    return 3.0 * xi * xi - 2.0 * xi * xi * xi;
}

std::pair<double, double> areaAndDerivative(const InputData& input,
                                            const double x)
{
    if (x <= input.duct_constant_length) {
        return {input.area_inlet, 0.0};
    }
    if (x <= input.duct_length) {
        const double slope = (input.duct_end_area - input.duct_constant_area)
            / input.duct_expansion_length;
        return {input.duct_constant_area
                    + (x - input.duct_constant_length) * slope,
                slope};
    }
    const double xi = (x - input.duct_length) / input.outlet_section_length;
    const double shape = -2.0 * std::pow(xi, 3) + 3.0 * std::pow(xi, 2)
        + input.duct_exit_norm_slope
            * (std::pow(xi, 3) - 2.0 * std::pow(xi, 2) + xi)
        + input.outlet_section_exit_norm_slope
            * (std::pow(xi, 3) - std::pow(xi, 2));
    const double dshape = -6.0 * xi * xi + 6.0 * xi
        + input.duct_exit_norm_slope
            * (3.0 * xi * xi - 4.0 * xi + 1.0)
        + input.outlet_section_exit_norm_slope * (3.0 * xi * xi - 2.0 * xi);
    return {input.duct_end_area
                + (input.outlet_section_end_area - input.duct_end_area) * shape,
            (input.outlet_section_end_area - input.duct_end_area)
                / input.outlet_section_length * dshape};
}

} // namespace

std::tuple<MatrixXd, MatrixXd, MatrixXd> mdot_profile_calculation(
    const double massflow_inlet,
    const double massflow_injection,
    const double injection_location,
    const MatrixXd& Y_injection,
    const int nSpecies,
    const VectorXd& xcell,
    const VectorXd& xface)
{
    if (xface.size() != xcell.size() + 1) {
        throw std::runtime_error("injection profile requires ncell+1 faces");
    }
    constexpr double transition_length = 0.05;
    MatrixXd mdot_profile(1, xcell.size());
    VectorXd mdot_face(xface.size());
    for (int face = 0; face < xface.size(); ++face) {
        mdot_face(face) = massflow_inlet + massflow_injection
            * injectionRamp(xface(face), injection_location, transition_length);
    }

    MatrixXd dmdot_dx(1, xcell.size());
    for (int cell = 0; cell < xcell.size(); ++cell) {
        mdot_profile(0, cell) = massflow_inlet + massflow_injection
            * injectionRamp(xcell(cell), injection_location, transition_length);
        const double dx = xface(cell + 1) - xface(cell);
        if (!(dx > 0.0)) {
            throw std::runtime_error("non-positive cell width in injection profile");
        }
        // Conservative finite-volume difference of the prescribed cumulative
        // mass-flow profile. Its cell integral telescopes exactly.
        dmdot_dx(0, cell) = (mdot_face(cell + 1) - mdot_face(cell)) / dx;
    }

    MatrixXd dmdot_species_dx = MatrixXd::Zero(nSpecies, xcell.size());
    for (int species = 0; species < nSpecies; ++species) {
        dmdot_species_dx.row(species)
            = dmdot_dx.array() * Y_injection(species);
    }
    return {mdot_profile, dmdot_dx, dmdot_species_dx};
}

void mix_profile_calculation(InputData& input)
{
    input.mix_array = MatrixXd::Zero(1, input.ncell);
    for (int cell = 0; cell < input.ncell; ++cell) {
        const double x = input.xcell(cell);
        if (x < input.location_injection) {
            input.mix_array(cell) = 0.0;
        } else if (x > input.location_injection + input.length_mix) {
            input.mix_array(cell) = 1.0;
        } else {
            const double xi = (x - input.location_injection) / input.length_mix;
            input.mix_array(cell) = (1.0 - std::exp(-input.alpha_mix * xi))
                / (1.0 - std::exp(-input.alpha_mix));
        }
    }
}

void A_profile_calculation(InputData& input)
{
    for (int cell = 0; cell < input.ncell; ++cell) {
        const auto [area, derivative] = areaAndDerivative(input, input.xcell(cell));
        (void)derivative;
        input.A_array(0, cell) = area;
    }
    for (int face = 0; face < input.nface; ++face) {
        input.A_face_array(0, face)
            = areaAndDerivative(input, input.xface(face)).first;
    }
    for (int cell = 0; cell < input.ncell; ++cell) {
        const double dx = input.xface(cell + 1) - input.xface(cell);
        if (!(dx > 0.0)) {
            throw std::runtime_error("non-positive cell width in area profile");
        }
        // Finite-volume cell average of dA/dx. This uses the same face
        // areas as the flux divergence, so the geometric momentum source
        // integrates exactly to p_i * (A_{i+1/2} - A_{i-1/2}).
        input.dA_dx(0, cell) =
            (input.A_face_array(0, cell + 1)
             - input.A_face_array(0, cell)) / dx;
    }
    input.dlnA_dx = input.dA_dx.array() / input.A_array.array();
}
