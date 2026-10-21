#include "preComputation.H"
#include "profileFunction.H"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include <Eigen/Dense>
#include <cantera/core.h>
#include <nlohmann/json.hpp>

#include "tool_Function.H"

using namespace Eigen;
using namespace Cantera;
using namespace std;

namespace {

void requirePositiveFinite(double value, const char* name)
{
    if (!(value > 0.0) || !std::isfinite(value)) {
        throw runtime_error(string(name) + " must be positive and finite");
    }
}

void requireNonnegativeFinite(double value, const char* name)
{
    if (value < 0.0 || !std::isfinite(value)) {
        throw runtime_error(string(name) + " must be non-negative and finite");
    }
}

void requireFinite(double value, const char* name)
{
    if (!std::isfinite(value)) {
        throw runtime_error(string(name) + " must be finite");
    }
}

} // namespace

InputData preComputation(nlohmann::json& j, bool use_steady_grid)
{
    InputData input;
    input.steady_cell = j.at("steady_cell").get<int>();
    input.unsteady_cell = j.at("unsteady_cell").get<int>();
    if (input.steady_cell < 2 || input.unsteady_cell < 2) {
        throw runtime_error("steady_cell and unsteady_cell must both be >= 2");
    }
    input.ncell = use_steady_grid ? input.steady_cell : input.unsteady_cell;
    input.nface = input.ncell + 1;
    input.mechanism = j.at("mechanism").get<string>();
    auto sol = newSolution(input.mechanism);
    auto gas = sol->thermo();
    input.nSpecies = static_cast<int>(gas->nSpecies());
    input.nVar = 3 + input.nSpecies;

    input.duct_constant_length = j.at("duct_constant_length").get<double>();
    input.duct_expansion_length = j.at("duct_expansion_length").get<double>();
    input.duct_expansion_ratio = j.at("duct_expansion_ratio").get<double>();
    input.outlet_section_length = j.at("outlet_section_length").get<double>();
    input.outlet_section_expansion_ratio = j.at("outlet_section_expansion_ratio").get<double>();
    input.outlet_section_exit_norm_slope = j.at("outlet_section_exit_norm_slope").get<double>();
    input.area_inlet = j.at("area_inlet").get<double>();
    requireNonnegativeFinite(input.duct_constant_length,
                             "duct_constant_length");
    requirePositiveFinite(input.duct_expansion_length,
                          "duct_expansion_length");
    requirePositiveFinite(input.duct_expansion_ratio,
                          "duct_expansion_ratio");
    requirePositiveFinite(input.outlet_section_length, "outlet_section_length");
    requirePositiveFinite(input.outlet_section_expansion_ratio,
                          "outlet_section_expansion_ratio");
    requireFinite(input.outlet_section_exit_norm_slope, "outlet_section_exit_norm_slope");
    requirePositiveFinite(input.area_inlet, "area_inlet");

    input.duct_length = input.duct_constant_length
        + input.duct_expansion_length;
    input.duct_constant_area = input.area_inlet;
    input.duct_end_area = input.area_inlet * input.duct_expansion_ratio;
    input.outlet_section_end_area = input.duct_end_area * input.outlet_section_expansion_ratio;
    input.duct_exit_slope =
        (input.duct_end_area - input.duct_constant_area)
        / input.duct_expansion_length;
    const double outlet_section_area_change =
        input.outlet_section_end_area - input.duct_end_area;
    const double area_scale = max(1.0, abs(input.duct_end_area));
    if (abs(outlet_section_area_change) > 1.0e-14 * area_scale) {
        input.duct_exit_norm_slope =
            input.duct_exit_slope * input.outlet_section_length
            / outlet_section_area_change;
    } else if (abs(input.duct_exit_slope) <= 1.0e-14 * area_scale) {
        input.duct_exit_norm_slope = 0.0;
    } else {
        throw runtime_error(
            "outlet_section area change is zero but duct exit slope is nonzero");
    }

    const double domain_length = input.outlet_section_length
        + input.duct_constant_length + input.duct_expansion_length;
    input.xface = VectorXd::LinSpaced(input.nface, 0.0, domain_length);
    input.xcell = 0.5 * (input.xface.head(input.ncell)
                         + input.xface.tail(input.ncell));
    input.A_array = MatrixXd::Zero(1, input.ncell);
    input.A_face_array = MatrixXd::Zero(1, input.nface);
    input.dA_dx = MatrixXd::Zero(1, input.ncell);
    input.dlnA_dx = MatrixXd::Zero(1, input.ncell);
    A_profile_calculation(input);

    input.fuelMoleFraction = j.at("fuelMoleFraction").get<string>();
    input.airMoleFraction = j.at("airMoleFraction").get<string>();
    input.pressure_inlet = j.at("pressure_inlet").get<double>();
    input.temperature_inlet = j.at("temperature_inlet").get<double>();
    input.massflow_inlet = j.at("massflow_inlet").get<double>();
    input.Height_inlet = j.at("Height_inlet").get<double>();
    requirePositiveFinite(input.pressure_inlet, "pressure_inlet");
    requirePositiveFinite(input.temperature_inlet, "temperature_inlet");
    requireNonnegativeFinite(input.massflow_inlet, "massflow_inlet");
    requirePositiveFinite(input.Height_inlet, "Height_inlet");
    input.crosswidth = input.area_inlet / input.Height_inlet;
    input.D_array = 2 * input.A_array.array()
        / (input.crosswidth + input.A_array.array() / input.crosswidth);

    gas->setState_TPX(input.temperature_inlet, input.pressure_inlet,
                      input.airMoleFraction);
    input.Y_inlet = MatrixXd::Zero(input.nSpecies, 1);
    gas->getMassFractions(input.Y_inlet.data());
    input.density_inlet = gas->density();
    input.internal_energy_inlet = gas->intEnergy_mass();
    input.velocity_inlet = input.massflow_inlet
        / input.density_inlet / input.area_inlet;
    input.Ma_inlet = input.velocity_inlet / gas->soundSpeed();
    input.type_inlet = input.Ma_inlet > 1.0 ? "supersonic" : "subsonic";

    input.injection_enabled = j.value("injection_enabled", false);
    input.location_injection = j.at("location_injection").get<double>();
    input.velocity_injection = j.at("velocity_injection").get<double>();
    input.angle_injection = j.at("angle_injection").get<double>();
    input.temperature_injection = j.at("temperature_injection").get<double>();
    input.pressure_injection = j.value("pressure_injection", 101325.0);
    requireNonnegativeFinite(input.location_injection, "location_injection");
    requireNonnegativeFinite(input.velocity_injection, "velocity_injection");
    requireFinite(input.angle_injection, "angle_injection");
    requirePositiveFinite(input.temperature_injection,
                          "temperature_injection");
    requirePositiveFinite(input.pressure_injection, "pressure_injection");
    gas->setState_TPX(input.temperature_injection, input.pressure_injection,
                      input.fuelMoleFraction);
    input.e_fuel = gas->intEnergy_mass();
    input.density_injection = gas->density();
    input.Y_injection = MatrixXd::Zero(input.nSpecies, 1);
    gas->getMassFractions(input.Y_injection.data());
    input.equivalence_ratio = j.at("equivalence_ratio").get<double>();
    requireNonnegativeFinite(input.equivalence_ratio, "equivalence_ratio");
    input.FA_ratio = get_fuel_air_ratio(
        input.fuelMoleFraction, input.airMoleFraction,
        input.equivalence_ratio, *gas);
    requireNonnegativeFinite(input.FA_ratio, "computed fuel-air ratio");
    input.massflow_injection = input.massflow_inlet * input.FA_ratio;
    auto [mdot_profile, dmdot_dx, dmdot_species_dx] =
        mdot_profile_calculation(
            input.massflow_inlet, input.massflow_injection,
            input.location_injection, input.Y_injection,
            input.nSpecies, input.xcell, input.xface);
    input.mdot_profile = mdot_profile;
    input.dmdot_dx = dmdot_dx;
    input.dmdot_species_dx = dmdot_species_dx;

    input.t_final = j.at("t_final").get<double>();
    input.dt = j.at("dt").get<double>();
    input.friction_coefficient = j.at("friction_coefficient").get<double>();
    input.friction_energy_model =
        j.value("friction_energy_model", std::string("legacy_power_sink"));
    input.reconstruction_variables =
        j.value("reconstruction_variables", std::string("conservative_per_area"));
    input.cfl_allowance = j.at("cfl_allowance").get<double>();
    input.error_tolerance = j.value("error_tolerance", 1e-4);
    input.steady_iter_max = j.value("steady_iter_max", 20000);
    requirePositiveFinite(input.t_final, "t_final");
    requirePositiveFinite(input.dt, "dt");
    requireNonnegativeFinite(input.friction_coefficient,
                             "friction_coefficient");
    if (input.friction_energy_model != "adiabatic"
        && input.friction_energy_model != "legacy_power_sink") {
        throw runtime_error(
            "friction_energy_model must be adiabatic or legacy_power_sink");
    }
    if (input.reconstruction_variables != "primitive"
        && input.reconstruction_variables != "conservative_per_area") {
        throw runtime_error(
            "reconstruction_variables must be primitive or conservative_per_area");
    }
    requirePositiveFinite(input.cfl_allowance, "cfl_allowance");
    requirePositiveFinite(input.error_tolerance, "error_tolerance");
    if (input.steady_iter_max <= 0) {
        throw runtime_error("steady_iter_max must be positive");
    }

    input.alpha_mix = j.at("alpha_mix").get<double>();
    input.length_mix = j.at("length_mix").get<double>();
    requirePositiveFinite(input.alpha_mix, "alpha_mix");
    requirePositiveFinite(input.length_mix, "length_mix");
    mix_profile_calculation(input);
    input.fullmix_flag = j.at("fullmix_flag").get<bool>();
    input.massflow_buildup_time = j.at("massflow_buildup_time").get<double>();
    input.mix_buildup_time = j.at("mix_buildup_time").get<double>();
    requirePositiveFinite(input.massflow_buildup_time,
                          "massflow_buildup_time");
    requirePositiveFinite(input.mix_buildup_time, "mix_buildup_time");
    input.mdot_profile_by_time = MatrixXd::Zero(1, input.ncell);
    input.dmdot_dx_by_time = MatrixXd::Zero(1, input.ncell);
    input.dmdot_species_dx_by_time =
        MatrixXd::Zero(input.nSpecies, input.ncell);
    if (use_steady_grid && input.injection_enabled) {
        input.mdot_profile_by_time = input.mdot_profile;
        input.dmdot_dx_by_time = input.dmdot_dx;
        input.dmdot_species_dx_by_time = input.dmdot_species_dx;
    }
    input.mix_by_time = MatrixXd::Zero(1, input.ncell);

    input.ambient_altitude_km = j.at("ambient_altitude_km").get<double>();
    const auto [T_env, P_env] = ambient_conditions_tp(
        input.ambient_altitude_km, input.airMoleFraction, *gas);
    input.temperature_environment = j.value("temperature_environment", T_env);
    input.pressure_environment = j.value("pressure_environment", P_env);
    requirePositiveFinite(input.temperature_environment,
                          "temperature_environment");
    requirePositiveFinite(input.pressure_environment,
                          "pressure_environment");

    if (!input.A_array.allFinite() || !input.A_face_array.allFinite()
        || (input.A_array.array() <= 0.0).any()
        || (input.A_face_array.array() <= 0.0).any()) {
        throw runtime_error("computed duct area must be positive and finite");
    }
    return input;
}
