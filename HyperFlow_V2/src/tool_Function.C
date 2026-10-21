#include "tool_Function.H"
#include "preComputation.H"
#include "coreMethod.H"

#include <nlohmann/json.hpp>

#include <hdf5.h>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <string>
#include <vector>
#include <filesystem>
#include <cmath>
#include <cantera/core.h>

double get_fuel_air_ratio(const std::string & fuelMoleFraction,
                          const std::string & airMoleFraction,
                          double equivalence_ratio,
                          Cantera::ThermoPhase & gas)
{
    const size_t nsp = gas.nSpecies();

    // 用纯空气识别“空气物种”下标（O2、N2 等 Y>0）
    gas.setState_TPX(300.0, 101325.0, airMoleFraction);

    std::vector<double> Y_air(nsp);
    gas.getMassFractions(Y_air.data());

    std::vector<size_t> air_indices;
    air_indices.reserve(nsp);
    for (size_t k = 0; k < nsp; ++k) {
        if (Y_air[k] > 0.0) {
            air_indices.push_back(k);
        }
    }

    gas.setEquivalenceRatio(equivalence_ratio, fuelMoleFraction, airMoleFraction);

    std::vector<double> Y(nsp);
    gas.getMassFractions(Y.data());
    double sum_air = 0.0;
    for (const size_t k : air_indices) {
        sum_air += Y[k];
    }
    if (!(sum_air > 0.0) || !std::isfinite(sum_air)) {
        throw std::runtime_error(
            "air composition has no positive mass fraction in equivalence-ratio state");
    }

    // 未燃混合气中除空气物种外视为燃料
    const double Y_fuel = 1.0 - sum_air;
    if (Y_fuel < -1.0e-12 || !std::isfinite(Y_fuel)) {
        throw std::runtime_error(
            "fuel and air species definitions overlap inconsistently");
    }
    return std::max(0.0, Y_fuel) / sum_air;
}

std::pair<double, double> ambient_conditions_tp(double H_km,
                                               const std::string & airMoleFraction,
                                               Cantera::ThermoPhase & gas)
{
    if (H_km < 0.0 || H_km > 47.0) {
        throw std::runtime_error(
            "高度 " + std::to_string(H_km) + " km 超出 ISA 模型上限 (47 km)。");
    }

    gas.setState_TPX(300.0, 101325.0, airMoleFraction);
    const double R_air = Cantera::GasConstant / gas.meanMolecularWeight();
    const double g = 9.81;
    double T = 0.0;
    double P = 0.0;

    if (H_km <= 11.0) {
        T = 288.15 - 6.5 * H_km;
        P = 101325.0 * std::pow(1.0 - 6.5 * H_km / 288.15, g / (R_air * 0.0065));
    } else if (H_km <= 20.0) {
        T = 216.65;
        P = 22632.0 * std::exp(g * (11000.0 - H_km * 1000.0) / (R_air * T));
    } else if (H_km <= 32.0) {
        T = 216.65 + (H_km - 20.0);
        P = 5474.9 * std::pow(T / 216.65, -g / (R_air * 0.001));
    } else {
        T = 228.65 + 2.8 * (H_km - 32.0);
        P = 868.02 * std::pow(T / 228.65, -g / (R_air * 0.0028));
    }

    return {T, P};
}

void writeInputDataJSON(const std::string & filename, const InputData & input)
{
    nlohmann::json j;
    j["mechanism"] = input.mechanism;
    j["fuelMoleFraction"] = input.fuelMoleFraction;
    j["airMoleFraction"] = input.airMoleFraction;
    j["steady_cell"] = input.steady_cell;
    j["unsteady_cell"] = input.unsteady_cell;
    j["nVar"] = input.nVar;
    j["nSpecies"] = input.nSpecies;
    j["crosswidth"] = input.crosswidth;
    j["Height_inlet"] = input.Height_inlet;
    j["massflow_inlet"] = input.massflow_inlet;
    j["pressure_inlet"] = input.pressure_inlet;
    j["temperature_inlet"] = input.temperature_inlet;
    j["density_inlet"] = input.density_inlet;
    j["velocity_inlet"] = input.velocity_inlet;
    j["Ma_inlet"] = input.Ma_inlet;
    j["internal_energy_inlet"] = input.internal_energy_inlet;
    j["area_inlet"] = input.area_inlet;
    j["duct_constant_length"] = input.duct_constant_length;
    j["duct_expansion_length"] = input.duct_expansion_length;
    j["duct_expansion_ratio"] = input.duct_expansion_ratio;
    j["outlet_section_length"] = input.outlet_section_length;
    j["outlet_section_expansion_ratio"] = input.outlet_section_expansion_ratio;
    j["outlet_section_exit_norm_slope"] = input.outlet_section_exit_norm_slope;
    j["duct_length"] = input.duct_length;
    j["duct_constant_area"] = input.duct_constant_area;
    j["duct_end_area"] = input.duct_end_area;
    j["outlet_section_end_area"] = input.outlet_section_end_area;
    j["duct_exit_slope"] = input.duct_exit_slope;
    j["duct_exit_norm_slope"] = input.duct_exit_norm_slope;
    j["type_inlet"] = input.type_inlet;
    j["injection_enabled"] = input.injection_enabled;
    j["location_injection"] = input.location_injection;
    j["velocity_injection"] = input.velocity_injection;
    j["angle_injection"] = input.angle_injection;
    j["temperature_injection"] = input.temperature_injection;
    j["pressure_injection"] = input.pressure_injection;
    j["density_injection"] = input.density_injection;
    j["friction_coefficient"] = input.friction_coefficient;
    j["friction_energy_model"] = input.friction_energy_model;
    j["reconstruction_variables"] = input.reconstruction_variables;
    j["equivalence_ratio"] = input.equivalence_ratio;
    j["FA_ratio"] = input.FA_ratio;
    j["massflow_injection"] = input.massflow_injection;
    j["t_final"] = input.t_final;
    j["dt"] = input.dt;
    j["cfl_allowance"] = input.cfl_allowance;
    j["error_tolerance"] = input.error_tolerance;
    j["steady_iter_max"] = input.steady_iter_max;
    j["alpha_mix"] = input.alpha_mix;
    j["length_mix"] = input.length_mix;
    j["fullmix_flag"] = input.fullmix_flag;
    j["massflow_buildup_time"] = input.massflow_buildup_time;
    j["mix_buildup_time"] = input.mix_buildup_time;
    j["ambient_altitude_km"] = input.ambient_altitude_km;
    j["pressure_environment"] = input.pressure_environment;
    j["temperature_environment"] = input.temperature_environment;

    std::ofstream out(filename);
    if (!out) {
        throw std::runtime_error("Cannot open output file: " + filename);
    }
    out << j.dump(2);
}

namespace
{

void h5_check(hid_t id, const std::string & msg)
{
    if (id < 0) {
        throw std::runtime_error(msg);
    }
}

void h5_write_attr_int(hid_t loc, const char * name, int value)
{
    const hid_t space = H5Screate(H5S_SCALAR);
    h5_check(space, "H5Screate failed for int attribute");
    const hid_t attr = H5Acreate(loc, name, H5T_NATIVE_INT, space, H5P_DEFAULT, H5P_DEFAULT);
    h5_check(attr, "H5Acreate failed for int attribute");
    H5Awrite(attr, H5T_NATIVE_INT, &value);
    H5Aclose(attr);
    H5Sclose(space);
}

void h5_write_attr_double(hid_t loc, const char * name, double value)
{
    const hid_t space = H5Screate(H5S_SCALAR);
    h5_check(space, "H5Screate failed for double attribute");
    const hid_t attr = H5Acreate(loc, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT);
    h5_check(attr, "H5Acreate failed for double attribute");
    H5Awrite(attr, H5T_NATIVE_DOUBLE, &value);
    H5Aclose(attr);
    H5Sclose(space);
}

void h5_write_attr_string(hid_t loc, const char * name, const std::string & value)
{
    const hid_t str_type = H5Tcopy(H5T_C_S1);
    h5_check(str_type, "H5Tcopy failed for string attribute");
    H5Tset_size(str_type, value.size());
    const hid_t space = H5Screate(H5S_SCALAR);
    h5_check(space, "H5Screate failed for string attribute");
    const hid_t attr = H5Acreate(loc, name, str_type, space, H5P_DEFAULT, H5P_DEFAULT);
    h5_check(attr, "H5Acreate failed for string attribute");
    H5Awrite(attr, str_type, value.c_str());
    H5Aclose(attr);
    H5Sclose(space);
    H5Tclose(str_type);
}

void h5_write_vector_1d(hid_t file, const char * name, const double * data, hsize_t n)
{
    const hsize_t dims[1] = {n};
    const hid_t space = H5Screate_simple(1, dims, nullptr);
    h5_check(space, "H5Screate_simple failed for 1D dataset");
    const hid_t dset = H5Dcreate(file, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    h5_check(dset, "H5Dcreate failed for 1D dataset");
    H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    H5Dclose(dset);
    H5Sclose(space);
}

void h5_write_matrix_rowmajor(hid_t file, const char * name, const std::vector<double> & data,
                              hsize_t n_rows, hsize_t n_cols)
{
    const hsize_t dims[2] = {n_rows, n_cols};
    const hid_t space = H5Screate_simple(2, dims, nullptr);
    h5_check(space, "H5Screate_simple failed for 2D dataset");
    const hid_t dset = H5Dcreate(file, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    h5_check(dset, "H5Dcreate failed for 2D dataset");
    H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
    H5Dclose(dset);
    H5Sclose(space);
}

std::vector<double> eigen_to_rowmajor(const Eigen::MatrixXd & mat);

void write_source_profile_attrs(hid_t file, const InputData & input)
{
    h5_write_attr_string(file, "mechanism", input.mechanism);
    h5_write_attr_double(file, "alpha_mix", input.alpha_mix);
    h5_write_attr_double(file, "length_mix", input.length_mix);
    h5_write_attr_int(file, "injection_enabled", input.injection_enabled ? 1 : 0);
    h5_write_attr_double(file, "location_injection", input.location_injection);
    h5_write_attr_double(file, "massflow_injection", input.massflow_injection);
    h5_write_attr_double(file, "velocity_injection", input.velocity_injection);
    h5_write_attr_double(file, "angle_injection", input.angle_injection);
    h5_write_attr_double(file, "temperature_injection", input.temperature_injection);
    h5_write_attr_double(file, "pressure_injection", input.pressure_injection);
    h5_write_attr_double(file, "density_injection", input.density_injection);
    h5_write_attr_double(file, "e_fuel", input.e_fuel);
    h5_write_attr_double(file, "friction_coefficient", input.friction_coefficient);
    h5_write_attr_string(file, "friction_energy_model", input.friction_energy_model);
    h5_write_attr_string(file, "reconstruction_variables", input.reconstruction_variables);
    h5_write_attr_double(file, "massflow_buildup_time", input.massflow_buildup_time);
    h5_write_attr_double(file, "mix_buildup_time", input.mix_buildup_time);
}

void write_source_profile_datasets(hid_t file, const InputData & input)
{
    const hsize_t ncell = static_cast<hsize_t>(input.ncell);
    const hsize_t nsp = static_cast<hsize_t>(input.nSpecies);
    h5_write_vector_1d(file, "dA_dx", input.dA_dx.data(), ncell);
    h5_write_vector_1d(file, "dlnA_dx", input.dlnA_dx.data(), ncell);
    h5_write_vector_1d(file, "D", input.D_array.data(), ncell);
    h5_write_vector_1d(file, "mix_array", input.mix_array.data(), ncell);
    h5_write_vector_1d(file, "mdot", input.mdot_profile.data(), ncell);
    h5_write_vector_1d(file, "dmdot_dx", input.dmdot_dx.data(), ncell);
    const std::vector<double> dmdot_sp_buf = eigen_to_rowmajor(input.dmdot_species_dx);
    h5_write_matrix_rowmajor(file, "dmdot_species_dx", dmdot_sp_buf, nsp, ncell);

    if (input.mdot_profile_by_time.size() > 0) {
        h5_write_vector_1d(file, "mdot_profile_by_time", input.mdot_profile_by_time.data(), ncell);
    }
    if (input.dmdot_dx_by_time.size() > 0) {
        h5_write_vector_1d(file, "dmdot_dx_by_time", input.dmdot_dx_by_time.data(), ncell);
    }
    if (input.dmdot_species_dx_by_time.size() > 0) {
        const std::vector<double> dmdot_sp_t_buf = eigen_to_rowmajor(input.dmdot_species_dx_by_time);
        h5_write_matrix_rowmajor(file, "dmdot_species_dx_by_time", dmdot_sp_t_buf, nsp, ncell);
    }
    if (input.mix_by_time.size() > 0) {
        h5_write_vector_1d(file, "mix_by_time", input.mix_by_time.data(), ncell);
    }
}

std::vector<double> eigen_to_rowmajor(const Eigen::MatrixXd & mat)
{
    const Eigen::Index rows = mat.rows();
    const Eigen::Index cols = mat.cols();
    std::vector<double> buf(static_cast<size_t>(rows * cols));
    for (Eigen::Index i = 0; i < rows; ++i) {
        for (Eigen::Index j = 0; j < cols; ++j) {
            buf[static_cast<size_t>(i * cols + j)] = mat(i, j);
        }
    }
    return buf;
}

bool h5_attr_exists(hid_t loc, const char * name)
{
    return H5Aexists(loc, name) > 0;
}

int h5_read_attr_int(hid_t loc, const char * name)
{
    const hid_t attr = H5Aopen(loc, name, H5P_DEFAULT);
    h5_check(attr, "H5Aopen failed for int attribute: " + std::string(name));
    int value = 0;
    H5Aread(attr, H5T_NATIVE_INT, &value);
    H5Aclose(attr);
    return value;
}

double h5_read_attr_double(hid_t loc, const char * name)
{
    const hid_t attr = H5Aopen(loc, name, H5P_DEFAULT);
    h5_check(attr, "H5Aopen failed for double attribute: " + std::string(name));
    double value = 0.0;
    H5Aread(attr, H5T_NATIVE_DOUBLE, &value);
    H5Aclose(attr);
    return value;
}

std::string h5_read_attr_string(hid_t loc, const char * name)
{
    const hid_t attr = H5Aopen(loc, name, H5P_DEFAULT);
    h5_check(attr, "H5Aopen failed for string attribute: " + std::string(name));
    const hid_t type = H5Aget_type(attr);
    h5_check(type, "H5Aget_type failed for string attribute: " + std::string(name));
    const size_t size = H5Tget_size(type);
    std::string value(size, '\0');
    H5Aread(attr, type, value.data());
    H5Tclose(type);
    H5Aclose(attr);
    while (!value.empty() && value.back() == '\0') {
        value.pop_back();
    }
    return value;
}

std::vector<double> h5_read_vector_1d(hid_t file, const char * name)
{
    const hid_t dset = H5Dopen(file, name, H5P_DEFAULT);
    h5_check(dset, "H5Dopen failed for 1D dataset: " + std::string(name));
    const hid_t space = H5Dget_space(dset);
    h5_check(space, "H5Dget_space failed for 1D dataset: " + std::string(name));
    const int ndims = H5Sget_simple_extent_ndims(space);
    if (ndims != 1) {
        H5Sclose(space);
        H5Dclose(dset);
        throw std::runtime_error("Expected 1D dataset: " + std::string(name));
    }
    hsize_t n = 0;
    H5Sget_simple_extent_dims(space, &n, nullptr);
    std::vector<double> data(n);
    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
    H5Sclose(space);
    H5Dclose(dset);
    return data;
}

void h5_read_matrix_rowmajor(hid_t file, const char * name,
                             std::vector<double> & data, hsize_t & n_rows, hsize_t & n_cols)
{
    const hid_t dset = H5Dopen(file, name, H5P_DEFAULT);
    h5_check(dset, "H5Dopen failed for 2D dataset: " + std::string(name));
    const hid_t space = H5Dget_space(dset);
    h5_check(space, "H5Dget_space failed for 2D dataset: " + std::string(name));
    const int ndims = H5Sget_simple_extent_ndims(space);
    if (ndims != 2) {
        H5Sclose(space);
        H5Dclose(dset);
        throw std::runtime_error("Expected 2D dataset: " + std::string(name));
    }
    hsize_t dims[2] = {0, 0};
    H5Sget_simple_extent_dims(space, dims, nullptr);
    n_rows = dims[0];
    n_cols = dims[1];
    data.resize(static_cast<size_t>(n_rows * n_cols));
    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
    H5Sclose(space);
    H5Dclose(dset);
}

Eigen::MatrixXd rowmajor_to_eigen(const std::vector<double> & buf, hsize_t n_rows, hsize_t n_cols)
{
    Eigen::MatrixXd mat(static_cast<Eigen::Index>(n_rows), static_cast<Eigen::Index>(n_cols));
    for (hsize_t i = 0; i < n_rows; ++i) {
        for (hsize_t j = 0; j < n_cols; ++j) {
            mat(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) =
                buf[static_cast<size_t>(i * n_cols + j)];
        }
    }
    return mat;
}

Eigen::VectorXd vector_to_eigen(const std::vector<double> & data)
{
    Eigen::VectorXd vec(static_cast<Eigen::Index>(data.size()));
    for (size_t i = 0; i < data.size(); ++i) {
        vec(static_cast<Eigen::Index>(i)) = data[i];
    }
    return vec;
}

std::vector<std::string> parse_species_names(const std::string & raw)
{
    std::vector<std::string> names;
    std::string token;
    for (char ch : raw) {
        if (ch == ',') {
            if (!token.empty()) {
                names.push_back(token);
                token.clear();
            }
        } else if (ch != ' ') {
            token.push_back(ch);
        }
    }
    if (!token.empty()) {
        names.push_back(token);
    }
    return names;
}

bool dataset_exists(hid_t file, const char * name)
{
    return H5Lexists(file, name, H5P_DEFAULT) > 0;
}

Eigen::MatrixXd read_optional_row_field(hid_t file, const char * name)
{
    if (!dataset_exists(file, name)) {
        return Eigen::MatrixXd();
    }
    const std::vector<double> data = h5_read_vector_1d(file, name);
    return vector_to_eigen(data).transpose();
}

} // namespace

void writeStateHDF5(const std::string & filename,
                    const StateSnapshotMeta & meta,
                    const InputData & input,
                    const Eigen::VectorXd & xcell,
                    const Eigen::MatrixXd & A_array,
                    const Eigen::MatrixXd & U,
                    Cantera::ThermoPhase & gas)
{
    if (xcell.size() != input.ncell || A_array.cols() != input.ncell
        || U.cols() != input.ncell) {
        throw std::runtime_error("cell-centered HDF5 state dimension mismatch");
    }
    auto [rhoA, u, e, Y] = baseVar_From_conserVar(U);
    auto [pA, T, M] = PTM_From_baseVar(rhoA, u, e, Y, gas);
    const Eigen::MatrixXd rho = rhoA.array() / A_array.array();
    const Eigen::MatrixXd p = pA.array() / A_array.array();

    const hid_t file = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    h5_check(file, "Cannot create HDF5 file: " + filename);

    h5_write_attr_int(file, "format_version", 3);
    h5_write_attr_string(file, "grid_location", "cell_center");
    h5_write_attr_string(file, "kind", meta.kind);
    h5_write_attr_double(file, "time", meta.time);
    h5_write_attr_int(file, "ncell", input.ncell);
    h5_write_attr_int(file, "nface", input.nface);
    h5_write_attr_int(file, "nVar", input.nVar);
    h5_write_attr_int(file, "nSpecies", input.nSpecies);
    h5_write_attr_int(file, "converged", meta.converged ? 1 : 0);
    h5_write_attr_int(file, "steady_iter", meta.steady_iter);
    h5_write_attr_double(file, "steady_error", meta.steady_error);
    if (!meta.steady_residual_definition.empty()) {
        h5_write_attr_string(file, "steady_residual_definition",
                             meta.steady_residual_definition);
        h5_write_attr_double(file, "steady_residual_l2", meta.steady_residual_l2);
        h5_write_attr_double(file, "steady_residual_initial_linf", meta.steady_residual_initial_linf);
        h5_write_attr_double(file, "steady_residual_reduction", meta.steady_residual_reduction);
        h5_write_attr_double(file, "steady_residual_mass_linf", meta.steady_residual_mass_linf);
        h5_write_attr_double(file, "steady_residual_momentum_linf", meta.steady_residual_momentum_linf);
        h5_write_attr_double(file, "steady_residual_energy_linf", meta.steady_residual_energy_linf);
        h5_write_attr_double(file, "steady_residual_species_linf", meta.steady_residual_species_linf);
        h5_write_attr_double(file, "steady_reference_time", meta.steady_reference_time);
        h5_write_attr_int(file, "steady_spatial_order", meta.steady_spatial_order);
    }
    write_source_profile_attrs(file, input);

    std::string species_names;
    for (int k = 0; k < input.nSpecies; ++k) {
        if (k > 0) {
            species_names += ',';
        }
        species_names += gas.speciesName(k);
    }
    h5_write_attr_string(file, "species_names", species_names);

    const hsize_t nstate = static_cast<hsize_t>(xcell.size());
    h5_write_vector_1d(file, "x", xcell.data(), nstate);
    h5_write_vector_1d(file, "A", A_array.data(), nstate);
    h5_write_vector_1d(file, "x_face", input.xface.data(),
                       static_cast<hsize_t>(input.nface));
    h5_write_vector_1d(file, "A_face", input.A_face_array.data(),
                       static_cast<hsize_t>(input.nface));

    const std::vector<double> U_buf = eigen_to_rowmajor(U);
    h5_write_matrix_rowmajor(file, "U", U_buf,
                             static_cast<hsize_t>(U.rows()),
                             static_cast<hsize_t>(U.cols()));

    h5_write_vector_1d(file, "rho", rho.data(), nstate);
    h5_write_vector_1d(file, "p", p.data(), nstate);
    h5_write_vector_1d(file, "u", u.data(), nstate);
    h5_write_vector_1d(file, "T", T.data(), nstate);
    h5_write_vector_1d(file, "Mach", M.data(), nstate);
    h5_write_vector_1d(file, "e", e.data(), nstate);

    const std::vector<double> Y_buf = eigen_to_rowmajor(Y);
    h5_write_matrix_rowmajor(file, "Y", Y_buf,
                             static_cast<hsize_t>(Y.rows()),
                             static_cast<hsize_t>(Y.cols()));

    write_source_profile_datasets(file, input);

    H5Fclose(file);
}

void appendSteadyResidualHDF5(const std::string & filename,
                              const Eigen::MatrixXd & scaled_residual,
                              const Eigen::VectorXd & xcell,
                              int peak_variable,
                              int peak_cell,
                              const std::string & peak_variable_name)
{
    if (scaled_residual.cols() != xcell.size()
        || scaled_residual.rows() <= 0 || scaled_residual.cols() <= 0
        || !scaled_residual.allFinite() || !xcell.allFinite()) {
        throw std::runtime_error("invalid steady residual field dimensions or values");
    }
    if (peak_variable < 0 || peak_variable >= scaled_residual.rows()
        || peak_cell < 0 || peak_cell >= scaled_residual.cols()) {
        throw std::runtime_error("steady residual peak index is out of range");
    }

    Eigen::VectorXd cell_linf(scaled_residual.cols());
    for (Eigen::Index cell = 0; cell < scaled_residual.cols(); ++cell) {
        cell_linf(cell) = scaled_residual.col(cell).cwiseAbs().maxCoeff();
    }

    const hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    h5_check(file, "Cannot reopen HDF5 file for steady residuals: " + filename);
    if (dataset_exists(file, "steady_residual_scaled")
        || dataset_exists(file, "steady_residual_cell_linf")) {
        H5Fclose(file);
        throw std::runtime_error("steady residual datasets already exist: " + filename);
    }

    const std::vector<double> residual_buffer = eigen_to_rowmajor(scaled_residual);
    h5_write_matrix_rowmajor(
        file, "steady_residual_scaled", residual_buffer,
        static_cast<hsize_t>(scaled_residual.rows()),
        static_cast<hsize_t>(scaled_residual.cols()));
    h5_write_vector_1d(file, "steady_residual_cell_linf", cell_linf.data(),
                       static_cast<hsize_t>(cell_linf.size()));
    h5_write_attr_int(file, "steady_residual_peak_variable", peak_variable);
    h5_write_attr_int(file, "steady_residual_peak_cell", peak_cell);
    h5_write_attr_double(file, "steady_residual_peak_x", xcell(peak_cell));
    h5_write_attr_double(file, "steady_residual_peak_signed",
                         scaled_residual(peak_variable, peak_cell));
    h5_write_attr_double(file, "steady_residual_peak_abs",
                         std::abs(scaled_residual(peak_variable, peak_cell)));
    h5_write_attr_string(file, "steady_residual_peak_name", peak_variable_name);
    H5Fclose(file);
}

StateSnapshot readStateHDF5(const std::string & filename)
{
    const hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    h5_check(file, "Cannot open HDF5 file: " + filename);

    StateSnapshot snap;

    if (h5_attr_exists(file, "format_version")) {
        snap.format_version = h5_read_attr_int(file, "format_version");
    }
    if (snap.format_version < 1 || snap.format_version > 3) {
        H5Fclose(file);
        throw std::runtime_error("Unsupported HDF5 format_version: "
                                 + std::to_string(snap.format_version));
    }
    if (h5_attr_exists(file, "grid_location")) {
        snap.grid_location = h5_read_attr_string(file, "grid_location");
    } else {
        snap.grid_location = "face";
    }

    if (h5_attr_exists(file, "kind")) {
        snap.meta.kind = h5_read_attr_string(file, "kind");
    }
    if (h5_attr_exists(file, "time")) {
        snap.meta.time = h5_read_attr_double(file, "time");
    }
    if (h5_attr_exists(file, "ncell")) {
        snap.ncell = h5_read_attr_int(file, "ncell");
    }
    if (h5_attr_exists(file, "nface")) {
        snap.nface = h5_read_attr_int(file, "nface");
    }
    if (h5_attr_exists(file, "nVar")) {
        snap.nVar = h5_read_attr_int(file, "nVar");
    }
    if (h5_attr_exists(file, "nSpecies")) {
        snap.nSpecies = h5_read_attr_int(file, "nSpecies");
    }
    if (h5_attr_exists(file, "converged")) {
        snap.meta.converged = h5_read_attr_int(file, "converged") != 0;
    }
    if (h5_attr_exists(file, "steady_iter")) {
        snap.meta.steady_iter = h5_read_attr_int(file, "steady_iter");
    }
    if (h5_attr_exists(file, "steady_error")) {
        snap.meta.steady_error = h5_read_attr_double(file, "steady_error");
    }
    if (h5_attr_exists(file, "steady_residual_definition")) {
        snap.meta.steady_residual_definition =
            h5_read_attr_string(file, "steady_residual_definition");
    }
    if (h5_attr_exists(file, "steady_spatial_order")) {
        snap.meta.steady_spatial_order = h5_read_attr_int(file, "steady_spatial_order");
    }
    if (h5_attr_exists(file, "steady_residual_l2")) {
        snap.meta.steady_residual_l2 = h5_read_attr_double(file, "steady_residual_l2");
    }
    if (h5_attr_exists(file, "steady_residual_initial_linf")) {
        snap.meta.steady_residual_initial_linf =
            h5_read_attr_double(file, "steady_residual_initial_linf");
    }
    if (h5_attr_exists(file, "steady_residual_reduction")) {
        snap.meta.steady_residual_reduction =
            h5_read_attr_double(file, "steady_residual_reduction");
    }
    if (h5_attr_exists(file, "steady_residual_mass_linf")) {
        snap.meta.steady_residual_mass_linf = h5_read_attr_double(file, "steady_residual_mass_linf");
    }
    if (h5_attr_exists(file, "steady_residual_momentum_linf")) {
        snap.meta.steady_residual_momentum_linf = h5_read_attr_double(file, "steady_residual_momentum_linf");
    }
    if (h5_attr_exists(file, "steady_residual_energy_linf")) {
        snap.meta.steady_residual_energy_linf = h5_read_attr_double(file, "steady_residual_energy_linf");
    }
    if (h5_attr_exists(file, "steady_residual_species_linf")) {
        snap.meta.steady_residual_species_linf = h5_read_attr_double(file, "steady_residual_species_linf");
    }
    if (h5_attr_exists(file, "steady_reference_time")) {
        snap.meta.steady_reference_time = h5_read_attr_double(file, "steady_reference_time");
    }
    if (h5_attr_exists(file, "species_names")) {
        snap.species_names = parse_species_names(h5_read_attr_string(file, "species_names"));
    }
    if (h5_attr_exists(file, "mechanism")) {
        snap.mechanism = h5_read_attr_string(file, "mechanism");
    }

    const std::vector<double> x_data = h5_read_vector_1d(file, "x");
    snap.xface = vector_to_eigen(x_data);
    const int coordinate_count = static_cast<int>(x_data.size());
    if (snap.grid_location == "cell_center") {
        if (snap.ncell > 0 && coordinate_count != snap.ncell) {
            H5Fclose(file);
            throw std::runtime_error("HDF5 cell-centered x length mismatch");
        }
        if (snap.ncell == 0) snap.ncell = coordinate_count;
        if (snap.nface == 0) snap.nface = snap.ncell + 1;
    } else {
        if (snap.nface > 0 && coordinate_count != snap.nface) {
            H5Fclose(file);
            throw std::runtime_error("legacy HDF5 x length mismatch with nface");
        }
        if (snap.nface == 0) snap.nface = coordinate_count;
        if (snap.ncell == 0) snap.ncell = std::max(1, snap.nface - 1);
    }

    const std::vector<double> a_data = h5_read_vector_1d(file, "A");
    if (a_data.size() != x_data.size()) {
        H5Fclose(file);
        throw std::runtime_error("HDF5 A length mismatch with x");
    }
    snap.A_array = vector_to_eigen(a_data).transpose();

    std::vector<double> u_buf;
    hsize_t u_rows = 0;
    hsize_t u_cols = 0;
    h5_read_matrix_rowmajor(file, "U", u_buf, u_rows, u_cols);
    snap.U = rowmajor_to_eigen(u_buf, u_rows, u_cols);
    if (snap.nVar > 0 && static_cast<hsize_t>(snap.nVar) != u_rows) {
        H5Fclose(file);
        throw std::runtime_error("HDF5 U row count mismatch with nVar attribute");
    }
    if (snap.nVar == 0) {
        snap.nVar = static_cast<int>(u_rows);
    }
    if (static_cast<int>(u_cols) != coordinate_count) {
        H5Fclose(file);
        throw std::runtime_error("HDF5 U column count mismatch with x");
    }

    snap.rho = read_optional_row_field(file, "rho");
    snap.p = read_optional_row_field(file, "p");
    snap.u = read_optional_row_field(file, "u");
    snap.T = read_optional_row_field(file, "T");
    snap.Mach = read_optional_row_field(file, "Mach");
    snap.e = read_optional_row_field(file, "e");

    if (dataset_exists(file, "Y")) {
        std::vector<double> y_buf;
        hsize_t y_rows = 0;
        hsize_t y_cols = 0;
        h5_read_matrix_rowmajor(file, "Y", y_buf, y_rows, y_cols);
        snap.Y = rowmajor_to_eigen(y_buf, y_rows, y_cols);
        if (snap.species_names.empty() && snap.nSpecies > 0) {
            for (int k = 0; k < snap.nSpecies; ++k) {
                snap.species_names.push_back("species_" + std::to_string(k));
            }
        }
    }

    snap.dA_dx = read_optional_row_field(file, "dA_dx");
    snap.dlnA_dx = read_optional_row_field(file, "dlnA_dx");
    snap.D_array = read_optional_row_field(file, "D");
    snap.mix_array = read_optional_row_field(file, "mix_array");
    snap.mdot_profile = read_optional_row_field(file, "mdot");
    snap.dmdot_dx = read_optional_row_field(file, "dmdot_dx");
    if (dataset_exists(file, "dmdot_species_dx")) {
        std::vector<double> y_buf;
        hsize_t y_rows = 0;
        hsize_t y_cols = 0;
        h5_read_matrix_rowmajor(file, "dmdot_species_dx", y_buf, y_rows, y_cols);
        snap.dmdot_species_dx = rowmajor_to_eigen(y_buf, y_rows, y_cols);
    }

    snap.mdot_profile_by_time = read_optional_row_field(file, "mdot_profile_by_time");
    snap.dmdot_dx_by_time = read_optional_row_field(file, "dmdot_dx_by_time");
    snap.mix_by_time = read_optional_row_field(file, "mix_by_time");
    if (dataset_exists(file, "dmdot_species_dx_by_time")) {
        std::vector<double> y_buf;
        hsize_t y_rows = 0;
        hsize_t y_cols = 0;
        h5_read_matrix_rowmajor(file, "dmdot_species_dx_by_time", y_buf, y_rows, y_cols);
        snap.dmdot_species_dx_by_time = rowmajor_to_eigen(y_buf, y_rows, y_cols);
    }

    H5Fclose(file);
    return snap;
}

Eigen::MatrixXd interpolateConservedState(const Eigen::VectorXd & x_src,
                                          const Eigen::MatrixXd & U_src,
                                          const Eigen::VectorXd & x_dst)
{
    const int n_src = static_cast<int>(x_src.size());
    const int n_dst = static_cast<int>(x_dst.size());
    if (n_src < 2) {
        throw std::runtime_error("interpolateConservedState: x_src must have at least 2 points");
    }
    if (n_dst < 1) {
        throw std::runtime_error("interpolateConservedState: x_dst must not be empty");
    }
    if (U_src.cols() != n_src) {
        throw std::runtime_error(
            "interpolateConservedState: U_src cols (" + std::to_string(U_src.cols())
            + ") != x_src size (" + std::to_string(n_src) + ")");
    }
    for (int i = 0; i < n_src - 1; ++i) {
        if (x_src(i + 1) <= x_src(i)) {
            throw std::runtime_error("interpolateConservedState: x_src must be strictly increasing");
        }
    }

    Eigen::MatrixXd U_dst(U_src.rows(), n_dst);
    int seg = 0;
    for (int j = 0; j < n_dst; ++j) {
        const double x = x_dst(j);
        if (x <= x_src(0)) {
            U_dst.col(j) = U_src.col(0);
            continue;
        }
        if (x >= x_src(n_src - 1)) {
            U_dst.col(j) = U_src.col(n_src - 1);
            continue;
        }
        while (seg + 1 < n_src - 1 && x_src(seg + 1) < x) {
            ++seg;
        }
        while (seg > 0 && x < x_src(seg)) {
            --seg;
        }
        const double dx = x_src(seg + 1) - x_src(seg);
        const double t = (x - x_src(seg)) / dx;
        U_dst.col(j) = (1.0 - t) * U_src.col(seg) + t * U_src.col(seg + 1);
    }
    return U_dst;
}

CasePaths resolveCaseFiles(const std::string & case_dir,
                           const char * input_json_name,
                           bool require_steady_h5)
{
    namespace fs = std::filesystem;
    const fs::path dir(case_dir);
    if (!fs::is_directory(dir)) {
        throw std::runtime_error("Case directory not found: " + case_dir);
    }

    const std::string input_name(input_json_name ? input_json_name : kInputChemJson);
    CasePaths paths;
    paths.case_dir = fs::absolute(dir).lexically_normal().string();
    paths.input_json = (dir / input_name).string();
    if (input_name == kInputNochemJson) {
        paths.computed_input_json = (dir / kComputedNochemJson).string();
    } else if (input_name == kInputChemJson) {
        paths.computed_input_json = (dir / kComputedChemJson).string();
    } else {
        paths.computed_input_json = (dir / ("input_" + input_name)).string();
    }
    paths.steady_h5 = (dir / "steady_state.h5").string();

    if (!fs::is_regular_file(paths.input_json)) {
        throw std::runtime_error(
            "Missing " + input_name + " in case directory: " + paths.case_dir);
    }
    if (require_steady_h5 && !fs::is_regular_file(paths.steady_h5)) {
        throw std::runtime_error(
            "Missing steady_state.h5 in case directory (run HyperFlow_nochem_steady first): "
            + paths.case_dir);
    }
    return paths;
}

std::string resolveInitialStateH5(const std::string & case_dir,
                                  const std::string & initial_state_h5_spec)
{
    namespace fs = std::filesystem;
    fs::path ic_path;
    if (initial_state_h5_spec.empty()) {
        ic_path = fs::path(case_dir) / "steady_state.h5";
    } else {
        ic_path = fs::path(initial_state_h5_spec);
        if (!ic_path.is_absolute()) {
            ic_path = fs::path(case_dir) / ic_path;
        }
    }
    ic_path = fs::absolute(ic_path).lexically_normal();
    if (!fs::is_regular_file(ic_path)) {
        throw std::runtime_error(
            "Initial state HDF5 not found: " + ic_path.string()
            + " (set input_chem.json \"initial_state_h5\" or place steady_state.h5 in case directory)");
    }
    return ic_path.string();
}

