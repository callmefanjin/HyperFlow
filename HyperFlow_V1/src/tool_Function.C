#include "tool_Function.H"
#include "preComputation.H"
#include "coreMethod.H"

#include <hdf5.h>

#include <stdexcept>
#include <string>
#include <vector>
#include <filesystem>
#include <cantera/core.h>

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
                    const Eigen::VectorXd & xface,
                    const Eigen::MatrixXd & A_array,
                    const Eigen::MatrixXd & U,
                    Cantera::ThermoPhase & gas)
{
    auto [rhoA, u, e, Y] = baseVar_From_conserVar(U);
    auto [pA, T, M] = PTM_From_baseVar(rhoA, u, e, Y, gas);
    const Eigen::MatrixXd rho = rhoA.array() / A_array.array();
    const Eigen::MatrixXd p = pA.array() / A_array.array();

    const hid_t file = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    h5_check(file, "Cannot create HDF5 file: " + filename);

    h5_write_attr_int(file, "format_version", 3);
    h5_write_attr_string(file, "kind", meta.kind);
    h5_write_attr_double(file, "time", meta.time);
    h5_write_attr_int(file, "ncell", input.ncell);
    h5_write_attr_int(file, "nface", input.nface);
    h5_write_attr_int(file, "nVar", input.nVar);
    h5_write_attr_int(file, "nSpecies", input.nSpecies);
    h5_write_attr_int(file, "converged", meta.converged ? 1 : 0);
    h5_write_attr_int(file, "chemistry", input.chemistry ? 1 : 0);
    h5_write_attr_string(file, "mechanism", input.mechanism);

    std::string species_names;
    for (int k = 0; k < input.nSpecies; ++k) {
        if (k > 0) {
            species_names += ',';
        }
        species_names += gas.speciesName(k);
    }
    h5_write_attr_string(file, "species_names", species_names);

    const hsize_t nface = static_cast<hsize_t>(xface.size());
    h5_write_vector_1d(file, "x", xface.data(), nface);
    h5_write_vector_1d(file, "A", A_array.data(), nface);

    const std::vector<double> U_buf = eigen_to_rowmajor(U);
    h5_write_matrix_rowmajor(file, "U", U_buf,
                             static_cast<hsize_t>(U.rows()),
                             static_cast<hsize_t>(U.cols()));

    h5_write_vector_1d(file, "rho", rho.data(), nface);
    h5_write_vector_1d(file, "p", p.data(), nface);
    h5_write_vector_1d(file, "u", u.data(), nface);
    h5_write_vector_1d(file, "T", T.data(), nface);
    h5_write_vector_1d(file, "Mach", M.data(), nface);
    h5_write_vector_1d(file, "e", e.data(), nface);

    const std::vector<double> Y_buf = eigen_to_rowmajor(Y);
    h5_write_matrix_rowmajor(file, "Y", Y_buf,
                             static_cast<hsize_t>(Y.rows()),
                             static_cast<hsize_t>(Y.cols()));

    h5_write_vector_1d(file, "dA_dx", input.dA_dx.data(), nface);
    h5_write_vector_1d(file, "dlnA_dx", input.dlnA_dx.data(), nface);

    H5Fclose(file);
}

StateSnapshot readStateHDF5(const std::string & filename)
{
    const hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    h5_check(file, "Cannot open HDF5 file: " + filename);

    StateSnapshot snap;
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
    if (h5_attr_exists(file, "species_names")) {
        snap.species_names = parse_species_names(h5_read_attr_string(file, "species_names"));
    }
    if (h5_attr_exists(file, "mechanism")) {
        snap.mechanism = h5_read_attr_string(file, "mechanism");
    }

    const std::vector<double> x_data = h5_read_vector_1d(file, "x");
    snap.xface = vector_to_eigen(x_data);
    if (snap.nface == 0) {
        snap.nface = static_cast<int>(x_data.size());
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
    if (snap.nVar == 0) {
        snap.nVar = static_cast<int>(u_rows);
    }
    if (snap.ncell == 0) {
        snap.ncell = snap.nface - 1;
    }
    if (snap.nSpecies == 0) {
        snap.nSpecies = snap.nVar - 3;
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
    }
    snap.dA_dx = read_optional_row_field(file, "dA_dx");
    snap.dlnA_dx = read_optional_row_field(file, "dlnA_dx");

    H5Fclose(file);
    return snap;
}

CasePaths resolveCaseFiles(const std::string & case_dir)
{
    namespace fs = std::filesystem;
    const fs::path dir(case_dir);
    if (!fs::is_directory(dir)) {
        throw std::runtime_error("Case directory not found: " + case_dir);
    }
    CasePaths paths;
    paths.case_dir = fs::absolute(dir).lexically_normal().string();
    paths.config_json = (dir / kConfigJson).string();
    paths.out_dir = (dir / "out").string();
    if (!fs::is_regular_file(paths.config_json)) {
        throw std::runtime_error("Missing config.json in case directory: " + paths.case_dir);
    }
    fs::create_directories(paths.out_dir);
    return paths;
}
