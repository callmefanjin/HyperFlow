#ifndef TOOL_FUNCTION_H
#define TOOL_FUNCTION_H

#include <Eigen/Dense>

#include <string>
#include <utility>
#include <vector>

namespace Cantera
{
class ThermoPhase;
}

/** 未燃混合气燃料/空气质量流量比（按空气物种质量分数加和计算） */
double get_fuel_air_ratio(const std::string & fuel,
                          const std::string & air,
                          double equivalence_ratio,
                          Cantera::ThermoPhase & gas);

/**
 * 根据飞行高度 (km) 与空气组分，按 ISA 计算环境温度、压力。
 * H_km 有效范围 [0, 47]；gas 用于 meanMolecularWeight（与 airMoleFraction 一致）。
 * @return {T [K], P [Pa]}
 */
std::pair<double, double> ambient_conditions_tp(double H_km,
                                               const std::string & airMoleFraction,
                                               Cantera::ThermoPhase & gas);

struct InputData;

/** 将预计算后的 InputData 标量字段写入 JSON */
void writeInputDataJSON(const std::string & filename, const InputData & input);

/** HDF5 快照元数据 */
struct StateSnapshotMeta
{
    std::string kind = "steady";
    double time = 0.0;
    bool converged = true;
    int steady_iter = 0;
    double steady_error = 0.0;
    std::string steady_residual_definition;
    double steady_residual_l2 = 0.0;
    double steady_residual_initial_linf = 0.0;
    double steady_residual_reduction = 0.0;
    double steady_residual_mass_linf = 0.0;
    double steady_residual_momentum_linf = 0.0;
    double steady_residual_energy_linf = 0.0;
    double steady_residual_species_linf = 0.0;
    double steady_reference_time = 0.0;
    int steady_spatial_order = 2;
};

/**
 * 将流场状态写入 HDF5（守恒量 U + 原始量 + 源项剖面，供 restart 与 Python 后处理）
 *
 * format_version=3 使用单元中心存储，v1/v2 旧节点快照仍可读取：
 *   x/A/rho/p/u/T/Mach/e[ncell], U[nVar,ncell], Y[nSpecies,ncell]
 *   x_face/A_face[nface]；源项剖面均为 ncell
 *   dmdot_species_dx[nSpecies,ncell]
 *   mdot_profile_by_time, dmdot_dx_by_time, dmdot_species_dx_by_time, mix_by_time（非稳态快照）
 * 属性: mechanism, alpha_mix, length_mix, massflow_buildup_time, mix_buildup_time, ...
 */
void writeStateHDF5(const std::string & filename,
                    const StateSnapshotMeta & meta,
                    const InputData & input,
                    const Eigen::VectorXd & xcell,
                    const Eigen::MatrixXd & A_array,
                    const Eigen::MatrixXd & U,
                    Cantera::ThermoPhase & gas);

/**
 * 向刚写完的稳态 HDF5 追加最终无量纲半离散 RHS 残差场。
 *
 * scaled_residual[nVar,ncell] 为有符号残差；同时写入每单元跨变量 Linf，
 * 以及全场峰值的变量、单元、坐标、符号值和变量名。仅供稳态诊断，
 * 不改变 restart 所需的状态数据集。
 */
void appendSteadyResidualHDF5(const std::string & filename,
                              const Eigen::MatrixXd & scaled_residual,
                              const Eigen::VectorXd & xcell,
                              int peak_variable,
                              int peak_cell,
                              const std::string & peak_variable_name);

/** HDF5 快照完整数据（readStateHDF5 输出，与 writeStateHDF5 格式对应） */
struct StateSnapshot
{
    StateSnapshotMeta meta;
    int format_version = 1;
    std::string grid_location = "face";
    int ncell = 0;
    int nface = 0;
    int nVar = 0;
    int nSpecies = 0;
    std::string mechanism;
    std::vector<std::string> species_names;
    Eigen::VectorXd xface; // x dataset; see grid_location
    Eigen::MatrixXd A_array;
    Eigen::MatrixXd U;
    Eigen::MatrixXd rho;
    Eigen::MatrixXd p;
    Eigen::MatrixXd u;
    Eigen::MatrixXd T;
    Eigen::MatrixXd Mach;
    Eigen::MatrixXd e;
    Eigen::MatrixXd Y;
    /** 源项剖面（format_version>=2 快照；旧文件为空） */
    Eigen::MatrixXd dA_dx;
    Eigen::MatrixXd dlnA_dx;
    Eigen::MatrixXd D_array;
    Eigen::MatrixXd mix_array;
    Eigen::MatrixXd mdot_profile;
    Eigen::MatrixXd dmdot_dx;
    Eigen::MatrixXd dmdot_species_dx;
    Eigen::MatrixXd mdot_profile_by_time;
    Eigen::MatrixXd dmdot_dx_by_time;
    Eigen::MatrixXd dmdot_species_dx_by_time;
    Eigen::MatrixXd mix_by_time;
};

/**
 * 从 HDF5 读取流场快照（与 writeStateHDF5 格式对应）
 *
 * 必选数据集: x, A, U；其余原始量字段若存在则一并读入。
 */
StateSnapshot readStateHDF5(const std::string & filename);

/**
 * 沿 x 将守恒量 U（nVar×n_src）分段线性插值到目标坐标。
 * x_src 须严格单调递增，且与 U_src 列数一致。
 */
Eigen::MatrixXd interpolateConservedState(const Eigen::VectorXd & x_src,
                                          const Eigen::MatrixXd & U_src,
                                          const Eigen::VectorXd & x_dst);

/** HyperFlow_nochem_steady 读取的 case 输入文件名 */
inline constexpr const char kInputNochemJson[] = "input_nochem.json";
/** HyperFlow 化学非稳态读取的 case 输入文件名 */
inline constexpr const char kInputChemJson[] = "input_chem.json";
/** preComputation 写回的无化学稳态派生 JSON 文件名 */
inline constexpr const char kComputedNochemJson[] = "input_nochem_input.json";
/** preComputation 写回的化学非稳态派生 JSON 文件名（预留，与 nochem 对称） */
inline constexpr const char kComputedChemJson[] = "input_chem_input.json";

/** case 目录内路径：input_* .json、steady_state.h5 等 */
struct CasePaths
{
    std::string case_dir;
    std::string input_json;
    std::string computed_input_json;
    std::string steady_h5;
};

/**
 * 解析 case 目录路径。
 * input_json_name 为 case 目录下的输入 JSON 文件名（如 kInputNochemJson / kInputChemJson）。
 */
CasePaths resolveCaseFiles(const std::string & case_dir,
                           const char * input_json_name,
                           bool require_steady_h5 = false);

/**
 * 解析非稳态初场 HDF5 路径。
 * initial_state_h5_spec 为空时使用 case_dir/steady_state.h5；
 * 否则为相对 case_dir 或绝对路径，须为 writeStateHDF5 格式。
 */
std::string resolveInitialStateH5(const std::string & case_dir,
                                  const std::string & initial_state_h5_spec);

#endif
