#ifndef PRE_COMPUTATION_H
#define PRE_COMPUTATION_H

#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <string>

/** 输入与预计算结果（几何、入口、喷注等） */
struct InputData
{
    std::string mechanism;
    std::string fuelMoleFraction;
    std::string airMoleFraction;
    int steady_cell;    // 稳态预迭代网格区间数
    int unsteady_cell;  // 非稳态计算网格区间数
    int ncell;          // 当前运行使用的物理控制体数；nface = ncell + 1
    int nVar;
    int nSpecies;
    double cfl_allowance;
    double crosswidth;
    double Height_inlet;
    double massflow_inlet;
    double pressure_inlet;
    double temperature_inlet;
    double ambient_altitude_km;
    double pressure_environment;
    double temperature_environment;
    double density_inlet;
    double velocity_inlet;
    double Ma_inlet;
    double internal_energy_inlet;
    Eigen::MatrixXd Y_inlet;
    double area_inlet;
    double duct_constant_length;
    double duct_expansion_length;
    double duct_expansion_ratio;
    double duct_length;
    double duct_constant_area;
    double duct_end_area;
    double duct_exit_slope;
    double duct_exit_norm_slope;
    double outlet_section_length;
    double outlet_section_expansion_ratio;
    double outlet_section_end_area;
    double outlet_section_exit_norm_slope;
    std::string type_inlet;
    bool injection_enabled;
    double location_injection;
    double velocity_injection;
    double angle_injection;   // 喷注角度，单位：度 计算的是从管壁顺时针到喷注方向的角度，一维考虑上下两个管壁，这里假设喷注一定位于上管壁
    double temperature_injection;
    double pressure_injection;
    double density_injection;
    double e_fuel;
    Eigen::MatrixXd Y_injection;
    double friction_coefficient; // 摩擦系数，单位：无量纲 本计算中考虑全程固定一个摩擦系数
    std::string friction_energy_model;
    std::string reconstruction_variables;
    double equivalence_ratio;
    double FA_ratio;
    double massflow_injection;
    double t_final;
    double dt;
    double error_tolerance;
    int steady_iter_max;
    Eigen::MatrixXd A_array;       // 单元中心截面积，长度 ncell
    Eigen::MatrixXd A_face_array;  // 控制面截面积，长度 nface
    Eigen::MatrixXd D_array;
    Eigen::MatrixXd dA_dx;
    Eigen::MatrixXd dlnA_dx;
    Eigen::VectorXd xface;  // 控制面坐标，长度 nface
    Eigen::VectorXd xcell;  // 控制体中心坐标，长度 ncell
    int nface;
    Eigen::MatrixXd mdot_profile;
    Eigen::MatrixXd dmdot_dx;
    Eigen::MatrixXd dmdot_species_dx;
    Eigen::MatrixXd mdot_profile_by_time;
    Eigen::MatrixXd dmdot_dx_by_time;
    Eigen::MatrixXd dmdot_species_dx_by_time;
    double massflow_buildup_time;
    double alpha_mix;
    double length_mix;
    double mix_buildup_time;
    Eigen::MatrixXd mix_array;
    Eigen::MatrixXd mix_by_time;
    bool fullmix_flag;
};

/** 读 JSON、建网格、Cantera 入口与喷注参数；use_steady_grid 为 true 时用 steady_cell 建网格，否则用 unsteady_cell */
InputData preComputation(nlohmann::json & j, bool use_steady_grid);

#endif
