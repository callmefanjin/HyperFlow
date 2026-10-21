#include "sourceTerm.H"

#include <cmath>
#include <Eigen/Dense>

using namespace Eigen;

void sourceTerm_calculation(const MatrixXd & pA,
                            const MatrixXd & rhoA,
                            const MatrixXd & u,
                            const InputData & input,
                            bool isInjection,
                            MatrixXd & S_out)
{
    S_out.setZero();

    // 动量：几何源 pA * dlnA/dx + 摩擦
    S_out.row(1) = (pA.array() * input.dlnA_dx.array()).matrix();
    S_out.row(1) += (-2.0 * rhoA.array() * u.array().square()
                     * input.friction_coefficient / input.D_array.array())
                        .matrix();

    if (input.friction_energy_model == "legacy_power_sink") {
        // Historical HyperFlow model: remove u times the wall force from
        // the area-integrated total-energy equation.
        S_out.row(2) = (-2.0 * rhoA.array() * u.array().pow(3)
                        * input.friction_coefficient / input.D_array.array())
                           .matrix();
    } else {
        // Adiabatic stationary wall (Fanno flow): wall shear converts kinetic
        // energy to internal energy but does not remove total energy.
        S_out.row(2).setZero();
    }

    if (!isInjection) {
        return;
    }

    // 喷注增量：连续 / 动量 / 能量 / 组分
    S_out.row(0) = input.dmdot_dx_by_time;

    const double u_inj_x =
        input.velocity_injection * std::cos(input.angle_injection * M_PI / 180.0);
    S_out.row(1) += (u_inj_x * input.dmdot_dx_by_time.array()).matrix();

    // Open-control-volume injection transports h=e+p/rho. Build it
    // explicitly from Cantera internal energy to avoid an enthalpy convention.
    const double e_inj = input.e_fuel
        + input.pressure_injection / input.density_injection
        + 0.5 * input.velocity_injection * input.velocity_injection;
    S_out.row(2) += (e_inj * input.dmdot_dx_by_time.array()).matrix();

    S_out.block(3, 0, input.nSpecies, input.ncell) = input.dmdot_species_dx_by_time;
}
