# HyperFlow V1 — Formulation

HyperFlow is an unsteady one-dimensional compressible reacting-flow solver for variable-area ducts.

## Conservation equations

$$
\frac{\partial \mathbf{U}}{\partial t}
+\frac{\partial \mathbf{F}}{\partial x}
=\mathbf{S},
\qquad
\mathbf{U}=A[\rho,\rho u,\rho e_t,\rho Y_k]^\mathsf{T}.
$$

$$
\mathbf{F}=A[\rho u,\rho u^2+p,u(\rho e_t+p),\rho uY_k]^\mathsf{T},
\qquad
\mathbf{S}=[0,p\,\mathrm{d}A/\mathrm{d}x,0,A\dot\omega_k]^\mathsf{T}.
$$

Here, $A(x)$ is the prescribed duct area, $e_t=e+u^2/2$ is specific total energy, $Y_k$ is species mass fraction and $\dot\omega_k$ is species mass production rate. Cantera provides thermodynamic properties and reaction rates.

## Numerical method

- NND reconstruction and HLLC fluxes for transport.
- Explicit Heun integration for the flow step.
- Strang splitting: half flow step, full chemistry step, half flow step. Chemistry uses independent constant-volume Cantera reactors.

The [Sod comparison](../README.md#sod-shock-tube-example) shows the nonreacting flow response. The [V2 overview](../../HyperFlow_V2/README.md) describes the coupled implicit formulation and added source terms.
