#!/usr/bin/env python3
# 黎曼求解器
# ------------------------- #
# 概述：给定sod激波管两侧的初始条件，隔板抽开，给出后续的分布情况

import numpy as np
from scipy.optimize import root_scalar


def judge_func(p_star, p, rho):
    """分段函数"""
    gamma = 1.4  # 比热比取1.4
    c = np.sqrt(gamma * p / rho)

    if p_star >= p:
        value = (p_star - p) / (
            rho * c * np.sqrt(
                ((gamma + 1) / (2 * gamma)) * (p_star / p)
                + ((gamma - 1) * 0.5 / gamma)
            )
        )
    else:
        value = 2 * c * (np.power(p_star / p, (gamma - 1) / 2 / gamma) - 1) / (gamma - 1)
    return value


def factor_A(p_star, p, rho):
    gamma = 1.4
    c = np.sqrt(gamma * p / rho)
    value = rho * c * np.sqrt(
        (gamma + 1) / (2 * gamma) * (p_star / p) + (gamma - 1) / (2 * gamma)
    )
    return value


def inner_expansion_1(x, t, p_1, u_1, rho_1):
    """求解左行膨胀波内部"""
    gamma = 1.4
    c_1 = np.sqrt(gamma * p_1 / rho_1)

    def f(c):
        return u_1 + 2 * c_1 / (gamma - 1) - (x / t + c + 2 * c / (gamma - 1))

    cxt = root_scalar(f, x0=0.5, bracket=[0.01, c_1], method="brentq").root
    uxt = x / t + cxt
    pxt = p_1 * (cxt / c_1) ** (2 * gamma / (gamma - 1))  # 等熵关系
    rhoxt = gamma * pxt / (cxt**2)
    return uxt, pxt, rhoxt


def inner_expansion_2(x, t, p_2, u_2, rho_2):
    """求解右行膨胀波内部"""
    gamma = 1.4
    c_2 = np.sqrt(gamma * p_2 / rho_2)

    def f(c):
        return u_2 - 2 * c_2 / (gamma - 1) - (x / t - c - 2 * c / (gamma - 1))

    c = root_scalar(f, x0=1.0, bracket=[0.01, c_2], method="brentq").root
    uxt = x / t + c
    pxt = p_2 * (c / c_2) ** (2 * gamma / (gamma - 1))
    rhoxt = gamma * pxt / (c**2)
    return uxt, pxt, rhoxt


def riemann_wave_params(u_1, rho_1, p_1, u_2, rho_2, p_2, gamma=1.4):
    """求解 Riemann 问题，返回波速与 star 区状态（与 test.m 逻辑一致）"""

    def equation(p_w):
        return u_1 - u_2 - judge_func(p_w, p_1, rho_1) - judge_func(p_w, p_2, rho_2)

    p_lo = min(p_1, p_2) * 0.01
    p_hi = max(p_1, p_2) * 2.0
    p_star = root_scalar(equation, bracket=[p_lo, p_hi], method="brentq").root
    u_star = u_1 - judge_func(p_star, p_1, rho_1)

    flag_1 = "s" if p_star > p_1 else "e"
    flag_2 = "s" if p_star > p_2 else "e"

    params = {
        "p_star": p_star,
        "u_star": u_star,
        "flag_1": flag_1,
        "flag_2": flag_2,
        "gamma": gamma,
    }

    if flag_1 == "s":
        A_1 = factor_A(p_star, p_1, rho_1)
        params["rho_star_1"] = rho_1 * A_1 / (A_1 - rho_1 * (u_1 - u_star))
        params["Z_1"] = u_1 - A_1 / rho_1
    else:
        c_1 = np.sqrt(gamma * p_1 / rho_1)
        c_star_1 = c_1 + 0.5 * (gamma - 1) * (u_1 - u_star)
        params["rho_star_1"] = gamma * p_star / (c_star_1**2)
        params["Z_1_head"] = u_1 - c_1
        params["Z_1_tail"] = u_star - c_star_1
        if flag_1 == "e" and flag_2 == "e":
            params["Z_1_tail"] = u_1 - 2 * c_1 / (gamma - 1)

    if flag_2 == "s":
        Z_2 = u_2 - (p_star - p_2) / ((u_2 - u_star) * rho_2)
        params["Z_2"] = Z_2
        params["rho_star_2"] = rho_2 * (u_2 - Z_2) / (u_star - Z_2)
    else:
        c_2 = np.sqrt(gamma * p_2 / rho_2)
        c_star_2 = c_2 + 0.5 * (gamma - 1) * (u_2 - u_star)
        params["rho_star_2"] = gamma * p_star / (c_2**2)
        params["Z_2_head"] = u_2 + c_2
        params["Z_2_tail"] = u_star + c_star_2
        if flag_1 == "e" and flag_2 == "e":
            params["Z_2_tail"] = u_2 - 2 * c_2 / (gamma - 1)

    return params


def sample_riemann(x, t, u_1, rho_1, p_1, u_2, rho_2, p_2, gamma=1.4):
    """
    在单个空间点 x 采样 Riemann 精确解。
    x 为相对隔板（初始间断面）的坐标，隔板处 x=0。
    """
    if t <= 0:
        if x < 0:
            return u_1, p_1, rho_1
        return u_2, p_2, rho_2

    w = riemann_wave_params(u_1, rho_1, p_1, u_2, rho_2, p_2, gamma)
    p_star = w["p_star"]
    u_star = w["u_star"]

    if w["flag_1"] == "e" and w["flag_2"] == "s":
        if x <= w["Z_1_head"] * t:
            return u_1, p_1, rho_1
        if x <= w["Z_1_tail"] * t:
            return inner_expansion_1(x, t, p_1, u_1, rho_1)
        if x <= u_star * t:
            return u_star, p_star, w["rho_star_1"]
        if x <= w["Z_2"] * t:
            return u_star, p_star, w["rho_star_2"]
        return u_2, p_2, rho_2

    if w["flag_1"] == "s" and w["flag_2"] == "s":
        if x <= w["Z_1"] * t:
            return u_1, p_1, rho_1
        if x <= u_star * t:
            return u_star, p_star, w["rho_star_1"]
        if x <= w["Z_2"] * t:
            return u_star, p_star, w["rho_star_2"]
        return u_2, p_2, rho_2

    if w["flag_1"] == "e" and w["flag_2"] == "e":
        if x <= w["Z_1_head"] * t:
            return u_1, p_1, rho_1
        if x <= w["Z_1_tail"] * t:
            return inner_expansion_1(x, t, p_1, u_1, rho_1)
        if x <= u_star * t:
            return u_star, p_star, w["rho_star_1"]
        if x <= w["Z_2_tail"] * t:
            return inner_expansion_2(x, t, p_2, u_2, rho_2)
        return u_2, p_2, rho_2

    if w["flag_1"] == "s" and w["flag_2"] == "e":
        if x <= w["Z_1"] * t:
            return u_1, p_1, rho_1
        if x <= u_star * t:
            return u_star, p_star, w["rho_star_1"]
        if x <= w["Z_2_head"] * t:
            return u_star, p_star, w["rho_star_2"]
        if x <= w["Z_2_tail"] * t:
            return inner_expansion_2(x, t, p_2, u_2, rho_2)
        return u_2, p_2, rho_2

    raise NotImplementedError(f"wave pattern {w['flag_1']}/{w['flag_2']} not implemented")


def sample_riemann_array(x_arr, t, u_1, rho_1, p_1, u_2, rho_2, p_2, gamma=1.4):
    """在多个空间点采样 Riemann 精确解"""
    u = np.zeros_like(x_arr, dtype=float)
    p = np.zeros_like(x_arr, dtype=float)
    rho = np.zeros_like(x_arr, dtype=float)
    for i, xi in enumerate(x_arr):
        u[i], p[i], rho[i] = sample_riemann(xi, t, u_1, rho_1, p_1, u_2, rho_2, p_2, gamma)
    return u, p, rho

