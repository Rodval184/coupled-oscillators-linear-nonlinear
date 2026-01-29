# -*- coding: utf-8 -*-
"""
Coupled oscillators: linear vs nonlinear dynamics.

Numerical simulation of two coupled oscillators using RK4 integration.
Comparison between linear coupling and nonlinear (cubic) interaction.

Author: Benjamín Rodríguez Valdez
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------- Parameters ----------------
m = 1.0          # mass
k = 1.0          # spring constant
alpha = 0.1      # nonlinear coupling strength
dt = 0.01
t_max = 100.0

# Initial conditions
x1_0, x2_0 = 1.0, 0.0
v1_0, v2_0 = 0.0, 0.0


def derivatives_linear(state):
    x1, x2, v1, v2 = state
    dx1 = v1
    dx2 = v2
    dv1 = -k * (2*x1 - x2) / m
    dv2 = -k * (2*x2 - x1) / m
    return np.array([dx1, dx2, dv1, dv2])


def derivatives_nonlinear(state):
    x1, x2, v1, v2 = state
    dx1 = v1
    dx2 = v2
    dv1 = -k * (2*x1 - x2) / m - alpha * (x1 - x2)**3
    dv2 = -k * (2*x2 - x1) / m + alpha * (x1 - x2)**3
    return np.array([dx1, dx2, dv1, dv2])


def rk4_step(f, y, dt):
    k1 = f(y)
    k2 = f(y + 0.5 * dt * k1)
    k3 = f(y + 0.5 * dt * k2)
    k4 = f(y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def simulate(deriv_func):
    t = np.arange(0, t_max, dt)
    state = np.array([x1_0, x2_0, v1_0, v2_0], dtype=float)
    states = np.zeros((len(t), 4))

    for i in range(len(t)):
        states[i] = state
        state = rk4_step(deriv_func, state, dt)

    return t, states


def main():
    # Linear case
    t, states_lin = simulate(derivatives_linear)
    x1_lin, x2_lin = states_lin[:, 0], states_lin[:, 1]

    # Nonlinear case
    t, states_non = simulate(derivatives_nonlinear)
    x1_non, x2_non = states_non[:, 0], states_non[:, 1]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(t, x1_lin, label="x1 (linear)", linestyle="--")
    plt.plot(t, x2_lin, label="x2 (linear)", linestyle="--")
    plt.plot(t, x1_non, label="x1 (nonlinear)")
    plt.plot(t, x2_non, label="x2 (nonlinear)")
    plt.xlabel("Time")
    plt.ylabel("Displacement")
    plt.title("Coupled Oscillators: Linear vs Nonlinear")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig("figures/coupled_oscillators.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
