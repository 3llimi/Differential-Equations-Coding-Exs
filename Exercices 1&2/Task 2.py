#Ahmed Baha Eddine Alimi ISE-05
#a.alimi@innopolis.university
import math
import numpy as np
import matplotlib.pyplot as plt


# Define the ODE system: y'' = y * sin(x)
def f(x, y, yp):
    return y * math.sin(x)


# Runge-Kutta 2nd order (RK2)
def runge_kutta_2(h, N):
    y = np.zeros(N + 1)
    yp = np.zeros(N + 1)
    x = np.linspace(0, N * h, N + 1)
    y[0] = 0.0
    yp[0] = 1.0

    for i in range(N):
        k1 = h * yp[i]
        l1 = h * f(x[i], y[i], yp[i])

        k2 = h * (yp[i] + l1 / 2.0)
        l2 = h * f(x[i] + h / 2.0, y[i] + k1 / 2.0, yp[i] + l1 / 2.0)

        y[i + 1] = y[i] + k2
        yp[i + 1] = yp[i] + l2

    return x, y


# Runge-Kutta 4th order (RK4)
def runge_kutta_4(h, N):
    y = np.zeros(N + 1)
    yp = np.zeros(N + 1)
    x = np.linspace(0, N * h, N + 1)
    y[0] = 0.0
    yp[0] = 1.0

    for i in range(N):
        k1 = h * yp[i]
        l1 = h * f(x[i], y[i], yp[i])

        k2 = h * (yp[i] + l1 / 2.0)
        l2 = h * f(x[i] + h / 2.0, y[i] + k1 / 2.0, yp[i] + l1 / 2.0)

        k3 = h * (yp[i] + l2 / 2.0)
        l3 = h * f(x[i] + h / 2.0, y[i] + k2 / 2.0, yp[i] + l2 / 2.0)

        k4 = h * (yp[i] + l3)
        l4 = h * f(x[i] + h, y[i] + k3, yp[i] + l3)

        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        yp[i + 1] = yp[i] + (l1 + 2 * l2 + 2 * l3 + l4) / 6.0

    return x, y


def compute_relative_errors_fixed(method, N_values, h_values):
    all_rel_errors = []
    all_abs_errors = []

    for i in range(len(N_values) - 1):
        N1 = N_values[i]
        N2 = N_values[i + 1]
        h1, h2 = h_values[i], h_values[i + 1]

        x1, y1 = method(h1, N1)
        x2, y2 = method(h2, N2)

        indices = np.round(np.linspace(0, N2, N1 + 1)).astype(int)

        rel_errors = np.abs(y2[indices] - y1)
        all_rel_errors.append(rel_errors[1:])
        all_abs_errors.append(np.max(rel_errors[1:]))

    return all_rel_errors, all_abs_errors


def print_relative_errors_fixed(errors, N_values, threshold=1e-15, file=None):
    for i, N in enumerate(N_values[:-1]):
        output = f"N_{N_values[i + 1]}\n"

        non_zero_errors = [e for e in errors[i] if e > threshold]

        output += " ".join(f"{e:.15f}" for e in non_zero_errors[:10]) + "\n"

        if file:
            file.write(output)
        print(output, end='')


def plot_absolute_errors_fixed(rk2_abs, rk4_abs, N_values):
    log_rk2 = np.log2(rk2_abs)
    log_rk4 = np.log2(rk4_abs)

    plt.figure(figsize=(8, 6))
    plt.plot(N_values[1:], log_rk2, 'o-', label="RK2 Absolute Errors (AHAL69I05)")
    plt.plot(N_values[1:], log_rk4, 's-', label="RK4 Absolute Errors(AHAL69I05)")
    plt.xlabel("N")
    plt.ylabel("log2(Absolute Error)")
    plt.title("Absolute Error Analysis for RK2 and RK4 (Task2)")
    plt.grid(True, which="both", linestyle="--")
    plt.legend()
    plt.show()


N_values = [10, 20, 100, 200, 1000]
h_values = [0.1, 0.05, 0.01, 0.005, 0.001]

with open("task2.txt", "w") as file:
    file.write("RK2\n")
    rk2_rel_errors, rk2_abs_errors = compute_relative_errors_fixed(runge_kutta_2, N_values, h_values)
    print("RK2")
    print_relative_errors_fixed(rk2_rel_errors, N_values, file=file)
    file.write("\nRK4\n")
    rk4_rel_errors, rk4_abs_errors = compute_relative_errors_fixed(runge_kutta_4, N_values, h_values)
    print("RK4")
    print_relative_errors_fixed(rk4_rel_errors, N_values, file=file)
    plot_absolute_errors_fixed(rk2_abs_errors, rk4_abs_errors, N_values)
