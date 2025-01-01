#Ahmed Baha Eddine Alimi ISE-05
#a.alimi@innopolis.university
import math
import numpy as np
import matplotlib.pyplot as plt


# a) y' = x + cos(y), with y(1) = 30
def f_a(x, y):
    return x + math.cos(y)


# b) y' = x^2 + y^2, with y(2) = 1
def f_b(x, y):
    if np.abs(y) > 1e4:
        return 1e4
    return x ** 2 + y ** 2


# Runge-Kutta 2nd order (RK2)
def runge_kutta_2(h, N, f):
    y = np.zeros(N + 1)
    x = np.linspace(1, 2, N + 1)
    y[0] = 30 if f == f_a else 1

    for i in range(N):
        k1 = h * f(x[i], y[i])
        k2 = h * f(x[i] + h, y[i] + k1)

        y[i + 1] = y[i] + 0.5 * (k1 + k2)

    return x, y


# Runge-Kutta 4th order (RK4)
def runge_kutta_4(h, N, f):
    y = np.zeros(N + 1)
    x = np.linspace(1, 2, N + 1)
    y[0] = 30 if f == f_a else 1

    for i in range(N):
        k1 = h * f(x[i], y[i])
        k2 = h * f(x[i] + h / 2, y[i] + k1 / 2)
        k3 = h * f(x[i] + h / 2, y[i] + k2 / 2)
        k4 = h * f(x[i] + h, y[i] + k3)

        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return x, y


def compute_relative_errors_fixed(method, f, N_values, h_values):
    all_rel_errors = []
    all_abs_errors = []

    for i in range(len(N_values) - 1):
        N1 = N_values[i]
        N2 = N_values[i + 1]
        h1, h2 = h_values[i], h_values[i + 1]

        x1, y1 = method(h1, N1, f)
        x2, y2 = method(h2, N2, f)

        indices = np.round(np.linspace(0, len(y2) - 1, len(y1))).astype(int)

        y1 = np.nan_to_num(y1, nan=1e5, posinf=1e5, neginf=-1e5)
        y2 = np.nan_to_num(y2, nan=1e5, posinf=1e5, neginf=-1e5)

        rel_errors = np.abs(y2[indices] - y1)
        abs_errors = np.abs(y2[indices] - y1)

        all_rel_errors.append(rel_errors[1:])
        all_abs_errors.append(abs_errors[1:])

    return all_rel_errors, all_abs_errors


def print_relative_errors_fixed(errors, N_values, threshold=1e-15, file=None):
    for i, N in enumerate(N_values[:-1]):
        output = f"N_{N_values[i + 1]}\n"

        non_zero_errors = [e for e in errors[i] if e > threshold]
        output += " ".join(f"{e:.15f}" for e in non_zero_errors[:10]) + "\n"

        if file:
            file.write(output)
        print(output, end='')


def plot_absolute_errors_fixed(rk2_abs, rk4_abs, N_values, equation_label):
    rk2_avg_abs = [np.mean(errors) for errors in rk2_abs if len(errors) > 0]
    rk4_avg_abs = [np.mean(errors) for errors in rk4_abs if len(errors) > 0]

    N_vals_for_plot = N_values[1:len(rk2_avg_abs)+1]

    rk2_log_abs = [np.log2(err) if err > 1e-15 else -np.inf for err in rk2_avg_abs]
    rk4_log_abs = [np.log2(err) if err > 1e-15 else -np.inf for err in rk4_avg_abs]

    plt.figure(figsize=(8, 6))
    plt.plot(N_vals_for_plot, rk2_log_abs, 'o-', label="RK2 Absolute Errors AHAL69I05")
    plt.plot(N_vals_for_plot, rk4_log_abs, 's-', label="RK4 Absolute Errors AHAL69I05")
    plt.xlabel("N")
    plt.ylabel("log2(Absolute Error)")
    plt.title(f"Absolute Error Analysis for RK2 and RK4 ({equation_label})")
    plt.grid(True, which="both", linestyle="--")
    plt.legend()
    plt.show()


# Main code
N_values = [10, 20, 100, 200, 1000]
h_values = [0.1, 0.05, 0.01, 0.005, 0.001]

with open("task1.txt", "w") as file:
    file.write("RK2 (Equation a)\n")
    rk2_rel_errors_a, rk2_abs_errors_a = compute_relative_errors_fixed(runge_kutta_2, f_a, N_values, h_values)
    print("RK2 (Equation a)")
    print_relative_errors_fixed(rk2_rel_errors_a, N_values, file=file)

    file.write("\nRK4 (Equation a)\n")
    rk4_rel_errors_a, rk4_abs_errors_a = compute_relative_errors_fixed(runge_kutta_4, f_a, N_values, h_values)
    print("RK4 (Equation a)")
    print_relative_errors_fixed(rk4_rel_errors_a, N_values, file=file)

    file.write("\nRK2 (Equation b)\n")
    rk2_rel_errors_b, rk2_abs_errors_b = compute_relative_errors_fixed(runge_kutta_2, f_b, N_values, h_values)
    print("RK2 (Equation b)")
    print_relative_errors_fixed(rk2_rel_errors_b, N_values, file=file)

    file.write("\nRK4 (Equation b)\n")
    rk4_rel_errors_b, rk4_abs_errors_b = compute_relative_errors_fixed(runge_kutta_4, f_b, N_values, h_values)
    print("RK4 (Equation b)")
    print_relative_errors_fixed(rk4_rel_errors_b, N_values, file=file)

    plot_absolute_errors_fixed(rk2_abs_errors_a, rk4_abs_errors_a, N_values, "Equation 1a")
    plot_absolute_errors_fixed(rk2_abs_errors_b, rk4_abs_errors_b, N_values, "Equation 1b")