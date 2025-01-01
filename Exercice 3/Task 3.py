import numpy as np
import matplotlib.pyplot as plt

# Define the differential equations
def dxdt(x, y):
    return -x**2 - y

def dydt(x, y):
    return 2*x - y

# 2nd order Runge-Kutta method
def rk2(h, x0, y0, t_end, t_points):
    t = 0.0
    x = x0
    y = y0
    solutions = {}
    while t < t_end:
        k1_x = dxdt(x, y)
        k1_y = dydt(x, y)
        k2_x = dxdt(x + h*k1_x/2, y + h*k1_y/2)
        k2_y = dydt(x + h*k1_x/2, y + h*k1_y/2)
        x += h * k2_x
        y += h * k2_y
        t += h
        if np.isclose(t, t_points, atol=1e-10).any():
            solutions[t] = (x, y)
    return solutions

# 4th order Runge-Kutta method
def rk4(h, x0, y0, t_end, t_points):
    t = 0.0
    x = x0
    y = y0
    solutions = {}
    while t < t_end:
        k1_x = dxdt(x, y)
        k1_y = dydt(x, y)
        k2_x = dxdt(x + h*k1_x/2, y + h*k1_y/2)
        k2_y = dydt(x + h*k1_x/2, y + h*k1_y/2)
        k3_x = dxdt(x + h*k2_x/2, y + h*k2_y/2)
        k3_y = dydt(x + h*k2_x/2, y + h*k2_y/2)
        k4_x = dxdt(x + h*k3_x, y + h*k3_y)
        k4_y = dydt(x + h*k3_x, y + h*k3_y)
        x += h * (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6
        y += h * (k1_y + 2*k2_y + 2*k3_y + k4_y) / 6
        t += h
        if np.isclose(t, t_points, atol=1e-10).any():
            solutions[t] = (x, y)
    return solutions

def compute_errors(sol_h, sol_h2, t_points):
    errors = []
    for t in t_points:
        x_h, y_h = sol_h[t]
        x_h2, y_h2 = sol_h2[t]
        if abs(x_h2) < 1e-15 and abs(y_h2) < 1e-15:
            relative_error = 0.0
        else:
            relative_error = (abs(x_h - x_h2) + abs(y_h - y_h2)) / (abs(x_h2) + abs(y_h2) + 1e-15)
        errors.append(relative_error)
    return errors
x0 = 1.0
y0 = 1.0
t_end = 100.0
t_points = np.arange(1, 11, 1)
error_threshold = 1e-8
M = 1
while True:
    h = 1.0 / M
    sol_h_rk4 = rk4(h, x0, y0, t_end, t_points)
    h_half = h / 2
    sol_h2_rk4 = rk4(h_half, x0, y0, t_end, t_points)
    errors = compute_errors(sol_h_rk4, sol_h2_rk4, t_points)
    max_error = max(errors)
    if max_error < error_threshold:
        break
    M *= 2
h = 1.0 / M
sol_final_rk4 = rk4(h, x0, y0, t_end, t_points)
sol_final_rk2 = rk2(h, x0, y0, t_end, t_points)
N = int(t_end / h)
print(f"N = {N}")
error_final_rk4 = compute_errors(sol_final_rk4, sol_h2_rk4, t_points)
print(' '.join(['{:.10e}'.format(err) for err in error_final_rk4]))
h_plot = h / 4
N_plot = int(t_end / h_plot)
# For RK4
t_plot = np.linspace(0, t_end, N_plot+1)
x_rk4 = [x0]
y_rk4 = [y0]
x, y = x0, y0
for _ in range(N_plot):
    k1_x = dxdt(x, y)
    k1_y = dydt(x, y)
    k2_x = dxdt(x + h_plot*k1_x/2, y + h_plot*k1_y/2)
    k2_y = dydt(x + h_plot*k1_x/2, y + h_plot*k1_y/2)
    k3_x = dxdt(x + h_plot*k2_x/2, y + h_plot*k2_y/2)
    k3_y = dydt(x + h_plot*k2_x/2, y + h_plot*k2_y/2)
    k4_x = dxdt(x + h_plot*k3_x, y + h_plot*k3_y)
    k4_y = dydt(x + h_plot*k3_x, y + h_plot*k3_y)
    x += h_plot * (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6
    y += h_plot * (k1_y + 2*k2_y + 2*k3_y + k4_y) / 6
    x_rk4.append(x)
    y_rk4.append(y)
# For RK2
x_rk2 = [x0]
y_rk2 = [y0]
x, y = x0, y0
for _ in range(N_plot):
    k1_x = dxdt(x, y)
    k1_y = dydt(x, y)
    k2_x = dxdt(x + h_plot*k1_x/2, y + h_plot*k1_y/2)
    k2_y = dydt(x + h_plot*k1_x/2, y + h_plot*k1_y/2)
    x += h_plot * k2_x
    y += h_plot * k2_y
    x_rk2.append(x)
    y_rk2.append(y)
# Plot phase trajectories
plt.figure(figsize=(10, 6))
plt.plot(x_rk4, y_rk4, label='RK4', linewidth=2)
plt.plot(x_rk2, y_rk2, label='RK2', linewidth=1)
plt.xlabel('x(t)')
plt.ylabel('y(t)')
plt.title('Phase Trajectories of RK2 and RK4')
plt.legend()
plt.grid(True)
plt.show()
