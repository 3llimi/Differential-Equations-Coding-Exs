import numpy as np
import matplotlib.pyplot as plt

# Lorenz system parameters
sigma = 10
rho = 26.5
beta = 8 / 3

# Lorenz equations
def lorenz(t, state):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return np.array([dxdt, dydt, dzdt])

# Runge-Kutta 2nd order (RK2) method
def rk2_step(func, t, state, dt):
    k1 = func(t, state)
    k2 = func(t + dt / 2, state + dt / 2 * k1)
    return state + dt * k2

# Runge-Kutta 4th order (RK4) method
def rk4_step(func, t, state, dt):
    k1 = func(t, state)
    k2 = func(t + dt / 2, state + dt / 2 * k1)
    k3 = func(t + dt / 2, state + dt / 2 * k2)
    k4 = func(t + dt, state + dt * k3)
    return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

def solve_lorenz(method, t_max, dt):
    t_values = np.arange(0, t_max + dt, dt)
    state = np.array([0, 1, 0])
    trajectory = [state]

    for t in t_values[:-1]:
        state = method(lorenz, t, state, dt)
        trajectory.append(state)

    return np.array(t_values), np.array(trajectory)

def calculate_relative_error(traj1, traj2):
    return np.linalg.norm(traj1 - traj2, axis=1) / np.linalg.norm(traj2, axis=1)

t_max = 100
dt = 0.01

t_values_rk2, traj_rk2 = solve_lorenz(rk2_step, t_max, dt)
t_values_rk4, traj_rk4 = solve_lorenz(rk4_step, t_max, dt)

relative_error = calculate_relative_error(traj_rk2, traj_rk4)

t_targets = np.arange(10, t_max + 1, 10)
relative_errors_at_targets = [relative_error[np.argmin(np.abs(t_values_rk4 - t))] for t in t_targets]

print(f"N = {len(t_values_rk4)}")
print(" ".join(f"{e:.2e}" for e in relative_errors_at_targets))

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(traj_rk4[:, 0], traj_rk4[:, 1], traj_rk4[:, 2], label="RK4")
ax.plot(traj_rk2[:, 0], traj_rk2[:, 1], traj_rk2[:, 2], label="RK2", linestyle='dashed')
ax.set_title("Phase Trajectory of Lorenz System")
ax.set_xlabel("x(t)")
ax.set_ylabel("y(t)")
ax.set_zlabel("z(t)")
ax.legend()
plt.show()
