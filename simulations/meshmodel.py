# Mesh Model 3D Surface Simulator — Fix: Ensure All Fields Render Correctly
# Ensures psi, Phi, and K fields are always properly initialized and updated for all modes

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.widgets import RadioButtons, Button

# Grid setup
Nx, Ny = 100, 100
Lx, Ly = 2 * np.pi, 2 * np.pi
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)
dx, dy = x[1] - x[0], y[1] - y[0]
dt = 0.05
c0 = 1.0

# State
view_mode = 'Tension'
entity_mode = 'Wave'
running = True
time_step = 0

# Field containers
def reset_fields():
    global psi, v, Phi, K, time_step
    psi = np.zeros((Ny, Nx))
    v = np.zeros_like(psi)
    Phi = np.ones((Ny, Nx)) * 0.5
    K = np.zeros((Ny, Nx))
    time_step = 0

    if entity_mode == 'Wave':
        psi[Ny//2, Nx//2] = 1.0

    elif entity_mode == 'Particle':
        update_particle_fields(time_step)

    elif entity_mode == 'Higgs Decay':
        update_higgs_fields(time_step)

    elif entity_mode == 'Photon Trail':
        psi[Ny//2, Nx//4] = 1.0

    elif entity_mode == 'Entangled Pair':
        psi[Ny//2 - 10, Nx//4] = 1.0
        psi[Ny//2 + 10, Nx//4] = 1.0

# Field logic

def update_particle_fields(t_index):
    global psi, K, Phi
    center_x = (t_index % Nx) * dx
    center_y = Ly / 2
    r = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    psi = np.exp(-r**2 / 0.05) * 2.0
    K = np.exp(-r**2 / 3.0) * 0.1
    Phi = 1.0 + np.exp(-r**2 / 0.05)

higgs_decay_triggered = False

def update_higgs_fields(t_index):
    global psi, K, Phi, higgs_decay_triggered
    if not higgs_decay_triggered and t_index >= Nx // 3:
        higgs_decay_triggered = True

    if not higgs_decay_triggered:
        update_particle_fields(t_index)
    else:
        decay_distance = dx * (t_index - Nx // 3) * 0.5  # how far since decay started
        x0, y0 = (Nx // 3) * dx, Ly / 2  # Higgs center

        # 30 degrees angle components
        dx_30 = decay_distance * np.cos(np.radians(30))
        dy_30 = decay_distance * np.sin(np.radians(30))

        # Two symmetric gamma wavefronts
        gamma1 = np.exp(-((X - (x0 + dx_30))**2 + (Y - (y0 + dy_30))**2) / 0.02)
        gamma2 = np.exp(-((X - (x0 + dx_30))**2 + (Y - (y0 - dy_30))**2) / 0.02)
        psi[:] = gamma1 + gamma2
        Phi[:] = 1.0 + 0.3 * (gamma1 + gamma2)
        K[:] = 0.0

# Laplacian
def laplacian(f):
    return (
        np.roll(f, 1, axis=0) + np.roll(f, -1, axis=0) +
        np.roll(f, 1, axis=1) + np.roll(f, -1, axis=1) - 4 * f
    ) / (dx * dy)

# Plot setup
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_zlim(-2, 2)

# UI: View toggle
ax_mode = plt.axes([0.025, 0.5, 0.12, 0.15])
mode_buttons = RadioButtons(ax_mode, ['Tension', 'Curvature', 'Coherence'])
mode_buttons.on_clicked(lambda label: globals().update(view_mode=label))

# UI: Pause
ax_pause = plt.axes([0.025, 0.3, 0.1, 0.04])
pause_button = Button(ax_pause, 'Pause')

def toggle_pause(event):
    global running
    running = not running
    pause_button.label.set_text('Resume' if not running else 'Pause')

pause_button.on_clicked(toggle_pause)

# UI: Entity toggle
ax_entity = plt.axes([0.025, 0.05, 0.12, 0.22])
entity_buttons = RadioButtons(ax_entity, ['Wave', 'Particle', 'Higgs Decay', 'Photon Trail', 'Entangled Pair'])

def set_entity_mode(label):
    global entity_mode, higgs_decay_triggered
    entity_mode = label
    higgs_decay_triggered = False
    reset_fields()

entity_buttons.on_clicked(set_entity_mode)

# Update loop
def update(frame):
    global psi, v, Phi, time_step
    if running:
        if entity_mode == 'Wave':
            lap = laplacian(psi)
            wave_speed = c0 * Phi
            v += dt * wave_speed**2 * lap
            psi += dt * v
            psi += dt * K
            Phi += 0.01 * np.abs(psi)
            Phi = np.clip(Phi, 0.1, 2.0)

        elif entity_mode == 'Particle':
            update_particle_fields(time_step)
            time_step += 1

        elif entity_mode == 'Higgs Decay':
            update_higgs_fields(time_step)
            time_step += 1

        elif entity_mode == 'Photon Trail':
            center_x = Lx/4 + time_step * dx * 0.5
            center_y = Ly / 2
            r = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
            psi[:] = np.exp(-r**2 / 0.02) * 1.8
            Phi[:] = 1.0 + np.exp(-r**2 / 0.05)
            K[:] = 0.0
            time_step += 1

        elif entity_mode == 'Entangled Pair':
            offset = time_step * dx * 0.5
            gamma1 = np.exp(-((X - offset)**2 + (Y - Ly/2 + offset)**2) / 0.02)
            gamma2 = np.exp(-((X - offset)**2 + (Y - Ly/2 - offset)**2) / 0.02)
            psi[:] = gamma1 + gamma2
            Phi[:] = 1.0 + 0.3 * (gamma1 + gamma2)
            K[:] = 0.0
            time_step += 1

    ax.clear()
    ax.set_zlim(-2, 2)
    ax.set_title(f"Mesh Model: {view_mode} Field ({entity_mode})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if view_mode == 'Tension':
        ax.plot_surface(X, Y, psi, cmap=cm.plasma, linewidth=0, antialiased=True)
    elif view_mode == 'Curvature':
        ax.plot_surface(X, Y, K, cmap=cm.inferno, linewidth=0, antialiased=True)
    elif view_mode == 'Coherence':
        ax.plot_surface(X, Y, Phi, cmap=cm.viridis, linewidth=0, antialiased=True)

    return []

reset_fields()
ani = animation.FuncAnimation(fig, update, frames=300, interval=15, blit=False)
plt.show()