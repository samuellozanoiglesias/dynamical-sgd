# Let's run a test with the exact structure of the script to see the actual output
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

# Set style
plt.rcParams.update({

    'text.usetex': False,

    'font.family': 'serif',

    'font.serif': ['Times New Roman', 'DejaVu Serif'],

    'mathtext.fontset': 'cm',

    'font.size': 11,

    'axes.labelsize': 11,

    'axes.titlesize': 11,

    'xtick.labelsize': 9,

    'ytick.labelsize': 9,

    'legend.fontsize': 8.5,

    'axes.linewidth': 0.8,

})

CLASS_NAMES = ['Class 1', 'Class 2', 'Class 3']
CLASS_BASE = ['#C0392B', '#2471A3', '#B7950B']

def lighten(hex_color, amount=0.72):
    rgb = np.array(mcolors.to_rgb(hex_color))
    return tuple(rgb + (1.0 - rgb) * amount)

def darken(hex_color, amount=0.15):
    rgb = np.array(mcolors.to_rgb(hex_color))
    return tuple(rgb * (1.0 - amount))

CLASS_FILL = [lighten(c, 0.72) for c in CLASS_BASE]
CLASS_EDGE = [darken(c, 0.15) for c in CLASS_BASE]

PANEL_LABEL_KW = dict(fontsize=24, fontweight='bold', va='bottom', ha='left')

def _compute_focus_weight(period_step: int, period_length: int, w_max: float):
    t = float(period_step)
    period = float(period_length)
    slope = 2.0 * (w_max - 1.0) / period
    if t < period / 2.0:
        focus_weight = 1.0 + t * slope
    else:
        focus_weight = 2.0 * w_max - t * slope - 1.0
    phase = t / period
    return focus_weight, phase

def compute_class_proportions(total_steps, period_length, w_max, num_classes=3):
    steps = np.arange(total_steps)
    full_cycle = num_classes * period_length
    weights = np.ones((total_steps, num_classes), dtype=np.float64)
    pos_in_cycle = steps % full_cycle
    active_class = (pos_in_cycle // period_length).astype(int)
    period_step = pos_in_cycle % period_length
    for step_idx in range(total_steps):
        c = int(active_class[step_idx])
        fw, _ = _compute_focus_weight(int(period_step[step_idx]), period_length, w_max)
        weights[step_idx, c] = fw
    proportions = weights / weights.sum(axis=1, keepdims=True)
    return steps, proportions

def style_3d_axes(ax, zoom=1.3, show_grid=True): # we can pass zoom here!
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.set_box_aspect((1, 1, 1), zoom=zoom) # zoom!
    ax.tick_params(labelsize=14, pad=-1)

    if show_grid:
        # Convert hex color and opacity (alpha) into an RGBA tuple
        grid_rgba = mcolors.to_rgba('#cccccc', alpha=0.8)
        
        # Inject custom styles directly into the 3D axis engine
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis._axinfo['grid'].update({
                'color': grid_rgba,
                'linestyle': '--',
                'linewidth': 0.6
            })
        ax.grid(True)
    else:
        ax.grid(False)

# Assemble
fig = plt.figure(figsize=(15, 5.5), dpi=150)
gs = GridSpec(1, 3, figure=fig, wspace=0.22, left=0.06, right=0.96, top=0.76, bottom=0.12)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1], projection='3d')
ax_c = fig.add_subplot(gs[0, 2], projection='3d')

# Panel A
total_steps = 30000
period_length = 5000
w_max = 30.0
steps, proportions = compute_class_proportions(total_steps, period_length, w_max)
for c in range(3):
     ax_a.plot(steps, proportions[:, c], color=CLASS_BASE[c], linewidth=2.5)
ax_a.set_xlabel('Training steps', fontsize=16)
ax_a.set_ylabel('Class focus proportion', fontsize=16)
ax_a.set_xlim(0, total_steps)
ax_a.set_ylim(-0.02, 1.05)
ax_a.grid(True, linestyle='--', alpha=0.8, color='#cccccc', linewidth=0.6)
ax_a.set_xticks([0, 15000, 30000])
ax_a.set_xticklabels(['0', '15,000', '30,000'], fontsize=14)
ax_a.set_yticks([0.0, 0.5, 1.0])
ax_a.set_yticklabels(['0', '0.5', '1.0'], fontsize=14)
ax_a.tick_params(axis='both', which='major', width=1.2, length=6)
ax_a.set_title("Class-Focus Oscillations", fontsize=18, fontweight='semibold', pad=12)

# Panel B
R_b, r_b = 0.8, 0.35
angles = [0, 2 * np.pi / 3, 4 * np.pi / 3]
z_vals = np.linspace(0, 1.0, 80)
theta_vals = np.linspace(0, 2 * np.pi, 80)
Z, THETA = np.meshgrid(z_vals, theta_vals)
for i, angle in enumerate(angles):
    x_c = R_b * np.cos(angle) * (1.0 - Z)
    y_c = R_b * np.sin(angle) * (1.0 - Z)
    r_z = r_b * (1.0 - Z)
    X = x_c + r_z * np.cos(THETA)
    Y = y_c + r_z * np.sin(THETA)
    ax_b.plot_surface(X, Y, Z, color=CLASS_FILL[i], alpha=0.16, shade=True, linewidth=0, antialiased=True)
    ax_b.plot_wireframe(X, Y, Z, rstride=10, cstride=10, color=CLASS_EDGE[i], linewidth=0.35, alpha=0.55)

t = np.linspace(0, 1.0, 500)
rotations = 3.75
theta_spiral = 2 * np.pi * rotations * t - np.pi / 6
r_spiral = (R_b - r_b) * (1.0 - t)
x_spiral, y_spiral, z_spiral = r_spiral * np.cos(theta_spiral), r_spiral * np.sin(theta_spiral), t
ax_b.plot(x_spiral, y_spiral, z_spiral, color='#107C41', linewidth=2.5, label='Dyn. SGD')
ax_b.scatter([x_spiral[0]], [y_spiral[0]], [z_spiral[0]], color='#1F4E79', edgecolors='black', linewidths=0.7, s=55, zorder=10, label=r'Init $\theta_0$')
ax_b.scatter([0], [0], [1.0], color='#FFC000', marker='*', s=150, edgecolor='black', linewidth=0.8, zorder=12, label='Optimum')

# Standard SGD
rng = np.random.default_rng(42)
t_std = np.linspace(0, 0.45, 200)
x_std = x_spiral[0] + 0.12 * np.sin(8 * t_std) + rng.normal(0, 0.015, 200) - 0.25 * t_std
y_std = y_spiral[0] + 0.12 * np.cos(8 * t_std) + rng.normal(0, 0.015, 200) + 3 * t_std
z_std = 0.05 * t_std + rng.normal(0, 0.008, 200)
ax_b.plot(x_std, y_std, z_std, color='#595959', linestyle=':', linewidth=1.5, alpha=0.9, label='Std SGD')
ax_b.scatter([x_std[-1]], [y_std[-1]], [z_std[-1]], color='#C00000', marker='X', s=60, edgecolors='black', linewidths=0.7, zorder=10, label='Subopt.')

ax_b.set_xlabel(r'$\theta_1$', labelpad=-1, fontsize=20)
ax_b.set_ylabel(r'$\theta_2$', labelpad=-1, fontsize=20)
ax_b.set_zlabel(r'$\theta_3$', labelpad=-1, fontsize=20)
ax_b.view_init(elev=20, azim=38)
style_3d_axes(ax_b, zoom=1.35, show_grid=True) # Let's test 1.35 zoom on B!

ax_b.set_xticks([-0.8, 0, 0.8])
ax_b.set_xticklabels(['-1', '0', '1'], fontsize=14)
ax_b.set_yticks([-0.8, 0, 0.8])
ax_b.set_yticklabels(['-1', '0', '1'], fontsize=14)
ax_b.set_zticks([0.0, 0.5, 1.0])
ax_b.set_zticklabels(['-1', '0', '1'], fontsize=14)
ax_b.set_title("Parameter Space", fontsize=18, fontweight='semibold', pad=12)
ax_b.legend(loc='upper right', bbox_to_anchor=(1.2, 1.02), frameon=True, framealpha=0.9, edgecolor='#e0e0e0', facecolor='#ffffff', fontsize=14, handlelength=1.4)

# Panel C
N_points = 220
z_min, z_max = -0.95, 0.95
R_center = 0.52
tilts = [(0.08, -0.04), (-0.06, 0.08), (-0.02, -0.06)]
r_cyl = 0.11
for i, angle in enumerate(angles):
    x_base, y_base = R_center * np.cos(angle), R_center * np.sin(angle)
    z_pts = rng.uniform(z_min, z_max, N_points)
    r_pts = rng.uniform(0, r_cyl, N_points)
    theta_pts = rng.uniform(0, 2 * np.pi, N_points)
    dx, dy = tilts[i]
    x_center_z = x_base + dx * z_pts
    y_center_z = y_base + dy * z_pts
    x_pts = x_center_z + r_pts * np.cos(theta_pts)
    y_pts = y_center_z + r_pts * np.sin(theta_pts)
    ax_c.scatter(x_pts, y_pts, z_pts, color=CLASS_BASE[i], alpha=0.65, s=12, edgecolors='none', zorder=5)

boundary_angles = [np.pi / 3, np.pi, 5 * np.pi / 3]
r_grid, z_grid = np.linspace(0, 0.95, 20), np.linspace(-1.0, 1.0, 20)
R_G, Z_G = np.meshgrid(r_grid, z_grid)
for b_angle in boundary_angles:
    X_G, Y_G = R_G * np.cos(b_angle), R_G * np.sin(b_angle)
    ax_c.plot_surface(X_G, Y_G, Z_G, color='#7F8C8D', alpha=0.10, shade=False, zorder=2)
    ax_c.plot(X_G[:, -1], Y_G[:, -1], Z_G[:, -1], color='#7F8C8D', linewidth=0.5, alpha=0.3)

ax_c.set_xlabel(r'$\phi_1$', labelpad=-1, fontsize=20)
ax_c.set_ylabel(r'$\phi_2$', labelpad=-1, fontsize=20)
ax_c.set_zlabel(r'$\phi_3$', labelpad=-1, fontsize=20)
ax_c.set_xlim([-1.0, 1.0])
ax_c.set_ylim([-1.0, 1.0])
ax_c.set_zlim([-1.0, 1.0])
ax_c.view_init(elev=22, azim=40)
style_3d_axes(ax_c, zoom=1.35, show_grid=True) # Zoom 1.35 on C too!

ax_c.set_xticks([-1.0, 0, 1.0])
ax_c.set_xticklabels(['-1', '0', '1'], fontsize=14)
ax_c.set_yticks([-1.0, 0, 1.0])
ax_c.set_yticklabels(['-1', '0', '1'], fontsize=14)
ax_c.set_zticks([-1.0, 0, 1.0])
ax_c.set_zticklabels(['-1', '0', '1'], fontsize=14)
ax_c.set_title("Feature Space", fontsize=18, fontweight='semibold', pad=12)

# Panel Labels
ax_a.text(-0.16, 1.08, 'A', transform=ax_a.transAxes, **PANEL_LABEL_KW)
ax_b.text2D(-0.12, 1.08, 'B', transform=ax_b.transAxes, **PANEL_LABEL_KW)
ax_c.text2D(-0.12, 1.08, 'C', transform=ax_c.transAxes, **PANEL_LABEL_KW)

# Global Legend
lines = [plt.Line2D([0], [0], color=CLASS_BASE[i], lw=3.0) for i in range(3)]
fig.legend(lines, CLASS_NAMES, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3, 
           frameon=True, facecolor='white', edgecolor='#e0e0e0', framealpha=0.9, 
           columnspacing=4.0, fontsize=16)

fig.savefig('fig1_full_panel.png', bbox_inches='tight')
fig.savefig('fig1_full_panel.pdf', bbox_inches='tight')
print("Successfully generated plot layout")