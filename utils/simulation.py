"""
Simulation and visualization utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
import os


def simulate(mpc, path, obstacles, dt, max_steps=2000, verbose=True):
    
    x0, y0 = path.gamma(0)
    th0 = path.heading(0)
    state = np.array([x0, y0, th0, 0.0, 0.0])
    
    states = [state.copy()]
    controls = []
    infos = []
    preds = []
    
    u_prev = np.array([0.5, 0.0])
    
    if verbose:
        print("Running simulation...")
        print("-" * 50)
    
    for step in range(max_steps):
        s_cur = state[3]
        
        if s_cur >= path.length - 1.0:
            if verbose:
                print(f"  Step {step}: Reached end")
            break
        
        u, info = mpc.solve(state, u_prev)
        
        controls.append(u.copy())
        infos.append(info)
        preds.append(info.get('X_pred', None))
        u_prev = u.copy()
        
        # dynamics
        x_new = state[0] + dt * u[0] * np.cos(state[2])
        y_new = state[1] + dt * u[0] * np.sin(state[2])
        th_new = state[2] + dt * u[1]
        s_new = min(state[3] + dt * u[0], path.length)
        
        # lateral offset
        p = path.gamma(s_new)
        psi = path.heading(s_new)
        dx, dy = x_new - p[0], y_new - p[1]
        n_new = -np.sin(psi) * dx + np.cos(psi) * dy
        
        state = np.array([x_new, y_new, th_new, s_new, n_new])
        states.append(state.copy())
        
        if verbose and (step + 1) % 100 == 0:
            seg = path.get_segment(s_cur)
            print(f"  Step {step+1:4d}: s={s_cur:5.1f}m  n={state[4]:+5.2f}m  v={u[0]:.2f}m/s  {seg}")
    
    if verbose:
        print("-" * 50)
    
    return np.array(states), np.array(controls), infos, preds


def plot_results(path, states, controls, obstacles, infos, save_path=None):
    
    fig = plt.figure(figsize=(16, 10))
    
    # trajectory plot
    ax1 = plt.subplot(2, 3, (1, 4))
    
    hw = path.width / 2
    lx, ly, rx, ry = [], [], [], []
    for i in range(0, len(path.x), 3):
        h = path.h[i]
        n = np.array([-np.sin(h), np.cos(h)])
        p = np.array([path.x[i], path.y[i]])
        l = p + hw * n
        r = p - hw * n
        lx.append(l[0]); ly.append(l[1])
        rx.append(r[0]); ry.append(r[1])
    
    ax1.plot(lx, ly, 'k-', lw=2)
    ax1.plot(rx, ry, 'k-', lw=2)
    ax1.fill(lx + rx[::-1], ly + ry[::-1], color='gray', alpha=0.15)
    ax1.plot(path.x, path.y, 'y--', lw=1.5, alpha=0.7, label='Centerline')
    
    # trajectory
    ax1.plot(states[:, 0], states[:, 1], 'b-', lw=2, label='Trajectory')
    
    for obs in obstacles:
        c = Circle(obs['center'], obs['radius'], color='red', alpha=0.7)
        ax1.add_patch(c)
    
    ax1.plot(states[0, 0], states[0, 1], 'go', ms=12, label='Start')
    ax1.plot(states[-1, 0], states[-1, 1], 'r*', ms=15, label='Goal')
    
    # vehicle footprints
    idxs = np.linspace(0, len(states)-1, 12).astype(int)
    for i in idxs:
        x, y, th = states[i, :3]
        L, W = 1.0, 0.5
        corners = np.array([[-L/2, -W/2], [L/2, -W/2], [L/2, W/2], [-L/2, W/2]])
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        corners = (R @ corners.T).T + [x, y]
        poly = Polygon(corners, fc='lime', ec='darkgreen', alpha=0.5)
        ax1.add_patch(poly)
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Trajectory')
    ax1.set_aspect('equal')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # frenet states
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(states[:, 3], 'b-', lw=2)
    ax2.set_ylabel('s (m)', color='b')
    ax2b = ax2.twinx()
    ax2b.plot(states[:, 4], 'r-', lw=2)
    ax2b.axhline(0, color='r', ls='--', alpha=0.3)
    ax2b.set_ylabel('n (m)', color='r')
    ax2.set_xlabel('Step')
    ax2.set_title('Frenet Coordinates')
    ax2.grid(True, alpha=0.3)
    
    # velocity
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(states[:-1, 3], controls[:, 0], 'b-', lw=2)
    ax3.set_xlabel('s (m)')
    ax3.set_ylabel('v (m/s)')
    ax3.set_title('Velocity Profile')
    ax3.grid(True, alpha=0.3)
    
    # controls
    ax4 = plt.subplot(2, 3, 5)
    ax4.plot(controls[:, 0], 'b-', lw=1.5, label='v')
    ax4.plot(controls[:, 1] * 180/np.pi, 'r-', lw=1.5, label='w (deg/s)')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Control')
    ax4.set_title('Control Inputs')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # solve times
    ax5 = plt.subplot(2, 3, 6)
    times = np.array([info['solve_time'] * 1000 for info in infos])
    ax5.plot(times, 'g-', lw=1, alpha=0.7)
    ax5.axhline(np.mean(times), color='b', ls='--', label=f'Mean: {np.mean(times):.1f}ms')
    ax5.axhline(np.percentile(times, 95), color='r', ls='--', label=f'P95: {np.percentile(times, 95):.1f}ms')
    ax5.set_xlabel('Step')
    ax5.set_ylabel('Time (ms)')
    ax5.set_title('Solve Time')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    plt.show()
    return fig


def animate(path, states, controls, obstacles, infos, save_path=None, fps=30, speedup=2.0, show_prediction=True):
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # static elements
    hw = path.width / 2
    lx, ly, rx, ry = [], [], [], []
    for i in range(0, len(path.x), 3):
        h = path.h[i]
        n = np.array([-np.sin(h), np.cos(h)])
        p = np.array([path.x[i], path.y[i]])
        l = p + hw * n
        r = p - hw * n
        lx.append(l[0]); ly.append(l[1])
        rx.append(r[0]); ry.append(r[1])
    
    ax.plot(lx, ly, 'k-', lw=2)
    ax.plot(rx, ry, 'k-', lw=2)
    ax.fill(lx + rx[::-1], ly + ry[::-1], color='gray', alpha=0.15)
    ax.plot(path.x, path.y, 'y--', lw=1.5, alpha=0.7)
    
    for obs in obstacles:
        c = Circle(obs['center'], obs['radius'], color='red', alpha=0.7)
        ax.add_patch(c)
    
    ax.plot(states[0, 0], states[0, 1], 'go', ms=10)
    ax.plot(states[-1, 0], states[-1, 1], 'r*', ms=12)
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    
    # dynamic elements
    traj_line, = ax.plot([], [], 'b-', lw=2, alpha=0.6)
    pred_line, = ax.plot([], [], 'c-', lw=2, alpha=0.8, label='MPC prediction')
    pred_dots, = ax.plot([], [], 'co', ms=3, alpha=0.6)
    
    L, W = 1.0, 0.5
    veh_patch = Polygon(np.zeros((4, 2)), fc='lime', ec='darkgreen', alpha=0.8, zorder=10)
    ax.add_patch(veh_patch)
    
    if show_prediction:
        ax.legend(loc='upper right')
    
    info_txt = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=10,
                       va='top', family='monospace',
                       bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    
    skip = max(1, int(speedup))
    frames = list(range(0, len(states), skip))
    
    def init():
        traj_line.set_data([], [])
        pred_line.set_data([], [])
        pred_dots.set_data([], [])
        veh_patch.set_xy(np.zeros((4, 2)))
        info_txt.set_text('')
        return traj_line, veh_patch, pred_line, pred_dots, info_txt
    
    def update(fi):
        idx = frames[fi]
        
        traj_line.set_data(states[:idx+1, 0], states[:idx+1, 1])
        
        x, y, th = states[idx, :3]
        corners = np.array([[-L/2, -W/2], [L/2, -W/2], [L/2, W/2], [-L/2, W/2]])
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        corners = (R @ corners.T).T + [x, y]
        veh_patch.set_xy(corners)
        
        # show predicted trajectory
        if show_prediction and idx < len(infos) and 'X_pred' in infos[idx]:
            Xp = infos[idx]['X_pred']
            pred_line.set_data(Xp[:, 0], Xp[:, 1])
            pred_dots.set_data(Xp[::5, 0], Xp[::5, 1])  # dots every 5 steps
        else:
            pred_line.set_data([], [])
            pred_dots.set_data([], [])
        
        v = controls[min(idx, len(controls)-1), 0]
        s = states[idx, 3]
        n = states[idx, 4]
        seg = path.get_segment(s)
        
        info_txt.set_text(
            f"Step: {idx:4d}\n"
            f"s: {s:5.1f} / {path.length:.1f} m\n"
            f"n: {n:+5.2f} m\n"
            f"v: {v:4.2f} m/s\n"
            f"Segment: {seg}"
        )
        
        return traj_line, veh_patch, pred_line, pred_dots, info_txt
    
    anim = FuncAnimation(fig, update, init_func=init, frames=len(frames),
                         interval=1000/fps, blit=True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        anim.save(save_path, writer='ffmpeg', fps=fps, dpi=150)
    
    plt.show()
    return anim