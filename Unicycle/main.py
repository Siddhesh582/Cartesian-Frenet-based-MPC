"""
Frenet-Cartesian MPC for Unicycle Path Tracking

This main script runs the simulation of a unicycle model following a reference path
using a Frenet-Cartesian Model Predictive Controller.
"""

import numpy as np
import matplotlib.pyplot as plt
import time

from FrenetParameters import ReferencePathParameters
from Unicycle import Unicycle
from Controller import FrenetCartesianMPC
from simulation import simulate_system, plot_simulation_results

def main():
    # values for reference path with sinusoidal curve
    s_vals = np.linspace(0, 100, 2000)
    x_vals = s_vals
    y_vals = 1 * np.sin(0.05 * s_vals)
    lane_width = 15.0
    
    # generate reference path
    ref_path = ReferencePathParameters(s_vals, x_vals, y_vals, lane_width=lane_width)
    
    # default obstacles
    obstacles = [
        {'center': (15, 2.9), 'radius': 2.0},
        {'center': (30, -1.3), 'radius': 1.8},
        {'center': (45, 2.75), 'radius': 1.8},
        {'center': (55, 2.75), 'radius': 2.2},
        {'center': (85, 0.0), 'radius': 1.6},
        {'center': (72, 0.0), 'radius': 1.6}
    ]
    ref_path.default_obstacles = obstacles
    
    unicycle = Unicycle(ref_path, v_min=0.1, v_max=2.0, w_min=-np.pi/2, w_max=np.pi/2)
    
    mpc_controller = FrenetCartesianMPC(ref_path, unicycle)
    
    mpc_controller.set_reference_velocity(0.8)
    
    ''' Run simulation '''
    simulation_steps = 600
    print(f"Running simulation for {simulation_steps} steps...")
    start_time = time.time()
    states, controls, solve_times, predicted_trajectories = simulate_system(
        mpc_controller, ref_path, unicycle, simulation_steps=simulation_steps
    )
    total_time = time.time() - start_time
    print(f"Simulation completed in {total_time:.2f} seconds")
    
    # simulation results
    plot_simulation_results(ref_path, states, controls, unicycle)
    
    # additional plots
    plot_additional_metrics(states, controls, solve_times)
    
    # final path following with obstacles
    plot_final_path(ref_path, states, obstacles, unicycle)

def plot_additional_metrics(states, controls, solve_times):
    """additional metrics for analyzing the controller performance"""
    plt.figure(figsize=(15, 10))
    
    # linear velocity changes
    ax1 = plt.subplot(2, 2, 1)
    velocities = controls[:, 0]
    ax1.plot(states[:-1, 3], velocities, 'b-', linewidth=2)
    ax1.set_xlabel('Path Progress (s)')
    ax1.set_ylabel('Velocity (m/s)')
    ax1.set_title('Velocity Profile Along Path')
    ax1.grid(True)
    
    # angular velocity changes
    ax2 = plt.subplot(2, 2, 2)
    angular_velocities = controls[:, 1]
    ax2.plot(states[:-1, 3], angular_velocities, 'g-', linewidth=2)
    ax2.set_xlabel('Path Progress (s)')
    ax2.set_ylabel('Angular Velocity (rad/s)')
    ax2.set_title('Angular Velocity Profile Along Path')
    ax2.grid(True)
    
    # solver times
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(np.arange(len(solve_times)), solve_times * 1000, 'm-', linewidth=2)
    ax3.set_xlabel('Simulation Step')
    ax3.set_ylabel('Solver Time (ms)')
    ax3.set_title('MPC Solver Times')
    ax3.grid(True)
    
    # heading error
    ax4 = plt.subplot(2, 2, 4)
    heading_errors = []
    for i in range(len(states)):
        s = states[i, 3]
        theta = states[i, 2]
        path_heading = np.arctan2(0.05 * np.cos(0.1 * s), 1.0)  # approximate heading function
        heading_error = np.abs(theta - path_heading)
        heading_error = np.minimum(heading_error, 2*np.pi - heading_error)  
        heading_errors.append(heading_error)
    
    ax4.plot(states[:, 3], heading_errors, 'r-', linewidth=2)
    ax4.set_xlabel('Path Progress (s)')
    ax4.set_ylabel('Heading Error (rad)')
    ax4.set_title('Heading Error Along Path')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_final_path(ref_path, states, obstacles, unicycle):
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # road
    ref_path.plot_curvy_road(obstacles=obstacles, show=False, ax=ax)
    
    # vehicle trajectory
    ax.plot(states[:, 0], states[:, 1], 'r-', linewidth=2, label='Vehicle Trajectory')
    
    # vehicle at several points along the trajectory
    num_points = 10
    indices = np.linspace(0, len(states)-1, num_points).astype(int)
    
    for idx in indices:
        # vehicle
        x, y, theta = states[idx, 0], states[idx, 1], states[idx, 2]
        
        # rectangle car shaoe
        car_length = unicycle.length
        car_width = unicycle.width
        
        corners = np.array([
            [-car_length/2, -car_width/2],
            [car_length/2, -car_width/2],
            [car_length/2, car_width/2],
            [-car_length/2, car_width/2]
        ])
        
        # rotate corners based on vehicle heading
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        
        rotated_corners = np.array([R @ corner for corner in corners])
        
        # translate corners to vehicle position
        translated_corners = rotated_corners + np.array([x, y])

        car_polygon = plt.Polygon(translated_corners, fill=True, color='g', alpha=0.5)
        ax.add_patch(car_polygon)
        
        front_x = x + 0.7 * car_length/2 * np.cos(theta)
        front_y = y + 0.7 * car_length/2 * np.sin(theta)
        ax.plot([x, front_x], [y, front_y], 'r-', linewidth=1.5)
    
    ax.set_title('Path Following with Obstacles')
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()