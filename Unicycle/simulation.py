'''
def simulate system
def plot simulation result
def create path following animation
'''


import numpy as np
import matplotlib.pyplot as plt
import time

from FrenetParameters import ReferencePathParameters
from Unicycle import Unicycle
from Controller import FrenetCartesianMPC

def simulate_system(mpc_controller: FrenetCartesianMPC, ref_path: ReferencePathParameters, unicycle_model: Unicycle, simulation_steps=100):
    """Simulate the unicycle model with MPC control"""
    
    # initial state
    x0, y0 = ref_path.gamma(0)      # x,y cartesian position for s=0 arc length
    theta0 = ref_path.heading(0)
    s0 = 0
    n0 = 0
    state = np.array([x0, y0, theta0, s0, n0])
    
    # to log simualtion results
    states = np.zeros((simulation_steps+1, 5))
    controls = np.zeros((simulation_steps, 2))
    times = np.zeros(simulation_steps)
    predicted_trajectories = []
    
    # log the initial state
    states[0, :] = state
    
    # load default obstacles
    mpc_controller.set_obstacles(ref_path.default_obstacles)   # empty list for no obstacles
    
    ''' Simulation Loop'''
    for i in range(simulation_steps):
        ''' 
        Flow Structure:

        current arc length --> reference desired linearly spaced arc length vals --> 
        controller ( output: optimal control, trajectory based on s_ref, current state) considering minimal lateral deviation (n), etc.
        --> optimal control applied on unicycle in cartesian -> cartesian next state -> it's frenet state -> augmented state vector --> get the arc length 
        --> repeat loop
        
        
        '''
        # Reference trajectory (desired progress along path)
        s_current = state[3]                                                                       # current arc length
        s_ref = np.linspace(s_current, min(s_current + 10, ref_path.last_s_val), mpc_controller.N+1)    # linearly spaced values of desired arc length
        
        # Solve MPC problem
        t_start = time.time()
        u, predicted_trajectory = mpc_controller.solve(state, s_ref)    # get optimal control, trajectory based on current state + reference trajectory
        
        print(f"Step {i}: control = [{u[0]:.3f}, {u[1]:.3f}], position = ({state[0]:.2f}, {state[1]:.2f})")
        if predicted_trajectory is None:
            print(f"  Prediction: None (solver failed)")
        else:
            print(f"  Prediction horizon: {predicted_trajectory.shape[1]} steps")
                
        if u[0] == 0.0 and u[1] == 0.0:
            # backup controller
            print(f"Using fallback controller at step {i}")
            v_fallback = mpc_controller.v_ref 
            u = np.array([v_fallback, 0.0])  # Forward movement, no turning
        
        times[i] = time.time() - t_start
        
        # log the predicted trajectories
        predicted_trajectories.append(predicted_trajectory)
        
        # Apply control
        controls[i, :] = u
        
        # cartesian states
        cartesian_state = state[:3]  # [x, y, theta]
        cartesian_next = unicycle_model.step(cartesian_state, u, dt=mpc_controller.dt)  # applied the calculated optimal control

        # cartesian -> frenet
        frenet_next = unicycle_model.cartesian_to_frenet(cartesian_next, ref_path.s_vals) # frenet state based on this optimal control in cartesian
        
        # augmented state
        state = np.array([
            cartesian_next[0],  # x
            cartesian_next[1],  # y
            cartesian_next[2],  # theta
            frenet_next[0],     # s
            frenet_next[1]      # n
        ])
        
        # log the state
        states[i+1, :] = state
        
        # progress check
        if (i+1) % 10 == 0:
            print(f"Simulation step {i+1}/{simulation_steps}, position: ({state[0]:.2f}, {state[1]:.2f}), progress: {state[3]:.2f}")
    
    return states, controls, times, predicted_trajectories

def plot_simulation_results(ref_path, states, controls, unicycle_model):
    """Plot simulation results with visualization of covering circles"""
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Plot path and trajectory
    ax1 = plt.subplot(2, 3, 1)
    
    # Plot reference path
    s_vals = ref_path.s_vals
    x_vals = ref_path.x_vals
    y_vals = ref_path.y_vals
    ax1.plot(x_vals, y_vals, 'k--', linewidth=1.5, label='Reference Path')
    
    # Plot actual trajectory
    ax1.plot(states[:, 0], states[:, 1], 'b-', linewidth=2, label='Vehicle Trajectory')
    
    # Visualize the unicycle model at selected points
    sample_indices = np.linspace(0, states.shape[0]-1, 5).astype(int)
    for idx in sample_indices:
        vehicle_state = states[idx, :3]  # [x, y, theta]
        unicycle_model.plot_vehicle(vehicle_state, ax=ax1, alpha=0.5)
    
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_title("Path Tracking")
    ax1.axis('equal')  # This ensures equal scaling for x and y axes
    ax1.grid(True)
    ax1.legend()
    
    # Plot states
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(np.arange(len(states)), states[:, 3], 'b-', label='s (progress)')
    ax2.plot(np.arange(len(states)), states[:, 4], 'r-', label='n (lateral offset)')
    ax2.grid(True)
    ax2.legend()
    ax2.set_title('Frenet States')
    ax2.set_xlabel('Simulation Step')
    ax2.set_ylabel('Value')
    
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(np.arange(len(states)), states[:, 2], 'g-', label='theta (heading)')
    ax3.grid(True)
    ax3.legend()
    ax3.set_title('Heading')
    ax3.set_xlabel('Simulation Step')
    ax3.set_ylabel('Angle (rad)')
    
    # Plot controls
    ax4 = plt.subplot(2, 3, 4)
    ax4.step(np.arange(len(controls)), controls[:, 0], 'b-', label='v (velocity)')
    ax4.grid(True)
    ax4.legend()
    ax4.set_title('Velocity Control')
    ax4.set_xlabel('Simulation Step')
    ax4.set_ylabel('Velocity (m/s)')
    
    ax5 = plt.subplot(2, 3, 5)
    ax5.step(np.arange(len(controls)), controls[:, 1], 'g-', label='omega (angular velocity)')
    ax5.grid(True)
    ax5.legend()
    ax5.set_title('Angular Velocity Control')
    ax5.set_xlabel('Simulation Step')
    ax5.set_ylabel('Angular Velocity (rad/s)')
    
    # Plot lateral error vs progress
    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(states[:, 3], states[:, 4], 'r-', linewidth=2)
    ax6.grid(True)
    ax6.set_xlabel('Path progress (s)')
    ax6.set_ylabel('Lateral error (n)')
    ax6.set_title('Lateral Error vs Progress')
    
    plt.tight_layout()
    plt.show()


# def obstacles(path_length=50):
#     """Create custom obstacles along the path"""
#     return [
#         {'center': (10, 2.0), 'width': 2.0, 'height': 1.2},
#         {'center': (20, -1.5), 'width': 1.5, 'height': 1.5},
#         {'center': (30, 1.0), 'width': 1.0, 'height': 1.0},
#         {'center': (35, -1.0), 'width': 1.0, 'height': 2.0},
#         {'center': (40, 0.5), 'width': 1.3, 'height': 1.3}
#     ]


