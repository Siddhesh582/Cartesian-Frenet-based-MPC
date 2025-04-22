'''
def step
def cartesian to frenet
def frenet to cartesian
def frenet motion model
def step frenet
def unicycle_collision_circle
def check collision                --->     distance < (uni_radius + obs_radius)

'''

import numpy as np
import matplotlib.pyplot as plt
from gymnasium import spaces
from scipy.interpolate import CubicSpline
from FrenetParameters import ReferencePathParameters

class Unicycle:
    def __init__(self, reference_path: ReferencePathParameters, v_min=0, v_max=1, w_min = 2* np.pi, w_max = 2 * np.pi):
        self.ref_path = reference_path
        
        self.action_space = spaces.Box(
            low = np.array([v_min, w_min]),
            high = np.array([v_max, w_max]),
            shape=(2,),
            dtype=float,
        )

        # Vehicle physical parameters
        self.length = 2.0  # Vehicle length (for visualization)
        self.width = 1.0   # Vehicle width (for visualization)
        
        self.uni_radius = 1.0  # Radius of the collision circle

    def step(self, current_state: np.ndarray, action: np.ndarray, dt: float = 0.1) -> np.ndarray:
        current_state = np.asarray(current_state).reshape((-1, 3))
        action = np.asarray(action).reshape((-1, 2))
        
        clipped_action = np.clip(action, self.action_space.low, self.action_space.high)
        
        next_state = np.empty_like(current_state)    

        next_state[:, 0] = current_state[:, 0] + dt * clipped_action[:, 0] * np.cos(current_state[:, 2])
        next_state[:, 1] = current_state[:, 1] + dt * clipped_action[:, 0] * np.sin(current_state[:, 2])
        next_state[:, 2] = current_state[:, 2] + dt * clipped_action[:, 1]

        next_state[:, 2] = np.arctan2(np.sin(next_state[:, 2]), np.cos(next_state[:, 2]))       # theta

        return next_state.squeeze()
    
    def cartesian_to_frenet(self, state, s_vals):
        single_input = not isinstance(state[0], (list, np.ndarray))         # check if single input
        if single_input:
            x, y, theta = state
            s_star, n, beta = self.ref_path.project_point(x, y, theta)
            
            return np.array([s_star, n, beta])
        else:
            if s_vals is None:
                s_vals = self.ref_path.s_vals
                
            results = np.zeros((len(state), 3))
            for i in range(len(state)):
                x, y, theta = state[i]
                results[i] = self.ref_path.project_point(x, y, theta)
                
            return results
        
    def frenet_to_cartesian(self, frenet_state):
        """
        Frenet state [s, n, beta] --> Cartesian state [x, y, theta]

        """
        single_input = not isinstance(frenet_state[0], (list, np.ndarray))
        
        if single_input:
            s, n, beta = frenet_state
            
            # x,y,theta from s,n,beta
            path_pos = self.ref_path.gamma(s)
            theta_path = self.ref_path.heading(s)
            
            # normal vector
            normal = np.array([-np.sin(theta_path), np.cos(theta_path)])
            
            # unicycle position
            x = path_pos[0] + n * normal[0]
            y = path_pos[1] + n * normal[1]
            
            # unicycle heading
            theta = theta_path + beta
            
            # normalize theta to [-pi, pi]
            theta = np.arctan2(np.sin(theta), np.cos(theta))
            
            return np.array([x, y, theta])
        else:
            results = np.zeros((len(frenet_state), 3))
            for i in range(len(frenet_state)):
                results[i] = self.frenet_to_cartesian(frenet_state[i])
            return results

    def motion_model_frenet(self, frenet_state, control):
        """
         Time derivative of Frenet state [s_dot, n_dot, beta_dot]
        """
        s, n, beta = frenet_state
        v, omega = control
        
        # curvature at current arc length directly from reference path
        kappa = self.ref_path.curvature(s)

        denom = 1 - n * kappa
        if abs(denom) < 1e-6:
            denom = 1e-6 * np.sign(denom)

        # Frenet coordinate dynamics
        s_dot = v * (np.cos(beta) / denom)
        n_dot = v * np.sin(beta)
        beta_dot = omega - (kappa * s_dot)
        
        return np.array([s_dot, n_dot, beta_dot])
    
    def step_frenet(self, current_frenet_state, control, dt=0.1):
        """ 
             Next Frenet state [s, n, beta]
        """
        # RK4 integration helper function
        def f(t, state): 
            return self.motion_model_frenet(state, control)

        # Runge-Kutta 4th order integration
        k1 = f(0, current_frenet_state)
        k2 = f(0, current_frenet_state + dt / 2 * k1)
        k3 = f(0, current_frenet_state + dt / 2 * k2)
        k4 = f(0, current_frenet_state + dt * k3)

        next_frenet_state = current_frenet_state + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
        
        # normalize beta to [-pi, pi]
        next_frenet_state[2] = np.arctan2(np.sin(next_frenet_state[2]), np.cos(next_frenet_state[2]))

        # Ensure s stays within valid range for the path
        next_frenet_state[0] = np.clip(next_frenet_state[0], self.ref_path.s_vals[0], self.ref_path.s_vals[-1])

        return next_frenet_state
    
    def unicycle_collision_circle(self, state):
        """
        Collision Circle for unicycle
        
        """
        x, y, _ = state  # Theta isn't needed for a circular representation
        
        return {
            'center': (x, y),
            'radius': self.vehicle_radius
        }
    
    def check_collision(self, state, obstacles):
        """
        collision check with any obstacles
        
        Args:
            state: Cartesian state [x, y, theta]
            obstacles: list of obstacle dictionaries with center and radius
            
        Returns:
            collision: boolean indicating collision
        """
        uni_circle = self.unicycle_collision_circle(state)
        uni_center = np.array(uni_circle['center'])
        uni_radius = uni_circle['radius']
        
        # Check each obstacle
        for obstacle in obstacles:
            obs_center = np.array(obstacle['center'])
            if 'radius' in obstacle:
                obs_radius = obstacle['radius']
            else:
                # approximate a circle
                obs_width = obstacle['width']
                obs_height = obstacle['height']
                obs_radius = np.sqrt(obs_width**2 + obs_height**2) / 2
            
            # distance between centers 
            distance = np.linalg.norm(uni_center - obs_center)
            
            # collision check
            if distance < (uni_radius + obs_radius):
                return True
                
        return False
    
    def plot_vehicle(self, state, ax=None, alpha=1.0):
        """
        representation of the unicycle vehicle
        
        Args:
            state: Vehicle state [x, y, theta]
            ax: Matplotlib axis to plot on
            alpha: Transparency level
        """
        if ax is None:
            fig, ax = plt.subplots()
        
        x, y, theta = state
        
        # Simple car shape using a rectangle
        car_length = self.length
        car_width = self.width
        
        # Calculate corner positions of the car rectangle
        corners = np.array([
            [-car_length/2, -car_width/2],
            [car_length/2, -car_width/2],
            [car_length/2, car_width/2],
            [-car_length/2, car_width/2]
        ])
        
        # Rotate corners based on vehicle heading
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        
        rotated_corners = np.array([R @ corner for corner in corners])
        
        # Translate corners to vehicle position
        translated_corners = rotated_corners + np.array([x, y])
        
        # Draw the vehicle
        car_polygon = plt.Polygon(translated_corners, fill=True, color='g', alpha=alpha)
        ax.add_patch(car_polygon)
        
        # Add direction indicator (front of vehicle)
        front_x = x + 0.7 * car_length/2 * np.cos(theta)
        front_y = y + 0.7 * car_length/2 * np.sin(theta)
        ax.plot([x, front_x], [y, front_y], 'r-', linewidth=1.5)
        
        # Draw the collision circle
        collision_circle = plt.Circle((x, y), self.uni_radius, fill=False, color='r', linestyle='--', alpha=alpha)
        ax.add_patch(collision_circle)
        
        return ax
        