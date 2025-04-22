import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
import time as t
from FrenetParameters import ReferencePathParameters
from Unicycle import Unicycle

class FrenetCartesianMPC:
    def __init__(self, ref_path: ReferencePathParameters, unicycle_model: Unicycle):
        """
        Initialize the MPC controller for unicycle model in Frenet-Cartesian frames
        
        Args:
            ref_path: ReferencePath object => contains reference path information
            unicycle_model: Unicycle model object
        """
        # Model parameters
        self.dt = 0.1  # Sampling time
        self.N = 45    # Prediction horizon
        
        # instances
        self.ref_path = ref_path
        self.unicycle = unicycle_model
        
        # control limits
        action_bounds = self.unicycle.action_space
        self.v_min = action_bounds.low[0]
        self.v_max = action_bounds.high[0]
        self.omega_min = action_bounds.low[1]
        self.omega_max = action_bounds.high[1]
        
        # Unicycle model state and control dimensions
        self.nx = 5    # nx => number of state variables => [x, y, theta, s, n]  (Cartesian + Frenet states) 
        self.nu = 2    # nu => number of control input => [v, omega]
        
        # Lateral deviation bounds
        if hasattr(self.ref_path, 'lane_width'):
            self.n_max = (self.ref_path.lane_width / 2.0) - 0.2  # 0.2 is the safety buffer
            self.n_min = -((self.ref_path.lane_width / 2.0) - 0.2)
        else:
            self.n_max = 2.0
            self.n_min = -2.0

        # setting up the saftey circle for unicycle 
        self.circle_radius = self.unicycle.uni_radius
        
        # getting default obstacles setup for collision avoidance
        self.obstacles = self.ref_path.default_obstacles if hasattr(self.ref_path, 'default_obstacles') else []  
        self.obstacle_circles = []  
        
        # Cost function weights
        self.Q_s = 20.0       # track progress => encourage forward movement
        self.Q_s_ref = 10.0    # reference progress tracking
        self.Q_n = 25.0       # lateral deviation => minimize 
        self.Q_theta = 5.0    # Weight for heading
        self.R_v = 1.0        # Weight for velocity control
        self.R_omega = 10.0   # Weight for angular velocity control
        
        self.v_ref = 0.8      # to ensure non zero velocity due to control cost

        # Slack variable penalty weights (adding these lines fixes the error)
        self.obs_slack_weight = 1000.0    # Obstacle avoidance slack penalty
        self.curv_slack_weight = 100.0    # Curvature constraint slack penalty
        self.lat_slack_weight = 500.0     # Lateral deviation slack penalty

        # Build the MPC solver
        self.NMPC_setup()
    
    def NMPC_setup(self):
        ''' Setting up parameters '''
        self.opti = ca.Opti()  # CasADi Opti class instance
        
        # state variables over the prediction horizon
        self.X = self.opti.variable(self.nx, self.N+1)     # pedict the state variables till N+1 steps
        x = self.X[0, :]
        y = self.X[1, :]
        theta = self.X[2, :]
        s = self.X[3, :]
        n = self.X[4, :]
        
        # control variables over the prediction horizon
        self.U = self.opti.variable(self.nu, self.N)
        v = self.U[0, :]
        omega = self.U[1, :]
        
        # Parameters for initial state and reference trajectory
        self.X0 = self.opti.parameter(self.nx)
        self.S_ref = self.opti.parameter(self.N+1)
        
        ''' Cost Function '''
        cost = 0
        
        # adding components
        for k in range(self.N):
            cost -= self.Q_s * (s[k+1] - s[k])   # track progress maximization (negative cost for s)
            
            # Reference progress tracking (optional)
            cost += self.Q_s_ref * (s[k] - self.S_ref[k])**2
            
            cost += self.Q_n * n[k]**2           # path tracking cost 
            
            path_heading = self._casadi_path_heading(s[k])
            heading_diff = theta[k] - path_heading
            cost += self.Q_theta * (ca.sin(heading_diff))**2     # heading cost 
            
            # control cost
            cost += self.R_v * (v[k] - self.v_ref)**2
            cost += self.R_omega * omega[k]**2

            #control rate penalties
            if k > 0:
                # penalize control changes between consecutive steps
                cost += 2.0 * (v[k] - v[k-1])**2       # linearvelocity 
                cost += 5.0 * (omega[k] - omega[k-1])**2  # angular velocity 
        
        # terminal cost
        cost += self.Q_n * n[self.N]**2
        cost += self.Q_s_ref * (s[self.N] - self.S_ref[self.N])**2

        # Add terminal heading cost
        path_heading_terminal = self._casadi_path_heading(s[self.N])
        heading_diff_terminal = theta[self.N] - path_heading_terminal
        cost += self.Q_theta * 2.0 * (heading_diff_terminal)**2  # Higher weight for terminal heading
        
        # Set the cost function
        self.opti.minimize(cost)
        
        ''' Model Dynamics '''
        for k in range(self.N):
            # RK4 integration
            k1 = self.unicycle_model(self.X[:, k], self.U[:, k])                       # k1 = f(x_k, u_k)
            k2 = self.unicycle_model(self.X[:, k] + self.dt/2 * k1, self.U[:, k])      # k2 = f(x_k + 0.5*dt * k1, u_k)
            k3 = self.unicycle_model(self.X[:, k] + self.dt/2 * k2, self.U[:, k])      # k3 = f(x_k + 0.5*dt * k2, u_k)
            k4 = self.unicycle_model(self.X[:, k] + self.dt * k3, self.U[:, k])        # k4 = f(x_k + dt * k3, u_k)
            
            x_next = self.X[:, k] + self.dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            self.opti.subject_to(self.X[:, k+1] == x_next)                             # next state constraint
        
        self.opti.subject_to(self.X[:, 0] == self.X0)                                  # initial state constraint
        
        # Curvature constraint slack variables
        curv_slack_vars = []
        for k in range(self.N+1):
            # One slack variable for each curvature constraint (upper and lower)
            slack_upper = self.opti.variable()
            slack_lower = self.opti.variable()
            
            # Slack should be non-negative
            self.opti.subject_to(slack_upper >= 0)
            self.opti.subject_to(slack_lower >= 0)
            
            curv_slack_vars.append((slack_upper, slack_lower))
            
            # Add penalty to cost function
            cost += self.curv_slack_weight * slack_upper
            cost += self.curv_slack_weight * slack_lower
        
        # Lateral deviation constraint slack variables
        lat_slack_vars = []
        for k in range(self.N+1):
            # One slack variable for each lateral constraint (upper and lower)
            slack_upper = self.opti.variable()
            slack_lower = self.opti.variable()
            
            # Slack should be non-negative
            self.opti.subject_to(slack_upper >= 0)
            self.opti.subject_to(slack_lower >= 0)
            
            lat_slack_vars.append((slack_upper, slack_lower))
            
            # Add penalty to cost function
            cost += self.lat_slack_weight * slack_upper
            cost += self.lat_slack_weight * slack_lower
        
        # Apply softened constraints
        for k in range(self.N+1):
            # Softened curvature constraints
            kappa = self._casadi_path_curvature(s[k])
            slack_upper, slack_lower = curv_slack_vars[k]
            
            self.opti.subject_to(n[k] * kappa <= 0.99 + slack_upper)
            self.opti.subject_to(n[k] * kappa >= -0.99 - slack_lower)
            
            # Softened lateral deviation constraints
            slack_upper, slack_lower = lat_slack_vars[k]
            
            self.opti.subject_to(n[k] <= self.n_max + slack_upper)
            self.opti.subject_to(n[k] >= self.n_min - slack_lower)
        
        # velocity (linear + angular) constraints - these remain hard constraints
        for k in range(self.N):
            self.opti.subject_to(self.v_min <= v[k])
            self.opti.subject_to(v[k] <= self.v_max)
            self.opti.subject_to(self.omega_min <= omega[k])
            self.opti.subject_to(omega[k] <= self.omega_max)
        
        # ---- Soft obstacle avoidance constraints ----
        # Add slack variables for obstacle avoidance
        obs_slack_vars = []
        if self.obstacles and self.obstacle_circles:
            for k in range(1, self.N+1):
                for _ in self.obstacle_circles:
                    # Create a slack variable for each obstacle at each time step
                    slack = self.opti.variable()
                    # Slack should be non-negative
                    self.opti.subject_to(slack >= 0)
                    obs_slack_vars.append(slack)
                    
                    # Add penalty to cost function
                    cost += self.obs_slack_weight * slack
        
        # Add softened obstacle avoidance constraints
        slack_idx = 0
        if self.obstacles and self.obstacle_circles:
            for k in range(1, self.N+1):
                for obs_circle in self.obstacle_circles:
                    obs_x, obs_y = obs_circle['center']
                    obs_radius = obs_circle['radius']
                    
                    # Calculate distance between vehicle and obstacle
                    dist = ca.sqrt((x[k] - obs_x)**2 + (y[k] - obs_y)**2)
                    
                    # Minimum required distance: sum of radii plus safety margin
                    min_dist = self.circle_radius + obs_radius + 0.05
                    
                    # Add softened constraint: distance >= minimum distance - slack
                    self.opti.subject_to(dist >= min_dist - obs_slack_vars[slack_idx])
                    
                    slack_idx += 1
        
        # Set the final cost function
        self.opti.minimize(cost)
        
        ''' 
        NMPC Solver 
        
        Returns: optimized control sequence considering => dynamics + constraints + cost function 
        
        '''
        opts = {
            'ipopt.print_level': 0,
            'ipopt.sb': 'yes',
            'print_time': 0,
            'ipopt.max_iter': 500,  # Increase max iterations
            'ipopt.tol': 1e-4       # Slightly relax tolerance for faster convergence
        }
        self.opti.solver('ipopt', opts)   # IPOPT => gradient based NMPC solver 
    
    def unicycle_model(self, state, control):
        """
        Unicycle dynamics in combined Frenet-Cartesian coordinates
        
        Args:
            state: Current state [x, y, theta, s, n]
            control: Control input [v, omega]
        
        Returns:
            State derivatives [dx, dy, dtheta, ds, dn]
        """
        x, y, theta, s, n = state[0], state[1], state[2], state[3], state[4]
        v, omega = control[0], control[1]
        
        # Cartesian dynamics
        dx = v * ca.cos(theta)
        dy = v * ca.sin(theta)
        dtheta = omega
        
        # Get path curvature and heading
        kappa = self._casadi_path_curvature(s)
        path_heading = self._casadi_path_heading(s)
        
        # Calculate beta (heading difference to path)
        beta = theta - path_heading
        
        # Frenet dynamics - adapted from the Unicycle class motion_model_frenet
        denom = 1 - n * kappa
        # Add small regularization to prevent division by zero
        denom = ca.if_else(ca.fabs(denom) < 1e-6, 1e-6 * ca.sign(denom), denom)
        
        ds = v * ca.cos(beta) / denom
        dn = v * ca.sin(beta)
        
        return ca.vertcat(dx, dy, dtheta, ds, dn)
    
    def _casadi_path_curvature(self, s):
        """
        Create a CasADi function for path curvature interpolation
        Using a simplified approximation for CasADi compatibility
        """
        # This is a simplified approximation - in a real implementation,
        # you would create a more accurate CasADi interpolation function
        # based on your reference path data
        s_max = self.ref_path.last_s_val
        return 0.05 * ca.sin(0.1 * s)
    
    def _casadi_path_heading(self, s):
        """
        Create a CasADi function for path heading interpolation
        Using a simplified approximation for CasADi compatibility
        """
        # We need to approximate the heading function from ReferencePathParameters
        # for CasADi compatibility
        return ca.arctan2(0.05 * ca.cos(0.1 * s), 1.0)
    
    def set_reference_velocity(self, v_ref):
        self.v_ref = v_ref
    
    def solve(self, current_state, s_ref):
        """
        Solve the MPC problem for the current state
        
        Args:
            current_state: Current state [x, y, theta, s, n]
            s_ref: Reference progress along path for the horizon
        
        Returns:
            Optimal control input [v, omega]
        """
        # Create a speed ramp-up for smoother starts
        current_velocity = np.linalg.norm(current_state[:2])
        if current_velocity < 0.1:  # If nearly stopped
            # Create a gradually increasing reference velocity
            self.v_ref = min(0.8, current_velocity + 0.05)  # Gradually increase to max 0.8
        else:
            self.v_ref = 0.8  # Normal reference velocity
            
        # Set the initial state parameter
        self.opti.set_value(self.X0, current_state)
        
        # Set the reference trajectory parameter
        self.opti.set_value(self.S_ref, s_ref)
        
        # Solve the optimization problem
        try:
            sol = self.opti.solve()
            # Extract the first control action
            u_optimal = sol.value(self.U)[:, 0]
            return u_optimal, sol.value(self.X)
        except Exception as e:
            print(f"Error solving MPC problem: {str(e)}")
            
            # Check if any obstacles are nearby
            x, y = current_state[0], current_state[1]
            for i, obs in enumerate(self.obstacles):
                if 'center' in obs and 'radius' in obs:
                    obs_x, obs_y = obs['center']
                    obs_r = obs['radius']
                    dist = np.sqrt((x - obs_x)**2 + (y - obs_y)**2)
                    print(f"Distance to obstacle {i}: {dist:.2f}, min safe: {self.circle_radius + obs_r + 0.05:.2f}")
            
            return np.array([0.0, 0.0]), None
    
    def set_obstacles(self, obstacles):
        """
        Set obstacles for collision avoidance using a single circle per obstacle
        
        Args:
            obstacles: List of obstacle dictionaries with center, width, height
        """
        if obstacles is None:
            if hasattr(self.ref_path, 'default_obstacles'):
                self.obstacles = self.ref_path.default_obstacles
            else:
                self.obstacles = []
        else:
            self.obstacles = obstacles
        
        # Convert obstacles to circle representation
        self.obstacle_circles = []
        
        for obstacle in self.obstacles:
            # Keep obstacles that already have a radius
            if 'radius' in obstacle:
                self.obstacle_circles.append(obstacle)
        
        # Rebuild the MPC problem with new obstacles
        self.NMPC_setup()