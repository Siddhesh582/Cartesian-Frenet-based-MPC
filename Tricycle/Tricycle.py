import numpy as np
import matplotlib.pyplot as plt

class TricycleKinematics:
    """
    Tricycle Kinematics in Cartesian Coordinates
    
    The state vector ζp = [x, y, φ, v, α], where:
    - x, y: position in the Cartesian coordinate frame
    - φ: heading angle
    - v: velocity at the wheel
    - α: steering angle
    
    The control vector u = [a, omega], where:
    - a: acceleration at the wheel
    - omega: steering rate

    """
    
    def __init__(self, wheelbase=1.03):
        self.d = 1.03                # wheelbase => distance b/w rear and front axle in meters
        self.v_min = 0.0                  # linear velocity (m/s)
        self.v_max = 1.0   
        self.v = np.clip(self.v, self.v_min, self.v_max)

        self.alpha_min = -np.pi/4         # steering angle (rad)
        self.alpha_max = np.pi/4  
        self.alpha = np.clip(self.alpha, self.alpha_min, self.alpha_max)

        self.a_min = -0.5                 # acceleration (m/s^2) 
        self.a_max = 0.5
        self.a = np.clip(self.a, self.a_min, self.a_max)

        self.omega_min = -0.5             # steering rate (rad/s)
        self.omega_max = 0.5
        self.omega = np.clip(self.omega, self.omega_min, self.omega_max) 

        self.kappa_min = -0.25           # track curvature (1/m)
        self.kappa_max = 0.25
        self.kappa = np.clip(self.kappa, self.kappa_min, self.kappa_max)

        self.length_agv = 2.914             # agv length (m)
        self.breadth_agv = 1.115            # metres
    
    def motion_model(self, t, state, control):
        '''
        Computes time derivative of state.

        Arguments: 
        t => time
        state (np.ndarray) => current state [x,y,phi,v,alpha]
        control (np.ndarray) => control input [a, omega]

        Returns:
        motion model = [x_dot, y_dot, phi_dot, v_dot = a, alpha_dot = omega]
        '''

        x, y, phi, v, alpha = state                           # unpacking state vector 
        a, omega = control                                    # unpacking control input

        #time derivatives
        x_dot = v * np.cos(alpha) * np.cos(phi)      
        y_dot = v * np.cos(alpha) * np.sin(phi)
        phi_dot = (v / self.d) * np.sin(alpha)
        v_dot = a
        alpha_dot = omega
        
        return np.array([x_dot, y_dot, phi_dot, v_dot, alpha_dot])     
    
    def ref_path_func(s):
        return np.array([s, 0.0])
    
    def ref_heading_func(s):
        return 0.0
    
    def cartesian_to_frenet(self, state, ref_path_func, ref_heading_func, s_values):
        """
        Cartesian Frame ---> Frenet Frame

        Args:
            state: np.array([x, y, phi]) - AGV Cartesian state
            ref_path_func: function(s) -> np.array([x, y]) - gives reference path position at arc length s
            ref_heading_func: function(s) -> float - gives heading (tangent angle) at s
            s_vals: np.array of s values to search for closest projection

        Returns:
            s_star: closest arc length
            n: lateral deviation from path
            beta: heading difference
        """
        x, y, phi = state
        p_agv = np.array([x, y])                            # AGV position w.r.t world frame
        
        # track progress  
        min_dist = float('inf')                             #initially inf
        s_star = None                                       # s* => arc length on reference curve where the distance b/w AGV and curve is least
        
        for s in s_values:
            gamma_p = ref_path_func(s)                      # gamma_p = [x, y] => point on reference curve at arc length s 
            
            dist = np.linalg.norm(p_agv - gamma_p)          # minimize (s) given dist
            if dist < min_dist:
                min_dist = dist
                s_star = s

        gamma_p_star = ref_path_func(s_star)                # i.e. [x,y] on reference curve at arc-length s*
        gamma_phi_star = ref_heading_func(s_star)           # angle phi from AGV to reference curve at arc length s*

        # lateral displacement
        e_n = np.array([-np.sin(gamma_phi_star), np.cos(gamma_phi_star)])
        n = np.dot(e_n, p_agv - gamma_p_star)

        # heading difference
        beta = phi - gamma_phi_star                          

        return np.array([s_star, n, beta])                    # frenet state
    
    def motion_model_frenet(self, t, state, control):
        s, n, beta, v, alpha = state
        a, omega = control

        # Guard against divide-by-zero when close to curvature center
        denom = 1 - n * self.kappa
        if abs(denom) < 1e-6:
            denom = 1e-6 * np.sign(denom)

        # time derivatives
        s_dot = v * np.cos(alpha) * np.cos(beta) / denom
        n_dot = v * np.cos(alpha) * np.sin(beta)
        beta_dot = (v / self.d) * np.sin(alpha) - self.kappa * s_dot
        v_dot = a
        alpha_dot = omega

        return np.array([s_dot, n_dot, beta_dot, v_dot, alpha_dot])

