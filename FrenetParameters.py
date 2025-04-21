'''
def gamma 

def compute heading
def heading

def compute curvature 
def curvature

def s_star
def project point
def plot_curvy_road
def curvature

'''


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

class ReferencePathParameters:
    def __init__(self, s_vals, x_vals, y_vals,  lane_width ):
        self.s_vals = s_vals
        self.x_vals = x_vals
        self.y_vals = y_vals

        self.lane_width = lane_width

        self.x_spline = CubicSpline(s_vals, x_vals)
        self.y_spline = CubicSpline(s_vals, y_vals)

        self.last_s_val = s_vals[-1]

        self.default_obstacles = []      # can be setup in main

        # Precompute curvature at each s value for faster lookup
        self.curvature_values = np.array([self.compute_curvature(s) for s in s_vals])
        self.curvature_spline = CubicSpline(s_vals, self.curvature_values)
        
        # Precompute heading at each s value
        self.heading_values = np.array([self.compute_heading(s) for s in s_vals])
        self.heading_spline = CubicSpline(s_vals, self.heading_values)


    def gamma(self, s):
        """Returns [x(s), y(s)] for given arc length s"""
        return np.array([self.x_spline(s), self.y_spline(s)])
        
#####################

    '''
    Compute Theta: compute_heading (arctan2 (dy,dx)) --> CubicSpline of these heading values
    '''
    def compute_heading(self, s):
        """Compute path heading (tangent angle) at arc-length s"""
        dx = self.x_spline.derivative(1)(s)
        dy = self.y_spline.derivative(1)(s)
        return np.arctan2(dy, dx)

    def heading(self, s):
        """Returns path heading (tangent angle) at arc-length s using precomputed spline"""
        # Ensure s is within valid range
        s = np.clip(s, self.s_vals[0], self.s_vals[-1])
        return self.heading_spline(s)

######################

    def compute_curvature(self, s):                              # curvature => kappa
        """Returns curvature at arc-length s"""
        dx = self.x_spline.derivative(1)(s) 
        dy = self.y_spline.derivative(1)(s)
        ddx = self.x_spline.derivative(2)(s)
        ddy = self.y_spline.derivative(2)(s)

        return (dx * ddy - dy * ddx) / ((dx**2 + dy**2)**1.5 + 1e-8)  # positive value => left turn; negative values => right turn
    
    def curvature(self, s):
        """Returns curvature at arc-length s using precomputed spline"""
        # Ensure s is within valid range
        s = np.clip(s, self.s_vals[0], self.s_vals[-1])
        return self.curvature_spline(s)

#####################

    def s_star(self, x, y):
        """
        Find the closest point on the path to the given (x,y) coordinates
        Returns the corresponding s value
        """
        # Create a vector of points from the reference path
        path_points = np.column_stack((self.x_vals, self.y_vals))
        
        # Calculate distances to all path points
        point = np.array([x, y])
        distances = np.linalg.norm(path_points - point, axis=1)
        
        # Find index of closest point
        closest_idx = np.argmin(distances)
        
        # Return corresponding s value
        return self.s_vals[closest_idx]
    
    def project_point(self, x, y, theta):
        """
        Takes Unicycle Cartesian (x,y) --> finds s* --> gets x,y,theta for that s* --> computes n, beta  --> final frenet state: s*, n, beta
        """
        # For now , assuming the closest s as an approximation for the frenet state
        s = self.s_star(x, y)
        
        # Refined projection can be done using gradient descent for better accuracy
        
        x_uni, y_uni = self.gamma(s)      # (x,y) for arc length 's'
        uni_heading = self.heading(s)     # theta at arc length 's'
        
        # normal vector to the path
        normal_x = -np.sin(uni_heading)
        normal_y = np.cos(uni_heading)
        
        # lateral distance
        dx = x - x_uni
        dy = y - y_uni
        n = dx * normal_x + dy * normal_y
        
        #beta
        beta = theta - uni_heading

        return s, n, beta
    

###########################
##     PLOT ROADS
##########################

    # def plot_straight_road(self, length=30, lane_width=3.5, obstacles=None, center_y=0.0, show=True, ax=None):
    #     """
    #     Plot a straight road layout with dashed centerline and lane edges.

    #     Args:
    #         length (float): Length of the road in X direction.
    #         lane_width (float): Width of the lane (centerline to each edge).
    #         obstacles (list): Optional list of obstacle dicts.
    #         center_y (float): Vertical center of the road.
    #         show (bool): Whether to call plt.show().
    #         ax (matplotlib axis): Optional axis to draw on.
    #     """
    #     x_vals = np.linspace(0, length, 300)
    #     y_center = np.ones_like(x_vals) * center_y
    #     y_left = y_center + (lane_width / 2)
    #     y_right = y_center - (lane_width / 2)

    #     if ax is None:
    #         fig, ax = plt.subplots(figsize=(10, 6))

    #     # Centerline (dashed)
    #     ax.plot(x_vals, y_center, linestyle='--', color='blue', linewidth=2, label='Centerline')

    #     # Lane boundaries (solid)
    #     ax.plot(x_vals, y_left, linestyle='-', color='black', linewidth=1.5, label='Lane Edge')
    #     ax.plot(x_vals, y_right, linestyle='-', color='black', linewidth=1.5)

    #     # Plot obstacles
    #     if obstacles is None:
    #         obstacles = self.default_obstacles

    #     for i, obs in enumerate(obstacles):
    #         cx, cy = obs['center']
    #         width = obs['width']
    #         height = obs['height']
    #         rect = plt.Rectangle(
    #             (cx - width / 2, cy - height / 2),
    #             width,
    #             height,
    #             color='red',
    #             alpha=0.6,
    #             label='Obstacle' if i == 0 else None
    #         )
    #         ax.add_patch(rect)

    #     ax.set_xlabel("X")
    #     ax.set_ylabel("Y")
    #     ax.set_title("Straight Road View")
    #     ax.axis('equal')
    #     ax.grid(True)
    #     ax.legend()

    #     if show:
    #         plt.show()

    #     return ax

    def plot_curvy_road(self, obstacles=None, lane_width=None, show=True, ax=None, 
                         color='blue', label='Centerline'):
        """
        Plot a curvy road layout with the reference path as centerline.
        
        Args:
            obstacles (list): Optional list of obstacle dicts.
            lane_width (float): Width of the lane.
            show (bool): Whether to call plt.show().
            ax (matplotlib axis): Optional axis to draw on.
            color (str): Color for the centerline.
            label (str): Label for the centerline.
        """
        s = self.s_vals
        x_center = self.x_spline(s)
        y_center = self.y_spline(s)
        theta = np.array([self.heading(s_val) for s_val in s])

        if lane_width is None:
            lane_width =  self.lane_width

        # Compute lane boundaries
        x_left  = x_center - (lane_width / 2) * np.sin(theta)
        y_left  = y_center + (lane_width / 2) * np.cos(theta)
        x_right = x_center + (lane_width / 2) * np.sin(theta)
        y_right = y_center - (lane_width / 2) * np.cos(theta)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Plot centerline as dashed
        ax.plot(x_center, y_center, linestyle='--', color=color, linewidth=2, label=label)

        # Plot road edges as solid lines
        ax.plot(x_left, y_left, linestyle='-', color='black', linewidth=1.5, label='Lane Edge' if obstacles is None else None)
        ax.plot(x_right, y_right, linestyle='-', color='black', linewidth=1.5)

        # Plot obstacles
        if obstacles is None:
            obstacles = self.default_obstacles

        for i, obs in enumerate(obstacles):
            cx, cy = obs['center']
            radius = obs['radius']  # Make sure each obstacle has a 'radius' key
            
            circle = plt.Circle(
                (cx, cy),
                radius,
                color='red',
                alpha=0.6,
                label='Obstacle' if i == 0 else None
            )
            ax.add_patch(circle)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Reference Path and Lane")
        ax.axis('equal')
        ax.grid(True)
        ax.legend()

        if show:
            plt.show()

        return ax
        
    def plot_curvature(self, show=True):
        """Plot the curvature along the path"""
        plt.figure(figsize=(10, 4))
        plt.plot(self.s_vals, self.curvature_values)
        plt.xlabel('Arc length (s)')
        plt.ylabel('Curvature (κ)')
        plt.title('Path Curvature')
        plt.grid(True)
        
        if show:
            plt.show()

# s_vals = np.linspace(0, 100, 2000)
# x_vals = s_vals
# y_vals = 1 * np.sin(0.05 * s_vals)
# ref_path = ReferencePathParameters(s_vals, x_vals, y_vals, lane_width=15)
# ref_path.plot_curvy_road()
