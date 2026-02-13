"""
Unicycle kinematic model.
"""

import numpy as np
import matplotlib.pyplot as plt


class Vehicle:
    
    def __init__(self, v_min=0.0, v_max=1.0, w_min=-np.pi, w_max=np.pi):
        self.v_min = v_min
        self.v_max = v_max
        self.w_min = w_min
        self.w_max = w_max
        
        self.length = 1.0
        self.width = 0.5
        self.radius = 0.5
    
    def step(self, state, u, dt=0.1):
        """Euler integration: state = [x, y, theta], u = [v, w]"""
        x, y, th = state
        v = np.clip(u[0], self.v_min, self.v_max)
        w = np.clip(u[1], self.w_min, self.w_max)
        
        x_next = x + dt * v * np.cos(th)
        y_next = y + dt * v * np.sin(th)
        th_next = th + dt * w
        th_next = np.arctan2(np.sin(th_next), np.cos(th_next))
        
        return np.array([x_next, y_next, th_next])
    
    def check_collision(self, state, obstacles):
        x, y = state[0], state[1]
        for obs in obstacles:
            ox, oy = obs['center']
            r = obs['radius']
            d = np.sqrt((x - ox)**2 + (y - oy)**2)
            if d < self.radius + r:
                return True
        return False
    
    def clearance(self, state, obstacles):
        x, y = state[0], state[1]
        min_c = float('inf')
        for obs in obstacles:
            ox, oy = obs['center']
            r = obs['radius']
            d = np.sqrt((x - ox)**2 + (y - oy)**2)
            c = d - self.radius - r
            min_c = min(min_c, c)
        return min_c
    
    def draw(self, state, ax, alpha=1.0, color='lime'):
        x, y, th = state
        
        corners = np.array([
            [-self.length/2, -self.width/2],
            [self.length/2, -self.width/2],
            [self.length/2, self.width/2],
            [-self.length/2, self.width/2]
        ])
        
        R = np.array([
            [np.cos(th), -np.sin(th)],
            [np.sin(th), np.cos(th)]
        ])
        
        corners = (R @ corners.T).T + np.array([x, y])
        
        poly = plt.Polygon(corners, fill=True, fc=color, ec='darkgreen', alpha=alpha)
        ax.add_patch(poly)
        
        return ax


if __name__ == "__main__":
    
    veh = Vehicle(v_min=0.1, v_max=2.0)
    state = np.array([0.0, 0.0, 0.0])
    u = np.array([1.0, 0.1])
    
    print("Test dynamics:")
    for i in range(5):
        state = veh.step(state, u, dt=0.1)
        print(f"  {i+1}: x={state[0]:.3f}, y={state[1]:.3f}, th={state[2]:.3f}")