"""
MPC controller for path tracking with obstacle avoidance.
"""

import numpy as np
import casadi as ca
from dataclasses import dataclass
import time


@dataclass
class Config:
    N: int = 50
    dt: float = 0.1
    
    v_min: float = 0.3
    v_max: float = 1.5
    w_min: float = -np.pi / 2
    w_max: float = np.pi / 2
    
    # acceleration limits
    a_max: float = 0.3        # m/s^2 - gentle acceleration
    a_min: float = -0.5       # m/s^2 - moderate braking
    
    # cost weights
    q_lat: float = 40.0
    q_head: float = 25.0
    q_vel: float = 10.0
    q_prog: float = -5.0
    r_v: float = 0.5
    r_w: float = 2.0
    r_dv: float = 500.0       # very high penalty for velocity changes
    r_dw: float = 10.0
    q_obs: float = 400.0
    q_lane: float = 150.0
    
    obs_margin: float = 0.4
    lane_margin: float = 0.1


class MPC:
    
    def __init__(self, path, r_vehicle=0.5, config=None):
        self.path = path
        self.r_veh = r_vehicle
        self.cfg = config or Config()
        self.obstacles = []
        self.lane_w = getattr(path, 'width', 4.0)
        
        self._build()
        self._last_sol = None
        self.times = []
        self.status = []
    
    def _build(self):
        cfg = self.cfg
        N = cfg.N
        
        self.X = ca.MX.sym('X', 5, N + 1)
        self.U = ca.MX.sym('U', 2, N)
        
        max_obs = 10
        n_path = 2 * (N + 1)
        self.P = ca.MX.sym('P', 7 + n_path + 3 * max_obs)
        self.max_obs = max_obs
        self.n_path = n_path
        
        g = []
        lb_g = []
        ub_g = []
        
        # initial state
        g.append(self.X[:, 0] - self.P[:5])
        lb_g += [0]*5
        ub_g += [0]*5
        
        # dynamics
        for k in range(N):
            xk = self.X[:, k]
            uk = self.U[:, k]
            xn = self.X[:, k + 1]
            
            v, w = uk[0], uk[1]
            th = xk[2]
            sk, nk = xk[3], xk[4]
            
            psi = self.P[7 + k]
            kap = self.P[7 + N + 1 + k]
            th_e = th - psi
            
            s_dot = v * ca.cos(th_e) / (1 - nk * kap + 1e-6)
            n_dot = v * ca.sin(th_e)
            
            x_dot = v * ca.cos(th)
            y_dot = v * ca.sin(th)
            th_dot = w
            
            xn_pred = ca.vertcat(
                xk[0] + cfg.dt * x_dot,
                xk[1] + cfg.dt * y_dot,
                xk[2] + cfg.dt * th_dot,
                xk[3] + cfg.dt * s_dot,
                xk[4] + cfg.dt * n_dot
            )
            
            g.append(xn - xn_pred)
            lb_g += [0]*5
            ub_g += [0]*5
        
        # cost
        J = 0
        v_prev = self.P[5]
        w_prev = self.P[6]
        
        for k in range(N):
            xk = self.X[:, k]
            uk = self.U[:, k]
            
            nk = xk[4]
            thk = xk[2]
            psi = self.P[7 + k]
            th_e = ca.atan2(ca.sin(thk - psi), ca.cos(thk - psi))
            
            J += cfg.q_lat * nk**2
            J += cfg.q_head * th_e**2
            
            if k > 0:
                J += cfg.q_prog * (xk[3] - self.X[3, k-1])
            
            kap = self.P[7 + N + 1 + k]
            v_ref = cfg.v_max * ca.fmax(0.3, 1 - 2.0 * ca.fabs(kap))
            J += cfg.q_vel * (uk[0] - v_ref)**2
            
            J += cfg.r_v * uk[0]**2
            J += cfg.r_w * uk[1]**2
            
            if k == 0:
                dv = uk[0] - v_prev
                dw = uk[1] - w_prev
            else:
                dv = uk[0] - self.U[0, k-1]
                dw = uk[1] - self.U[1, k-1]
            J += cfg.r_dv * dv**2
            J += cfg.r_dw * dw**2
            
            half_lane = self.lane_w / 2 - cfg.lane_margin
            lane_viol = ca.fmax(0, ca.fabs(nk) - half_lane)
            J += cfg.q_lane * lane_viol**2
            
            obs_start = 7 + self.n_path
            for i in range(self.max_obs):
                ox = self.P[obs_start + 3*i]
                oy = self.P[obs_start + 3*i + 1]
                orad = self.P[obs_start + 3*i + 2]
                
                d2 = (xk[0] - ox)**2 + (xk[1] - oy)**2
                d = ca.sqrt(d2 + 1e-6)
                clearance = self.r_veh + orad + cfg.obs_margin
                
                pen = ca.fmax(0, clearance - d)
                J += cfg.q_obs * pen**2
        
        # terminal cost
        xN = self.X[:, N]
        psi_N = self.P[7 + N]
        th_e_N = ca.atan2(ca.sin(xN[2] - psi_N), ca.cos(xN[2] - psi_N))
        J += cfg.q_lat * 3 * xN[4]**2
        J += cfg.q_head * 2 * th_e_N**2
        
        # control bounds
        for k in range(N):
            g.append(self.U[0, k])
            lb_g.append(cfg.v_min)
            ub_g.append(cfg.v_max)
            
            g.append(self.U[1, k])
            lb_g.append(cfg.w_min)
            ub_g.append(cfg.w_max)
        
        # acceleration constraints
        for k in range(N):
            if k == 0:
                dv = (self.U[0, k] - v_prev) / cfg.dt
            else:
                dv = (self.U[0, k] - self.U[0, k-1]) / cfg.dt
            g.append(dv)
            lb_g.append(cfg.a_min)
            ub_g.append(cfg.a_max)
        
        # obstacle constraints
        obs_start = 7 + self.n_path
        for k in range(0, N + 1, 2):
            xk = self.X[:, k]
            for i in range(self.max_obs):
                ox = self.P[obs_start + 3*i]
                oy = self.P[obs_start + 3*i + 1]
                orad = self.P[obs_start + 3*i + 2]
                
                d2 = (xk[0] - ox)**2 + (xk[1] - oy)**2
                min_d = self.r_veh + orad + 0.15
                
                slack = 1000 * (0.01 - ca.fmin(orad, 0.01)) / 0.01
                g.append(d2 - min_d**2 + slack)
                lb_g.append(0)
                ub_g.append(np.inf)
        
        # build solver
        z = ca.vertcat(ca.reshape(self.X, -1, 1), ca.reshape(self.U, -1, 1))
        
        nlp = {'x': z, 'f': J, 'g': ca.vertcat(*g), 'p': self.P}
        opts = {
            'ipopt.print_level': 0,
            'ipopt.max_iter': 150,
            'ipopt.tol': 1e-4,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.sb': 'yes',
            'print_time': 0
        }
        
        self.solver = ca.nlpsol('mpc', 'ipopt', nlp, opts)
        self.nz_x = 5 * (N + 1)
        self.nz_u = 2 * N
        self.lb_g = lb_g
        self.ub_g = ub_g
    
    def set_obstacles(self, obs):
        self.obstacles = obs
    
    def solve(self, state, u_prev=None):
        cfg = self.cfg
        N = cfg.N
        
        if u_prev is None:
            u_prev = np.array([cfg.v_max * 0.5, 0.0])
        
        # parameters
        np_total = 7 + self.n_path + 3 * self.max_obs
        p = np.zeros(np_total)
        p[:5] = state
        p[5:7] = u_prev
        
        s_cur = state[3]
        for k in range(N + 1):
            sk = min(s_cur + k * cfg.dt * cfg.v_max, self.path.length - 0.1)
            p[7 + k] = self.path.heading(sk)
            p[7 + N + 1 + k] = self.path.curvature(sk)
        
        obs_start = 7 + self.n_path
        for i, ob in enumerate(self.obstacles[:self.max_obs]):
            p[obs_start + 3*i] = ob['center'][0]
            p[obs_start + 3*i + 1] = ob['center'][1]
            p[obs_start + 3*i + 2] = ob['radius']
        
        # initial guess
        if self._last_sol is not None:
            z0 = self._last_sol
        else:
            z0 = np.zeros(self.nz_x + self.nz_u)
            xi = state.copy()
            for k in range(N + 1):
                z0[5*k:5*k+5] = xi
                if k < N:
                    z0[self.nz_x + 2*k] = cfg.v_max * 0.5
                    z0[self.nz_x + 2*k + 1] = 0.0
                xi[0] += cfg.dt * cfg.v_max * 0.5 * np.cos(xi[2])
                xi[1] += cfg.dt * cfg.v_max * 0.5 * np.sin(xi[2])
                xi[3] += cfg.dt * cfg.v_max * 0.5
        
        # bounds
        lbz = np.full(self.nz_x + self.nz_u, -np.inf)
        ubz = np.full(self.nz_x + self.nz_u, np.inf)
        
        for k in range(N + 1):
            lbz[5*k + 3] = 0
        
        for k in range(N):
            lbz[self.nz_x + 2*k] = cfg.v_min
            ubz[self.nz_x + 2*k] = cfg.v_max
            lbz[self.nz_x + 2*k + 1] = cfg.w_min
            ubz[self.nz_x + 2*k + 1] = cfg.w_max
        
        # solve
        t0 = time.time()
        
        try:
            sol = self.solver(x0=z0, p=p, lbx=lbz, ubx=ubz, lbg=self.lb_g, ubg=self.ub_g)
            
            dt_solve = time.time() - t0
            self.times.append(dt_solve)
            
            z_opt = np.array(sol['x']).flatten()
            self._last_sol = z_opt
            
            u = z_opt[self.nz_x:self.nz_x + 2]
            X_pred = z_opt[:self.nz_x].reshape(N + 1, 5)
            U_pred = z_opt[self.nz_x:].reshape(N, 2)
            
            stat = self.solver.stats()['return_status']
            self.status.append(stat)
            
            info = {
                'solve_time': dt_solve,
                'status': stat,
                'cost': float(sol['f']),
                'X_pred': X_pred,
                'U_pred': U_pred
            }
            
        except Exception as e:
            print(f"Solver failed: {e}")
            u = np.array([cfg.v_min, 0.0])
            info = {'solve_time': 0, 'status': 'failed', 'cost': np.inf}
        
        return u, info