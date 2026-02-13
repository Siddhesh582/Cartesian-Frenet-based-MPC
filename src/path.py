"""
Reference path generation: straight -> curve -> straight -> u-turn -> straight
"""

import numpy as np
from dataclasses import dataclass


@dataclass 
class Segment:
    s0: float
    s1: float
    type: str


class Path:
    
    def __init__(self, L1=30.0, R1=20.0, angle=90.0, L2=30.0, R2=8.0, L3=20.0,
                 width=4.0, pts_per_m=10):
        
        self.width = width
        self._build(L1, R1, angle, L2, R2, L3, pts_per_m)
    
    def _build(self, L1, R1, angle_deg, L2, R2, L3, ppm):
        
        angle = np.radians(angle_deg)
        
        x_all, y_all, h_all, k_all, s_all = [], [], [], [], []
        segments = []
        s = 0.0
        
        # segment 1: straight
        n1 = int(L1 * ppm)
        s0 = s
        for i in range(n1):
            t = i / (n1 - 1)
            x_all.append(t * L1)
            y_all.append(0.0)
            h_all.append(0.0)
            k_all.append(0.0)
            s_all.append(s)
            if i < n1 - 1:
                s += L1 / (n1 - 1)
        segments.append(Segment(s0, s, 'straight'))
        
        # segment 2: curve
        cx, cy = L1, R1
        arc = R1 * angle
        n2 = int(arc * ppm)
        s0 = s
        for i in range(n2):
            t = i / (n2 - 1)
            th = -np.pi/2 + t * angle
            x_all.append(cx + R1 * np.cos(th))
            y_all.append(cy + R1 * np.sin(th))
            h_all.append(th + np.pi/2)
            k_all.append(1.0 / R1)
            s_all.append(s)
            if i < n2 - 1:
                s += arc / (n2 - 1)
        segments.append(Segment(s0, s, 'curve'))
        
        end_th = -np.pi/2 + angle
        cx_end = cx + R1 * np.cos(end_th)
        cy_end = cy + R1 * np.sin(end_th)
        h_end = end_th + np.pi/2
        
        # segment 3: straight
        n3 = int(L2 * ppm)
        s0 = s
        dx, dy = np.cos(h_end), np.sin(h_end)
        for i in range(n3):
            t = i / (n3 - 1)
            x_all.append(cx_end + t * L2 * dx)
            y_all.append(cy_end + t * L2 * dy)
            h_all.append(h_end)
            k_all.append(0.0)
            s_all.append(s)
            if i < n3 - 1:
                s += L2 / (n3 - 1)
        segments.append(Segment(s0, s, 'straight'))
        
        ux, uy, uh = x_all[-1], y_all[-1], h_end
        
        # segment 4: u-turn
        ucx = ux + R2 * np.cos(uh + np.pi/2)
        ucy = uy + R2 * np.sin(uh + np.pi/2)
        uarc = np.pi
        ulen = R2 * uarc
        n4 = int(ulen * ppm)
        s0 = s
        start_ang = np.arctan2(uy - ucy, ux - ucx)
        for i in range(n4):
            t = i / (n4 - 1)
            th = start_ang + t * uarc
            x_all.append(ucx + R2 * np.cos(th))
            y_all.append(ucy + R2 * np.sin(th))
            h_all.append(th + np.pi/2)
            k_all.append(1.0 / R2)
            s_all.append(s)
            if i < n4 - 1:
                s += ulen / (n4 - 1)
        segments.append(Segment(s0, s, 'uturn'))
        
        ex, ey, eh = x_all[-1], y_all[-1], h_all[-1]
        
        # segment 5: straight
        n5 = int(L3 * ppm)
        s0 = s
        dx, dy = np.cos(eh), np.sin(eh)
        for i in range(n5):
            t = i / (n5 - 1)
            x_all.append(ex + t * L3 * dx)
            y_all.append(ey + t * L3 * dy)
            h_all.append(eh)
            k_all.append(0.0)
            s_all.append(s)
            if i < n5 - 1:
                s += L3 / (n5 - 1)
        segments.append(Segment(s0, s, 'straight'))
        
        self.x = np.array(x_all)
        self.y = np.array(y_all)
        self.h = np.array(h_all)
        self.k = np.array(k_all)
        self.s = np.array(s_all)
        self.segments = segments
        self.length = self.s[-1]
    
    def gamma(self, s):
        s = np.clip(s, 0, self.length)
        x = np.interp(s, self.s, self.x)
        y = np.interp(s, self.s, self.y)
        return np.array([x, y])
    
    def heading(self, s):
        s = np.clip(s, 0, self.length)
        return float(np.interp(s, self.s, self.h))
    
    def curvature(self, s):
        s = np.clip(s, 0, self.length)
        return float(np.interp(s, self.s, self.k))
    
    def get_segment(self, s):
        for seg in self.segments:
            if seg.s0 <= s <= seg.s1:
                return seg.type
        return 'unknown'


def create_obstacles(path):
    
    obs = []
    
    # obstacle 1: first straight, on centerline
    s1 = 10.0
    p1 = path.gamma(s1)
    obs.append({'center': (p1[0], p1[1]), 'radius': 0.7})
    
    # obstacle 2: first straight, offset left
    s2 = 22.0
    p2 = path.gamma(s2)
    h2 = path.heading(s2)
    n2 = np.array([-np.sin(h2), np.cos(h2)])
    obs.append({'center': (p2[0] + 0.4*n2[0], p2[1] + 0.4*n2[1]), 'radius': 0.5})
    
    # obstacle 3: curve entry
    s3 = path.segments[1].s0 + 3.0
    p3 = path.gamma(s3)
    obs.append({'center': (p3[0], p3[1]), 'radius': 0.55})
    
    # obstacle 4: mid curve
    s4 = (path.segments[1].s0 + path.segments[1].s1) / 2
    p4 = path.gamma(s4)
    obs.append({'center': (p4[0], p4[1]), 'radius': 0.5})
    
    # obstacle 5: second straight
    s5 = path.segments[2].s0 + 10.0
    p5 = path.gamma(s5)
    obs.append({'center': (p5[0], p5[1]), 'radius': 0.65})
    
    # obstacle 6: u-turn entry
    s6 = path.segments[3].s0 + 2.0
    p6 = path.gamma(s6)
    h6 = path.heading(s6)
    n6 = np.array([-np.sin(h6), np.cos(h6)])
    obs.append({'center': (p6[0] - 0.3*n6[0], p6[1] - 0.3*n6[1]), 'radius': 0.45})
    
    # obstacle 7: final straight
    s7 = path.segments[4].s0 + 5.0
    p7 = path.gamma(s7)
    obs.append({'center': (p7[0], p7[1]), 'radius': 0.5})
    
    return obs


if __name__ == "__main__":
    
    path = Path(L1=30.0, R1=15.0, angle=90.0, L2=25.0, R2=6.0, L3=15.0, width=5.0)
    
    print(f"Path length: {path.length:.1f} m")
    for seg in path.segments:
        print(f"  {seg.type}: {seg.s1 - seg.s0:.1f} m")
    
    obs = create_obstacles(path)
    print(f"Obstacles: {len(obs)}")