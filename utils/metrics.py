"""
Performance metrics for MPC evaluation.
"""

import numpy as np
import json
import os


def compute(states, controls, infos, path, obstacles, r_veh, dt):
    
    n = states[:, 4]
    v = controls[:, 0]
    w = controls[:, 1]
    
    # acceleration and jerk
    accel = np.diff(v) / dt
    jerk = np.diff(accel) / dt
    
    # heading error
    h_err = []
    for st in states:
        psi = path.heading(st[3])
        e = (st[2] - psi + np.pi) % (2*np.pi) - np.pi
        h_err.append(abs(e))
    h_err = np.array(h_err)
    
    # obstacle clearance
    clearances = []
    collisions = 0
    for st in states:
        for obs in obstacles:
            d = np.sqrt((st[0] - obs['center'][0])**2 + (st[1] - obs['center'][1])**2)
            c = d - obs['radius'] - r_veh
            clearances.append(c)
            if c < 0:
                collisions += 1
    clearances = np.array(clearances)
    
    # solve times
    times = np.array([info['solve_time'] for info in infos]) * 1000
    success = sum(1 for info in infos if 'Solve' in info.get('status', ''))
    
    metrics = {
        
        # tracking performance
        'tracking': {
            'lateral_rms_cm': float(np.sqrt(np.mean(n**2)) * 100),
            'lateral_max_cm': float(np.max(np.abs(n)) * 100),
            'lateral_mean_cm': float(np.mean(np.abs(n)) * 100),
            'heading_rms_deg': float(np.sqrt(np.mean(h_err**2)) * 180/np.pi),
            'heading_max_deg': float(np.max(h_err) * 180/np.pi),
        },
        
        # safety metrics
        'safety': {
            'collision_free': collisions == 0,
            'num_collisions': collisions,
            'min_clearance_cm': float(np.min(clearances) * 100),
            'mean_clearance_cm': float(np.mean(clearances) * 100),
            'clearance_std_cm': float(np.std(clearances) * 100),
        },
        
        # control quality
        'control': {
            'velocity_mean_mps': float(np.mean(v)),
            'velocity_std_mps': float(np.std(v)),
            'accel_max_mps2': float(np.max(np.abs(accel))),
            'accel_rms_mps2': float(np.sqrt(np.mean(accel**2))),
            'jerk_max_mps3': float(np.max(np.abs(jerk))),
            'jerk_rms_mps3': float(np.sqrt(np.mean(jerk**2))),
            'steering_max_dps': float(np.max(np.abs(w)) * 180/np.pi),
            'steering_rms_dps': float(np.sqrt(np.mean(w**2)) * 180/np.pi),
        },
        
        # computational performance
        'computation': {
            'solve_mean_ms': float(np.mean(times)),
            'solve_max_ms': float(np.max(times)),
            'solve_std_ms': float(np.std(times)),
            'solve_p95_ms': float(np.percentile(times, 95)),
            'solve_p99_ms': float(np.percentile(times, 99)),
            'success_rate_pct': float(success / len(infos) * 100) if infos else 0,
            'realtime_margin_pct': float((dt*1000 - np.percentile(times, 95)) / (dt*1000) * 100),
        },
        
        # task completion
        'completion': {
            'distance_m': float(states[-1, 3]),
            'path_length_m': float(path.length),
            'completion_pct': float(states[-1, 3] / path.length * 100),
            'duration_s': float(len(controls) * dt),
            'avg_velocity_mps': float(states[-1, 3] / (len(controls) * dt)) if len(controls) > 0 else 0,
        },
        
        # constraint satisfaction
        'constraints': {
            'velocity_violations': int(np.sum((v < 0.09) | (v > 2.01))),
            'lane_violations': int(np.sum(np.abs(n) > path.width/2)),
            'all_satisfied': bool(np.all(np.abs(n) <= path.width/2) and collisions == 0),
        },
    }
    
    return metrics


def print_report(m):
    
    print("\n" + "="*65)
    print("                  MPC PERFORMANCE METRICS")
    print("="*65)
    
    t = m['tracking']
    print(f"\n[TRACKING ACCURACY]")
    print(f"  Lateral error (RMS):     {t['lateral_rms_cm']:6.2f} cm")
    print(f"  Lateral error (max):     {t['lateral_max_cm']:6.2f} cm")
    print(f"  Heading error (RMS):     {t['heading_rms_deg']:6.2f} deg")
    print(f"  Heading error (max):     {t['heading_max_deg']:6.2f} deg")
    
    s = m['safety']
    print(f"\n[SAFETY]")
    print(f"  Collision-free:          {'Yes' if s['collision_free'] else 'No'}")
    print(f"  Min clearance:           {s['min_clearance_cm']:6.1f} cm")
    print(f"  Mean clearance:          {s['mean_clearance_cm']:6.1f} cm")
    
    c = m['control']
    print(f"\n[CONTROL SMOOTHNESS]")
    print(f"  Velocity (mean):         {c['velocity_mean_mps']:6.2f} m/s")
    print(f"  Acceleration (max):      {c['accel_max_mps2']:6.2f} m/s2")
    print(f"  Acceleration (RMS):      {c['accel_rms_mps2']:6.2f} m/s2")
    print(f"  Jerk (max):              {c['jerk_max_mps3']:6.2f} m/s3")
    print(f"  Jerk (RMS):              {c['jerk_rms_mps3']:6.2f} m/s3")
    print(f"  Steering rate (max):     {c['steering_max_dps']:6.1f} deg/s")
    
    cp = m['computation']
    print(f"\n[COMPUTATIONAL PERFORMANCE]")
    print(f"  Solve time (mean):       {cp['solve_mean_ms']:6.1f} ms")
    print(f"  Solve time (max):        {cp['solve_max_ms']:6.1f} ms")
    print(f"  Solve time (95th pct):   {cp['solve_p95_ms']:6.1f} ms")
    print(f"  Solve time (99th pct):   {cp['solve_p99_ms']:6.1f} ms")
    print(f"  Success rate:            {cp['success_rate_pct']:6.1f} %")
    print(f"  Realtime margin:         {cp['realtime_margin_pct']:6.1f} %")
    
    cm = m['completion']
    print(f"\n[TASK COMPLETION]")
    print(f"  Path completed:          {cm['completion_pct']:6.1f} %")
    print(f"  Distance traveled:       {cm['distance_m']:6.1f} m")
    print(f"  Duration:                {cm['duration_s']:6.1f} s")
    print(f"  Average velocity:        {cm['avg_velocity_mps']:6.2f} m/s")
    
    cs = m['constraints']
    print(f"\n[CONSTRAINT SATISFACTION]")
    print(f"  All constraints met:     {'Yes' if cs['all_satisfied'] else 'No'}")
    print(f"  Lane violations:         {cs['lane_violations']}")
    print(f"  Velocity violations:     {cs['velocity_violations']}")
    
    print("\n" + "="*65)


def save(metrics, path='results/metrics.json'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {path}")


def summary_for_readme(m):
    """One-liner metrics for README or resume."""
    
    return (
        f"Tracking: {m['tracking']['lateral_rms_cm']:.1f}cm RMS | "
        f"Safety: {m['safety']['min_clearance_cm']:.0f}cm clearance | "
        f"Compute: {m['computation']['solve_p95_ms']:.0f}ms @ 95th pct | "
        f"Completion: {m['completion']['completion_pct']:.0f}%"
    )


def export_latex_table(m, path='results/metrics_table.tex'):
    """Export metrics as LaTeX table for papers."""
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    latex = r"""
\begin{table}[h]
\centering
\caption{MPC Performance Metrics}
\begin{tabular}{lcc}
\hline
\textbf{Metric} & \textbf{Value} & \textbf{Unit} \\
\hline
Lateral Error (RMS) & %.2f & cm \\
Lateral Error (Max) & %.2f & cm \\
Heading Error (RMS) & %.2f & deg \\
Min Obstacle Clearance & %.1f & cm \\
Acceleration (RMS) & %.2f & m/s$^2$ \\
Jerk (RMS) & %.2f & m/s$^3$ \\
Solve Time (Mean) & %.1f & ms \\
Solve Time (95th pct) & %.1f & ms \\
Realtime Margin & %.1f & \%% \\
Path Completion & %.1f & \%% \\
\hline
\end{tabular}
\end{table}
""" % (
        m['tracking']['lateral_rms_cm'],
        m['tracking']['lateral_max_cm'],
        m['tracking']['heading_rms_deg'],
        m['safety']['min_clearance_cm'],
        m['control']['accel_rms_mps2'],
        m['control']['jerk_rms_mps3'],
        m['computation']['solve_mean_ms'],
        m['computation']['solve_p95_ms'],
        m['computation']['realtime_margin_pct'],
        m['completion']['completion_pct'],
    )
    
    with open(path, 'w') as f:
        f.write(latex.strip())
    
    print(f"Saved LaTeX table to {path}")