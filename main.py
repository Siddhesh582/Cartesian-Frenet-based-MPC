"""
MPC path tracking with obstacle avoidance demo.
"""

import numpy as np
import time
import os

from src.path import Path, create_obstacles
from src.mpc import MPC, Config
from utils.simulation import simulate, plot_results, animate
from utils.metrics import compute, print_report, save, summary_for_readme


def main():
    
    print("="*60)
    print("         MPC PATH TRACKING DEMONSTRATION")
    print("="*60)
    
    # path
    path = Path(
        L1=30.0,
        R1=15.0,
        angle=90.0,
        L2=25.0,
        R2=6.0,
        L3=15.0,
        width=5.0
    )
    
    obstacles = create_obstacles(path)
    
    print(f"\nPath: {path.length:.1f} m")
    for seg in path.segments:
        print(f"  {seg.type}: {seg.s1 - seg.s0:.1f} m")
    print(f"  Lane width: {path.width} m")
    print(f"  Obstacles: {len(obstacles)}")
    
    # controller
    cfg = Config(N=50, dt=0.1, v_max=1.5)
    mpc = MPC(path, r_vehicle=0.5, config=cfg)
    mpc.set_obstacles(obstacles)
    
    print(f"\nMPC: {cfg.N} steps ({cfg.N * cfg.dt}s horizon)")
    
    # simulate
    t0 = time.time()
    states, controls, infos, _ = simulate(mpc, path, obstacles, cfg.dt, max_steps=2000)
    elapsed = time.time() - t0
    
    print(f"\nCompleted in {elapsed:.1f}s (realtime factor: {len(controls)*cfg.dt/elapsed:.1f}x)")
    
    # metrics
    os.makedirs('results', exist_ok=True)
    m = compute(states, controls, infos, path, obstacles, 0.5, cfg.dt)
    print_report(m)
    save(m, 'results/metrics.json')
    print(f"\nSummary: {summary_for_readme(m)}")
    
    # plot
    plot_results(path, states, controls, obstacles, infos, save_path='results/trajectory.png')
    
    # animation
    animate(path, states, controls, obstacles, infos, save_path='results/animation.mp4', speedup=1.5)


if __name__ == "__main__":
    main()