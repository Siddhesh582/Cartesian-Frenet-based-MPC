# Frenet-Cartesian Model Predictive Control for Autonomous Path Tracking

Nonlinear MPC framework for unicycle-type vehicles using hybrid Frenet-Cartesian coordinates with real-time obstacle avoidance.

![Path Tracking Demo](docs/images/final_trajectory.png)

## Overview

This project implements a **Nonlinear Model Predictive Controller (NMPC)** that simultaneously tracks a reference path and avoids obstacles using a 5-state hybrid coordinate system. The controller uses **CasADi with IPOPT** for real-time nonlinear optimization.

### Key Results

| Metric | Value |
|--------|-------|
| **RMS Lateral Tracking Error** | 0.18 m |
| **Path Completion Rate** | 98.5% |
| **Collision-Free Navigation** | 100% (6 obstacles) |
| **Mean Solver Time** | 15.2 ms @ 10 Hz |
| **Prediction Horizon** | 4.5 s (45 steps) |
| **Real-Time Factor** | 6.5× faster than real-time |

## Technical Highlights

- **5-DOF State Space**: Combined Cartesian $(x, y, \theta)$ + Frenet $(s, n)$ for robust tracking
- **45-Step Prediction Horizon**: 4.5-second lookahead for proactive obstacle avoidance
- **Soft Constraint Formulation**: Slack variables ensure solver feasibility in tight scenarios
- **RK4 Integration**: Fourth-order accuracy for dynamics propagation within MPC
- **10 Hz Control Rate**: Real-time capable on standard hardware

## Mathematical Formulation

### State-Space Model

**Augmented State Vector:**
$$\mathbf{x} = [x, y, \theta, s, n]^T \in \mathbb{R}^5$$

**Control Inputs:**
$$\mathbf{u} = [v, \omega]^T, \quad v \in [0.1, 2.0] \text{ m/s}, \quad \omega \in [-\pi/2, \pi/2] \text{ rad/s}$$

### Frenet Frame Dynamics

$$\dot{s} = \frac{v \cos(\beta)}{1 - n\kappa(s)}, \quad \dot{n} = v \sin(\beta), \quad \dot{\beta} = \omega - \kappa(s)\dot{s}$$

Where:
- $s$: Arc-length progress along reference path
- $n$: Signed lateral deviation from centerline
- $\beta = \theta - \theta_{path}(s)$: Heading error relative to path tangent
- $\kappa(s)$: Path curvature (from cubic spline)

### Cost Function

$$J = \sum_{k=0}^{N-1} \underbrace{-Q_s \Delta s_k}_{\text{progress}} + \underbrace{Q_n n_k^2}_{\text{tracking}} + \underbrace{Q_\theta \sin^2(\beta_k)}_{\text{heading}} + \underbrace{R_v(v_k - v_{ref})^2 + R_\omega \omega_k^2}_{\text{control effort}} + \underbrace{\sum_i \rho_i \xi_{i,k}}_{\text{slack penalties}}$$

**Weight Parameters:**
| Weight | Value | Purpose |
|--------|-------|---------|
| $Q_s$ | 20.0 | Progress maximization |
| $Q_n$ | 25.0 | Lateral error minimization |
| $Q_\theta$ | 5.0 | Heading alignment |
| $R_v$ | 1.0 | Velocity regulation |
| $R_\omega$ | 10.0 | Steering smoothness |

### Obstacle Avoidance Constraints

For each obstacle $j$ with center $(x_j^{obs}, y_j^{obs})$ and radius $r_j$:

$$\sqrt{(x_k - x_j^{obs})^2 + (y_k - y_j^{obs})^2} \geq r_{vehicle} + r_j + \delta_{safety} - \xi_k^{obs}$$

Where $\xi_k^{obs} \geq 0$ is a slack variable with penalty weight $\rho = 1000$.

## Project Structure

```
Cartesian-Frenet-based-MPC/
├── README.md
├── requirements.txt
│
├── Unicycle/
│   ├── main.py                 # Simulation runner with metrics
│   ├── Controller.py           # NMPC (CasADi/IPOPT)
│   ├── Unicycle.py             # Vehicle dynamics & collision model
│   ├── FrenetParameters.py     # Reference path & Frenet utilities
│   ├── simulation.py           # Control loop & visualization
│   ├── metrics.py              # Performance metric computation
│   │
│   ├── results/                # Generated outputs
│   │   ├── mpc_metrics.json
│   │   ├── final_trajectory.png
│   │   └── metrics_analysis.png
│   │
│   └── docs/
│       ├── images/
│       └── Report.pdf
│
└── Tricycle/
    └── [Tricycle model - Ackermann steering]
```

## Installation

```bash
# Clone repository
git clone https://github.com/Siddhesh582/Cartesian-Frenet-based-MPC.git
cd Cartesian-Frenet-based-MPC

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥1.21 | Array operations |
| `scipy` | ≥1.7 | Cubic spline interpolation |
| `matplotlib` | ≥3.5 | Visualization |
| `casadi` | ≥3.5.5 | NLP modeling & IPOPT interface |
| `gymnasium` | ≥0.28 | Action space definitions |

## Usage

### Run Simulation

```bash
cd Unicycle
python main.py
```

### Configuration

Edit parameters in `main.py`:

```python
# Path configuration
path_length = 100      # meters
lane_width = 15.0      # meters

# MPC tuning
mpc_controller.N = 45           # Prediction horizon
mpc_controller.dt = 0.1         # Sampling time (10 Hz)
mpc_controller.Q_n = 25.0       # Lateral tracking weight
mpc_controller.v_ref = 0.8      # Reference velocity

# Obstacles
obstacles = [
    {'center': (15, 2.9), 'radius': 2.0},
    {'center': (30, -1.3), 'radius': 1.8},
    # ... add more
]
```

## Module Documentation

### `Controller.py` — FrenetCartesianMPC

Core NMPC implementation using CasADi's Opti stack:

- **Optimization Variables**: States $X \in \mathbb{R}^{5 \times (N+1)}$, Controls $U \in \mathbb{R}^{2 \times N}$
- **Solver**: IPOPT with analytical gradients (CasADi automatic differentiation)
- **Warm-Starting**: Previous solution used as initial guess
- **Constraint Types**: Hard (velocity bounds), Soft (obstacles, lateral limits)

### `Unicycle.py` — Vehicle Model

Unicycle kinematics with dual-coordinate state:

- `step()`: Cartesian propagation (Euler integration)
- `step_frenet()`: Frenet propagation (RK4 integration)
- `cartesian_to_frenet()`: Coordinate transformation via path projection
- `check_collision()`: Circle-based collision detection

### `FrenetParameters.py` — Reference Path

Path representation and Frenet frame utilities:

- **Cubic Spline Interpolation**: Smooth $C^2$-continuous path
- **Precomputed Curvature/Heading**: Vectorized lookup for MPC efficiency
- **Path Projection**: Finds closest point $s^*$ for arbitrary $(x, y)$

### `metrics.py` — Performance Evaluation

Comprehensive metric computation:

- Tracking: RMS/Max lateral error, heading error
- Progress: Distance traveled, completion percentage
- Control: Smoothness (std of control changes), effort integrals
- Safety: Min clearance, near-miss count, collision detection
- Computation: Solver time statistics, success rate

## Performance Benchmarks

Tested on: Intel i7-12700H, 16GB RAM, Ubuntu 22.04

| Scenario | Steps | Mean Solve Time | Max Solve Time | Tracking RMS |
|----------|-------|-----------------|----------------|--------------|
| No obstacles | 600 | 8.3 ms | 45 ms | 0.12 m |
| 6 obstacles | 600 | 15.2 ms | 127 ms | 0.18 m |
| Dense (10 obs) | 600 | 23.8 ms | 312 ms | 0.24 m |

## Results

### Tracking Performance
- **Mean lateral error**: 0.15 m over 100m path
- **Max lateral error**: 1.8 m (during tight obstacle avoidance)
- **Heading alignment**: < 5° mean error

### Control Quality
- **Velocity tracking**: Maintains 0.8 m/s ± 0.1 m/s reference
- **Steering smoothness**: σ(Δω) = 0.08 rad/s (no oscillations)
- **Constraint satisfaction**: Zero violations with soft constraints

### Obstacle Avoidance
- **Minimum clearance**: 0.31 m (with 0.05 m safety margin)
- **100% collision-free** across all test scenarios

## References

1. Werling, M., Ziegler, J., Kammel, S., & Thrun, S. (2010). *Optimal trajectory generation for dynamic street scenarios in a Frenét Frame*. ICRA.
2. Kong, J., Pfeiffer, M., Schildbach, G., & Borrelli, F. (2015). *Kinematic and dynamic vehicle models for autonomous driving control design*. IV.
3. Andersson, J. A., et al. (2019). *CasADi: A software framework for nonlinear optimization and optimal control*. Mathematical Programming Computation.

## License

MIT License

## Author

**Siddhesh Shingate**  
M.S. Robotics Engineering, Northeastern University  
[GitHub](https://github.com/Siddhesh582) • [LinkedIn](https://linkedin.com/in/siddhesh-shingate)