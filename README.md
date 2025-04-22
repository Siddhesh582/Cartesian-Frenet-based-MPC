# Cartesian-Frenet-based-MPC 🚗

This project presents the implementation of a Frenet–Cartesian NMPC framework inspired by the paper  
**"Model Predictive Control for Frenet-Cartesian Trajectory Tracking of a Tricycle Kinematic Automated Guided Vehicle"** ([IEEE Xplore](https://doi.org/10.1109/ICRA.2023.10802822)).

Instead of a tricycle model, this implementation uses a **unicycle kinematic model**, and focuses on achieving accurate trajectory tracking and collision avoidance in structured environments with static obstacles.4

## 🚀 Project Features

- **Frenet Frame-Based Trajectory Tracking:**  
  Path progress `s`, lateral deviation `n`, and heading error `β` are modeled in the Frenet frame for intuitive tracking.

- **Cartesian-Based Obstacle Avoidance:**  
  Collision constraints with circular obstacles are enforced in the Cartesian frame for geometric clarity.

- **Lifted State NMPC Formulation:**  
  Combines both coordinate frames in an extended state vector to balance spatial path-following and safety constraints.

- **Soft Constraints via Slack Variables:**  
  Curvature, lateral deviation, and obstacle distances are enforced using soft constraints to ensure feasible solutions.

- **Simulation and Visualization:**  
  Includes plotting and animation of trajectories, control profiles, heading error, solver time, and vehicle motion around obstacles.

---

## 🛠 Technologies Used

- **Python 3**
- **CasADi** — for symbolic computation and nonlinear optimization
- **IPOPT** — solver used for NMPC
- **Matplotlib** — for plotting and animation
- **NumPy**

---