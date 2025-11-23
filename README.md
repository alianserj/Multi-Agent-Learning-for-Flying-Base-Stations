# Multi-Agent Learning for Flying Base Stations

## Abstract
This project optimizes the deployment of Unmanned Aerial Vehicles (UAVs) to maximize wireless coverage in dynamic environments. We formulate the problem as a **Potential Game**, where UAVs optimize a global utility function (total users served) by maximizing their own **Marginal Contribution Utility (MCU)**.

## Methodology
- **Game Theory:** Modeled as a potential game to ensure convergence to a Nash Equilibrium.
- **Algorithm:** Implemented **Log-Linear Learning (LLL)** with Softmax decision-making to balance exploration and exploitation.
- **Optimization:** UAVs iteratively update action preferences based on the marginal increase in system-wide coverage.


