# Bayesian UAV BRL

Prototype for Bayesian reinforcement learning applied to multi-UAV search and tracking.

## Goal

Build a first Bayesian RL environment where:

- the world is a 20x20 grid,
- 3 UAVs move on the grid,
- 4 targets are hidden on the grid,
- the agent maintains a belief map over target locations,
- the policy chooses joint actions for all UAVs.

## Current status

Implemented:

- custom UAV grid environment,
- 3 UAVs,
- 4 targets,
- Bayesian-style belief map,
- random baseline,
- PyTorch feature network,
- clean project structure.

Next steps:

1. DQN baseline.
2. BDQN head inspired by `kazizzad/BDQN-MxNet-Gluon`.
3. Thompson sampling exploration.
4. Search-and-track reward extension.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt