# Reinforcement Learning Final Project

## Student Information

- Student ID: 40405234
- Base Seed: 3
- Maze Size: 18x18

## Project Description

This project implements and compares reinforcement learning algorithms in a stochastic maze environment containing walls, penalty cells, a key, a door, a goal, and teleporters.

The implemented algorithms are:

- Value Iteration
- Q-Learning
- SARSA(λ) with replacing eligibility traces

Both sparse and shaped reward modes are evaluated. The project includes experiments, visualizations, policy-agreement analysis, and a graphical interface. Experimental results, learned models, training metrics, and figures are stored in the `results` directory.

## Current Progress

- [x] Project repository initialized
- [x] Stochastic maze environment
- [x] Value Iteration
- [x] Q-Learning
- [x] SARSA(λ)
- [x] Graphical interface
- [x] Final analytical report

## Installation

Install the required Python packages:

```cmd
pip install -r requirements.txt
```

## Running the Project

Display and validate the maze environment:

```cmd
python main.py
```

Run all algorithm experiments:

```cmd
python experiments\run_experiments.py
```

Generate the analysis figures:

```cmd
python experiments\analysis.py
```

Launch the graphical interface:

```cmd
python gui\app.py
```

The graphical interface loads saved policies for Value Iteration, Q-Learning, and SARSA(λ). It supports sparse and shaped reward modes and provides Start, Pause, Step, and Reset controls for visualizing agent behavior.

## Output Directories

- `results/raw_data`: episode metrics, convergence data, summaries, and update logs
- `results/models`: learned values, Q-values, and policies
- `results/figures`: convergence, training, heatmap, and policy figures
- `experiments/configs`: saved experimental configurations
