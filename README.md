# Reinforcement Learning Final Project

## Student Information

- Student ID: 40405234
- Base Seed: 3
- Maze Size: 18x18

## Project Description

This project implements and compares the following reinforcement learning algorithms in a dynamic maze environment:

- Value Iteration
- Q-Learning
- SARSA(λ) with replacing eligibility traces

The project will also include experiments, visualizations, a graphical interface, and a final analytical report. Experimental results, learned models, training metrics, and visualizations are stored in the `results` directory.

## Current Progress

- [x] Project repository initialized
- [x] Dynamic maze environment
- [x] Value Iteration
- [x] Q-Learning
- [x] SARSA(λ)
- [ ] Graphical interface
- [ ] Final analytical report

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

## Output Directories

- `results/raw_data`: episode metrics, convergence data, summaries, and update logs
- `results/models`: learned values, Q-values, and policies
- `results/figures`: convergence, training, heatmap, and policy figures
- `experiments/configs`: saved experimental configurations
