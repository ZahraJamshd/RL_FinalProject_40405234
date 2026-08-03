from pathlib import Path

from config import BASE_SEED
from config import EXTRA_FEATURE
from config import MAZE_SIZE
from config import STUDENT_ID
from environments.generator import create_maze
from environments.generator import print_maze
from environments.generator import save_maze

from environments.generator import validate_maze

from environments.maze import MazeEnvironment

maze = create_maze(
    size=MAZE_SIZE,
    seed=BASE_SEED
)

validate_maze(maze)

map_file_path = Path(
    f"environments/maps/maze_seed_{BASE_SEED}.txt"
)

save_maze(
    maze=maze,
    file_path=map_file_path
)

print("Reinforcement Learning Final Project")
print("------------------------------------")
print(f"Student ID: {STUDENT_ID}")
print(f"Base seed: {BASE_SEED}")
print(f"Maze size: {MAZE_SIZE} x {MAZE_SIZE}")
print(f"Extra feature: {EXTRA_FEATURE}")
print()

print_maze(maze)

print()
print("Maze validation: passed")
print(f"Map saved in: {map_file_path}")

environment = MazeEnvironment(
    map_file_path=map_file_path,
    seed=BASE_SEED
)

initial_state = environment.reset()

print()
print(f"Initial state: {initial_state}")
print(f"Agent position: {environment.agent_position}")
print(f"Has key: {environment.has_key}")

print(
    f"Walkable cells: "
    f"{environment.walkable_cell_count}"
)

print(
    f"Maximum episode steps: "
    f"{environment.max_steps}"
)

valid_states = environment.get_valid_states()

print(f"Valid MDP states: {len(valid_states)}")