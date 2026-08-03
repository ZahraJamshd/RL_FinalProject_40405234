import random

from collections import deque

from config import MINIMUM_PENALTY_CELLS
from config import MINIMUM_WALL_RATIO
from config import TELEPORTER_COUNT


NORMAL = "."
WALL = "#"
START = "S"
KEY = "K"
DOOR = "D"
GOAL = "G"
PENALTY = "P"
TELEPORTER = "T"


def create_maze(size, seed):
    random_generator = random.Random(seed)

    maze = []

    for row in range(size):
        new_row = []

        for column in range(size):
            new_row.append(NORMAL)

        maze.append(new_row)

    for index in range(size):
        maze[0][index] = WALL
        maze[size - 1][index] = WALL
        maze[index][0] = WALL
        maze[index][size - 1] = WALL

    goal_row = size - 3
    goal_column = size - 3

    maze[goal_row][goal_column] = GOAL

    door_position = (goal_row, goal_column - 1)
    maze[door_position[0]][door_position[1]] = DOOR

    maze[goal_row - 1][goal_column] = WALL
    maze[goal_row][goal_column + 1] = WALL
    maze[goal_row + 1][goal_column] = WALL

    possible_wall_rows = range(3, size - 4)

    selected_wall_rows = random_generator.sample(
        list(possible_wall_rows),
        4
    )

    for row in selected_wall_rows:
        start_column = random_generator.randint(
            2,
            size - 8
        )

        for column in range(start_column, start_column + 4):
            maze[row][column] = WALL

    start_position = (1, 1)
    maze[start_position[0]][start_position[1]] = START

    available_cells = []

    for row in range(1, size - 1):
        for column in range(1, size - 1):
            if maze[row][column] == NORMAL:
                available_cells.append((row, column))

    key_candidates = []

    for position in available_cells:
        distance_from_start = (
            abs(position[0] - start_position[0])
            + abs(position[1] - start_position[1])
        )

        distance_from_door = (
            abs(position[0] - door_position[0])
            + abs(position[1] - door_position[1])
        )

        if (
            distance_from_start >= size // 2
            and distance_from_door >= size // 3
        ):
            key_candidates.append(position)

    key_position = random_generator.choice(key_candidates)

    maze[key_position[0]][key_position[1]] = KEY
    available_cells.remove(key_position)

    number_of_special_cells = (
        TELEPORTER_COUNT
        + MINIMUM_PENALTY_CELLS
    )

    selected_cells = random_generator.sample(
        available_cells,
        number_of_special_cells
    )

    teleporter_positions = selected_cells[:TELEPORTER_COUNT]
    penalty_positions = selected_cells[TELEPORTER_COUNT:]

    for row, column in teleporter_positions:
        maze[row][column] = TELEPORTER

    for row, column in penalty_positions:
        maze[row][column] = PENALTY
    
    wall_count = 0
    penalty_count = 0

    for row in maze:
        wall_count += row.count(WALL)
        penalty_count += row.count(PENALTY)

    total_cell_count = size * size
    wall_ratio = wall_count / total_cell_count

    if wall_ratio < MINIMUM_WALL_RATIO:
        raise ValueError(
            "The generated maze does not have enough walls."
        )

    if penalty_count < MINIMUM_PENALTY_CELLS:
        raise ValueError(
            "The generated maze does not have enough penalty cells."
        )

    return maze


def print_maze(maze):
    for row in maze:
        print("".join(row))


def save_maze(maze, file_path):
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as map_file:
        for row in maze:
            map_file.write("".join(row) + "\n")

def find_all_positions(maze, symbol):
    positions = []

    for row_index, row in enumerate(maze):
        for column_index, cell in enumerate(row):
            if cell == symbol:
                positions.append((row_index, column_index))

    return positions


def find_single_position(maze, symbol):
    positions = find_all_positions(maze, symbol)

    if len(positions) != 1:
        raise ValueError(
            f"Expected exactly one '{symbol}' in the maze."
        )

    return positions[0]


def path_exists(maze, start_position, goal_position, door_is_open):
    queue = deque([start_position])
    visited = {start_position}

    teleporter_positions = find_all_positions(
        maze,
        TELEPORTER
    )

    directions = [
        (-1, 0),  
        (1, 0),   
        (0, -1),  
        (0, 1)    
    ]

    while queue:
        current_position = queue.popleft()

        if current_position == goal_position:
            return True

        current_row, current_column = current_position

        for row_change, column_change in directions:
            next_row = current_row + row_change
            next_column = current_column + column_change

            if not (
                0 <= next_row < len(maze)
                and 0 <= next_column < len(maze[0])
            ):
                continue

            next_cell = maze[next_row][next_column]

            if next_cell == WALL:
                continue

            if next_cell == DOOR and not door_is_open:
                continue

            next_position = (next_row, next_column)

            if (
                next_cell == TELEPORTER
                and len(teleporter_positions) == 2
            ):
                if next_position == teleporter_positions[0]:
                    next_position = teleporter_positions[1]
                else:
                    next_position = teleporter_positions[0]

            if next_position not in visited:
                visited.add(next_position)
                queue.append(next_position)

    return False


def validate_maze(maze):
    start_position = find_single_position(maze, START)
    key_position = find_single_position(maze, KEY)
    door_position = find_single_position(maze, DOOR)
    goal_position = find_single_position(maze, GOAL)

    teleporter_positions = find_all_positions(
        maze,
        TELEPORTER
    )

    if len(teleporter_positions) != TELEPORTER_COUNT:
        raise ValueError(
            "The maze has an incorrect number of teleporters."
        )

    start_to_key_exists = path_exists(
        maze=maze,
        start_position=start_position,
        goal_position=key_position,
        door_is_open=False
    )

    if not start_to_key_exists:
        raise ValueError(
            "There is no valid path from start to key."
        )

    key_to_door_exists = path_exists(
        maze=maze,
        start_position=key_position,
        goal_position=door_position,
        door_is_open=True
    )

    if not key_to_door_exists:
        raise ValueError(
            "There is no valid path from key to door."
        )

    door_to_goal_exists = path_exists(
        maze=maze,
        start_position=door_position,
        goal_position=goal_position,
        door_is_open=True
    )

    if not door_to_goal_exists:
        raise ValueError(
            "There is no valid path from door to goal."
        )

    return True