import random
from pathlib import Path

from config import ACTIONS
from config import ACTION_SUCCESS_PROBABILITY
from config import SIDE_SLIP_PROBABILITY

from config import MAX_STEPS_MULTIPLIER
from config import SPARSE_REWARDS

from config import REWARD_MODES
from config import SHAPING_PROGRESS_REWARD
from config import SHAPING_REGRESS_PENALTY

NORMAL = "."
WALL = "#"
START = "S"
KEY = "K"
DOOR = "D"
GOAL = "G"
PENALTY = "P"
TELEPORTER = "T"


class MazeEnvironment:
    def __init__(
        self,
        map_file_path,
        seed,
        reward_mode="sparse"
    ):
        self.map_file_path = Path(map_file_path)
        self.maze = self.load_maze()

        if reward_mode not in REWARD_MODES:
            raise ValueError(
                f"Invalid reward mode: {reward_mode}"
            )

        self.reward_mode = reward_mode

        self.random_generator = random.Random(seed)

        self.start_position = self.find_position(START)

        self.key_position = self.find_position(KEY)
        self.goal_position = self.find_position(GOAL)

        self.teleporter_positions = self.find_all_positions(
            TELEPORTER
        )

        self.walkable_cell_count = (
            self.count_walkable_cells()
        )

        self.max_steps = (
            MAX_STEPS_MULTIPLIER
            * self.walkable_cell_count
        )

        self.agent_position = self.start_position
        self.has_key = False
        self.step_count = 0
        self.total_reward = 0.0
        self.done = False

    def load_maze(self):
        maze = []

        with self.map_file_path.open(
            "r",
            encoding="utf-8"
        ) as map_file:
            for line in map_file:
                clean_line = line.strip()

                if clean_line:
                    maze.append(list(clean_line))

        return maze

    def find_position(self, symbol):
        for row_index, row in enumerate(self.maze):
            for column_index, cell in enumerate(row):
                if cell == symbol:
                    return (row_index, column_index)

        raise ValueError(
            f"Symbol '{symbol}' was not found in the maze."
        )
    
    def find_all_positions(self, symbol):
        positions = []

        for row_index, row in enumerate(self.maze):
            for column_index, cell in enumerate(row):
                if cell == symbol:
                    positions.append(
                        (row_index, column_index)
                    )

        return positions
    
    def get_other_teleporter(self, current_teleporter):
        if len(self.teleporter_positions) != 2:
            raise ValueError(
                "The maze must have exactly two teleporters."
            )

        first_teleporter = self.teleporter_positions[0]
        second_teleporter = self.teleporter_positions[1]

        if current_teleporter == first_teleporter:
            return second_teleporter

        return first_teleporter

    def simulate_executed_action(
        self,
        state,
        executed_action
    ):
        if executed_action not in ACTIONS:
            raise ValueError(
                f"Invalid action: {executed_action}"
            )

        if self.is_terminal_state(state):
            return (
                state,
                0.0,
                True,
                "terminal_state"
            )

        current_row, current_column, key_status = state

        next_row = current_row
        next_column = current_column

        if executed_action == "up":
            next_row -= 1

        elif executed_action == "down":
            next_row += 1

        elif executed_action == "left":
            next_column -= 1

        elif executed_action == "right":
            next_column += 1

        done = False

        if not (
            0 <= next_row < len(self.maze)
            and 0 <= next_column < len(self.maze[0])
        ):
            next_state = state
            event = "wall_collision"

        else:
            next_cell = self.maze[next_row][next_column]

            if next_cell == WALL:
                next_state = state
                event = "wall_collision"

            elif next_cell == DOOR and key_status == 0:
                next_state = state
                event = "closed_door"

            else:
                next_key_status = key_status

                if next_cell == KEY and key_status == 0:
                    next_key_status = 1
                    event = "key_collected"

                elif next_cell == DOOR:
                    event = "door_passed"

                elif next_cell == GOAL:
                    event = "goal_reached"
                    done = True

                elif next_cell == TELEPORTER:
                    teleporter_destination = (
                        self.get_other_teleporter(
                            (next_row, next_column)
                        )
                    )

                    next_row, next_column = (
                        teleporter_destination
                    )

                    event = "teleported"

                elif next_cell == PENALTY:
                    event = "penalty_cell"

                else:
                    event = "normal_move"

                next_state = (
                    next_row,
                    next_column,
                    next_key_status
                )

        reward = self.get_reward(
            event=event,
            previous_state=state,
            next_state=next_state
        )

        return (
            next_state,
            reward,
            done,
            event
        )

    def get_transition_outcomes(
        self,
        state,
        selected_action
    ):
        if self.is_terminal_state(state):
            return []

        action_probabilities = (
            self.get_action_probabilities(
                selected_action
            )
        )

        transition_outcomes = []

        for executed_action, probability in action_probabilities:
            (
                next_state,
                reward,
                done,
                event
            ) = self.simulate_executed_action(
                state=state,
                executed_action=executed_action
            )

            outcome = {
                "probability": probability,
                "executed_action": executed_action,
                "next_state": next_state,
                "reward": reward,
                "done": done,
                "event": event
            }

            transition_outcomes.append(outcome)

        return transition_outcomes
    
    def get_state(self):
        row, column = self.agent_position

        if self.has_key:
            key_status = 1
        else:
            key_status = 0

        return (row, column, key_status)

    def count_walkable_cells(self):
        walkable_cell_count = 0

        for row in self.maze:
            for cell in row:
                if cell != WALL:
                    walkable_cell_count += 1

        return walkable_cell_count
    
    def reset(self):
        self.agent_position = self.start_position
        self.has_key = False
        self.step_count = 0
        self.total_reward = 0.0
        self.done = False

        return self.get_state()
    
    def get_action_probabilities(
        self,
        selected_action
    ):
        if selected_action not in ACTIONS:
            raise ValueError(
                f"Invalid action: {selected_action}"
            )

        if selected_action in ["up", "down"]:
            first_side_action = "left"
            second_side_action = "right"

        else:
            first_side_action = "up"
            second_side_action = "down"

        action_probabilities = [
            (
                selected_action,
                ACTION_SUCCESS_PROBABILITY
            ),
            (
                first_side_action,
                SIDE_SLIP_PROBABILITY
            ),
            (
                second_side_action,
                SIDE_SLIP_PROBABILITY
            )
        ]

        total_probability = sum(
            probability
            for action, probability
            in action_probabilities
        )

        if abs(total_probability - 1.0) > 0.000001:
            raise ValueError(
                "Action probabilities must sum to 1."
            )

        return action_probabilities

    def choose_executed_action(
        self,
        selected_action
    ):
        action_probabilities = (
            self.get_action_probabilities(
                selected_action
            )
        )

        random_number = (
            self.random_generator.random()
        )

        cumulative_probability = 0.0

        for action, probability in action_probabilities:
            cumulative_probability += probability

            if random_number < cumulative_probability:
                return action

        return action_probabilities[-1][0]
    
    def calculate_distance(
        self,
        first_position,
        second_position
    ):
        row_distance = abs(
            first_position[0] - second_position[0]
        )

        column_distance = abs(
            first_position[1] - second_position[1]
        )

        return row_distance + column_distance

    def get_shaping_reward(
        self,
        previous_state,
        next_state,
        event
    ):
        milestone_events = [
            "key_collected",
            "goal_reached",
            "step_limit_reached"
        ]

        if event in milestone_events:
            return 0.0

        previous_position = (
            previous_state[0],
            previous_state[1]
        )

        next_position = (
            next_state[0],
            next_state[1]
        )

        had_key_before_action = previous_state[2]

        if had_key_before_action == 0:
            target_position = self.key_position
        else:
            target_position = self.goal_position

        previous_distance = self.calculate_distance(
            previous_position,
            target_position
        )

        next_distance = self.calculate_distance(
            next_position,
            target_position
        )

        if next_distance < previous_distance:
            return SHAPING_PROGRESS_REWARD

        if next_distance > previous_distance:
            return SHAPING_REGRESS_PENALTY

        return 0.0
    
    def get_reward(
        self,
        event,
        previous_state,
        next_state
    ):
        if event not in SPARSE_REWARDS:
            raise ValueError(
                f"No reward is defined for event: {event}"
            )

        reward = SPARSE_REWARDS[event]

        if self.reward_mode == "shaped":
            shaping_reward = self.get_shaping_reward(
                previous_state=previous_state,
                next_state=next_state,
                event=event
            )

            reward += shaping_reward

        return reward
    
    def get_valid_states(self):
        valid_states = []

        for row_index, row in enumerate(self.maze):
            for column_index, cell in enumerate(row):
                if cell == WALL:
                    continue

                for key_status in [0, 1]:
                    impossible_without_key = (
                        key_status == 0
                        and cell in [KEY, DOOR, GOAL]
                    )

                    if impossible_without_key:
                        continue

                    state = (
                        row_index,
                        column_index,
                        key_status
                    )

                    valid_states.append(state)

        return valid_states

    def is_terminal_state(self, state):
        row, column, key_status = state

        is_goal_position = (
            row,
            column
        ) == self.goal_position

        return (
            is_goal_position
            and key_status == 1
        )
    
    def get_available_actions(self, state):
        if self.is_terminal_state(state):
            return []

        return ACTIONS.copy()
    
    def move(self, selected_action):
        if self.done:
            raise ValueError(
                "The episode is finished. Reset the environment."
            )

        if selected_action not in ACTIONS:
            raise ValueError(
                f"Invalid action: {selected_action}"
            )

        previous_state = self.get_state()

        executed_action = self.choose_executed_action(
            selected_action
        )

        (
            next_state,
            reward,
            transition_done,
            event
        ) = self.simulate_executed_action(
            state=previous_state,
            executed_action=executed_action
        )

        self.agent_position = (
            next_state[0],
            next_state[1]
        )

        self.has_key = bool(next_state[2])
        self.done = transition_done

        self.step_count += 1

        if (
            self.step_count >= self.max_steps
            and not self.done
        ):
            self.done = True
            event = "step_limit_reached"

            reward = self.get_reward(
                event=event,
                previous_state=previous_state,
                next_state=next_state
            )

        self.total_reward += reward

        return (
            next_state,
            reward,
            self.done,
            event,
            executed_action
        )