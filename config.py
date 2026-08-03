STUDENT_ID = "40405234"

BASE_SEED = int(STUDENT_ID[-2])

MAZE_SIZE = 15 + (BASE_SEED % 4)

ACTIONS = ["up", "down", "left", "right"]

ACTION_SUCCESS_PROBABILITY = 0.8
SIDE_SLIP_PROBABILITY = 0.1

MINIMUM_WALL_RATIO = 0.15
MINIMUM_PENALTY_CELLS = 5

EXTRA_FEATURE = "teleporter"
TELEPORTER_COUNT = 2

MAX_STEPS_MULTIPLIER = 3

SPARSE_REWARDS = {
    "normal_move": -0.1,
    "wall_collision": -1.0,
    "closed_door": -1.0,
    "key_collected": 10.0,
    "door_passed": -0.1,
    "goal_reached": 100.0,
    "teleported": -0.1,
    "penalty_cell": -5.0,
    "step_limit_reached": -10.0
}

REWARD_MODES = [
    "sparse",
    "shaped"
]

SHAPING_PROGRESS_REWARD = 0.5
SHAPING_REGRESS_PENALTY = -0.5

VALUE_ITERATION_GAMMA = 0.9

VALUE_ITERATION_GAMMAS = [
    0.5,
    0.9,
    0.99
]

VALUE_ITERATION_THRESHOLD = 0.000001
VALUE_ITERATION_MAX_ITERATIONS = 3000