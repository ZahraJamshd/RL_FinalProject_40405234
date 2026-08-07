import csv
import sys
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from config import REWARD_MODES
from config import VALUE_ITERATION_GAMMAS
from config import VALUE_ITERATION_THRESHOLD
from config import BASE_SEED
from config import VALUE_ITERATION_GAMMA
from config import Q_LEARNING_EPSILON_SCHEDULES
from config import Q_LEARNING_SEEDS
from config import Q_LEARNING_SUCCESS_WINDOW
from config import SARSA_LAMBDA_EPSILON_SCHEDULE
from config import SARSA_LAMBDA_GAMMA
from config import SARSA_LAMBDA_SEEDS
from config import SARSA_LAMBDA_SUCCESS_WINDOW
from config import SARSA_LAMBDA_VALUES

CONVERGENCE_FILE_PATH = (
    PROJECT_ROOT
    / "results"
    / "raw_data"
    / "value_iteration_convergence.csv"
)

FIGURE_FILE_PATH = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "value_iteration_convergence.png"
)

VALUE_MODELS_FILE_PATH = (
    PROJECT_ROOT
    / "results"
    / "models"
    / "value_iteration_models.csv"
)

MAP_FILE_PATH = (
    PROJECT_ROOT
    / "environments"
    / "maps"
    / f"maze_seed_{BASE_SEED}.txt"
)

VALUE_HEATMAP_FILE_PATH = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "value_iteration_heatmap.png"
)

POLICY_FIGURE_FILE_PATH = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "value_iteration_policy.png"
)

Q_LEARNING_EPISODE_FILE_PATH = (
    PROJECT_ROOT
    / "results"
    / "raw_data"
    / "q_learning_episode_metrics.csv"
)

Q_LEARNING_EPSILON_FIGURE_FILE_PATH = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "q_learning_epsilon_schedules.png"
)

Q_LEARNING_SPARSE_TRAINING_FIGURE_FILE_PATH = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "q_learning_training_sparse.png"
)

Q_LEARNING_SHAPED_TRAINING_FIGURE_FILE_PATH = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "q_learning_training_shaped.png"
)

Q_LEARNING_MODEL_FILE_PATH = (
    PROJECT_ROOT
    / "results"
    / "models"
    / "q_learning_models.csv"
)

Q_LEARNING_HEATMAP_FILE_PATH = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "q_learning_heatmap.png"
)

SELECTED_Q_GAMMA = VALUE_ITERATION_GAMMA
SELECTED_Q_EPSILON_SCHEDULE = "exponential"
SELECTED_Q_SEED = BASE_SEED

Q_LEARNING_POLICY_FILE_PATH = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "q_learning_policy.png"
)


SARSA_LAMBDA_EPISODE_FILE_PATH = (
    PROJECT_ROOT
    / "results"
    / "raw_data"
    / "sarsa_lambda_episode_metrics.csv"
)

SARSA_LAMBDA_SPARSE_TRAINING_FIGURE_FILE_PATH = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "sarsa_lambda_training_sparse.png"
)

SARSA_LAMBDA_SHAPED_TRAINING_FIGURE_FILE_PATH = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "sarsa_lambda_training_shaped.png"
)

SARSA_LAMBDA_MODEL_FILE_PATH = (
    PROJECT_ROOT
    / "results"
    / "models"
    / "sarsa_lambda_models.csv"
)

SARSA_LAMBDA_HEATMAP_FILE_PATH = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "sarsa_lambda_heatmap.png"
)

SELECTED_SARSA_LAMBDA = 0.3

SELECTED_SARSA_SEED = BASE_SEED

SARSA_LAMBDA_POLICY_FILE_PATH = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "sarsa_lambda_policy.png"
)

POLICY_AGREEMENT_SUMMARY_FILE_PATH = (
    PROJECT_ROOT
    / "results"
    / "raw_data"
    / "policy_agreement_summary.csv"
)

POLICY_AGREEMENT_FIGURE_FILE_PATH = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "policy_agreement_heatmap.png"
)

def load_convergence_data():
    convergence_data = {}

    with CONVERGENCE_FILE_PATH.open(
        "r",
        encoding="utf-8"
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            reward_mode = row["reward_mode"]
            gamma = float(row["gamma"])
            iteration = int(row["iteration"])
            maximum_change = float(
                row["maximum_change"]
            )

            result_key = (
                reward_mode,
                gamma
            )

            if result_key not in convergence_data:
                convergence_data[result_key] = {
                    "iterations": [],
                    "maximum_changes": []
                }

            convergence_data[result_key][
                "iterations"
            ].append(iteration)

            convergence_data[result_key][
                "maximum_changes"
            ].append(maximum_change)

    return convergence_data


def plot_value_iteration_convergence():
    convergence_data = load_convergence_data()

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
        sharey=True
    )

    gamma_colors = {
        0.5: "tab:blue",
        0.9: "tab:orange",
        0.99: "tab:green"
    }

    for axis, reward_mode in zip(
        axes,
        REWARD_MODES
    ):
        for gamma in VALUE_ITERATION_GAMMAS:
            result_key = (
                reward_mode,
                gamma
            )

            data = convergence_data[result_key]

            axis.plot(
                data["iterations"],
                data["maximum_changes"],
                label=f"Gamma = {gamma}",
                color=gamma_colors[gamma],
                linewidth=2
            )

        axis.axhline(
            y=VALUE_ITERATION_THRESHOLD,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="Convergence threshold"
        )

        axis.set_title(
            f"{reward_mode.capitalize()} reward"
        )

        axis.set_xlabel("Iteration")
        axis.set_yscale("log")

        axis.grid(
            True,
            which="both",
            alpha=0.25
        )

        axis.legend()

    axes[0].set_ylabel("Maximum value change")

    figure.suptitle(
        "Value Iteration Convergence",
        fontsize=15
    )

    figure.tight_layout()

    FIGURE_FILE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    figure.savefig(
        FIGURE_FILE_PATH,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(figure)

    print(
        f"Convergence figure saved in: "
        f"{FIGURE_FILE_PATH}"
    )

def plot_value_heatmaps():
    with open(MAP_FILE_PATH, "r", encoding="utf-8") as map_file:
        maze = [
            list(line.strip())
            for line in map_file
            if line.strip()
        ]

    maze_height = len(maze)
    maze_width = len(maze[0])

    value_matrices = {}

    for reward_mode in REWARD_MODES:
        value_matrices[(reward_mode, 0)] = np.full(
            (maze_height, maze_width),
            np.nan
        )
        value_matrices[(reward_mode, 1)] = np.full(
            (maze_height, maze_width),
            np.nan
        )

    with open(VALUE_MODELS_FILE_PATH, "r", encoding="utf-8") as model_file:
        reader = csv.DictReader(model_file)

        for row in reader:
            gamma = float(row["gamma"])

            if abs(gamma - VALUE_ITERATION_GAMMA) > 1e-9:
                continue

            reward_mode = row["reward_mode"]
            has_key = int(row["has_key"])
            state_row = int(row["row"])
            state_column = int(row["column"])
            value = float(row["value"])

            value_matrices[(reward_mode, has_key)][
                state_row,
                state_column
            ] = value

    all_values = []

    for matrix in value_matrices.values():
        valid_values = matrix[~np.isnan(matrix)]
        all_values.extend(valid_values)

    minimum_value = min(all_values)
    maximum_value = max(all_values)

    color_map = plt.get_cmap("viridis").copy()
    color_map.set_bad(color="lightgray")

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 10),
        constrained_layout=True
    )

    special_symbols = {"S", "K", "D", "G", "P", "T"}
    image = None

    for reward_index, reward_mode in enumerate(REWARD_MODES):
        for key_index, has_key in enumerate((0, 1)):
            axis = axes[reward_index, key_index]
            matrix = value_matrices[(reward_mode, has_key)]

            image = axis.imshow(
                matrix,
                cmap=color_map,
                vmin=minimum_value,
                vmax=maximum_value
            )

            key_label = "Before collecting key"

            if has_key == 1:
                key_label = "After collecting key"

            axis.set_title(
                f"{reward_mode.capitalize()} - {key_label}"
            )
            axis.set_xlabel("Column")
            axis.set_ylabel("Row")

            axis.set_xticks(range(0, maze_width, 2))
            axis.set_yticks(range(0, maze_height, 2))

            axis.set_xticks(
                np.arange(-0.5, maze_width, 1),
                minor=True
            )
            axis.set_yticks(
                np.arange(-0.5, maze_height, 1),
                minor=True
            )
            axis.grid(
                which="minor",
                color="white",
                linewidth=0.3,
                alpha=0.4
            )
            axis.tick_params(
                which="minor",
                bottom=False,
                left=False
            )

            for row_index in range(maze_height):
                for column_index in range(maze_width):
                    symbol = maze[row_index][column_index]

                    if symbol in special_symbols:
                        axis.text(
                            column_index,
                            row_index,
                            symbol,
                            ha="center",
                            va="center",
                            fontsize=7,
                            fontweight="bold",
                            color="black",
                            bbox={
                                "facecolor": "white",
                                "alpha": 0.65,
                                "edgecolor": "none",
                                "pad": 0.5
                            }
                        )

    figure.suptitle(
        f"Value Iteration heatmaps (gamma = {VALUE_ITERATION_GAMMA})"
    )

    color_bar = figure.colorbar(
        image,
        ax=axes.ravel().tolist(),
        shrink=0.85
    )
    color_bar.set_label("State value V(s)")

    VALUE_HEATMAP_FILE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    figure.savefig(
        VALUE_HEATMAP_FILE_PATH,
        dpi=200,
        bbox_inches="tight"
    )
    plt.close(figure)

    print(f"Value heatmap saved in: {VALUE_HEATMAP_FILE_PATH}")

def plot_value_iteration_policy():

    with open(MAP_FILE_PATH, "r", encoding="utf-8") as map_file:
        maze = [
            list(line.strip())
            for line in map_file
            if line.strip()
        ]

    maze_height = len(maze)
    maze_width = len(maze[0])

    policies = {}

    for reward_mode in REWARD_MODES:
        policies[(reward_mode, 0)] = {}
        policies[(reward_mode, 1)] = {}

    action_arrows = {
        "up": "↑",
        "down": "↓",
        "left": "←",
        "right": "→"
    }

    with open(VALUE_MODELS_FILE_PATH, "r", encoding="utf-8") as model_file:
        reader = csv.DictReader(model_file)

        for row in reader:
            gamma = float(row["gamma"])

            if abs(gamma - VALUE_ITERATION_GAMMA) > 1e-9:
                continue

            if row["is_terminal"] == "True":
                continue

            best_action = row["best_action"]

            if best_action not in action_arrows:
                continue

            reward_mode = row["reward_mode"]
            has_key = int(row["has_key"])
            state_row = int(row["row"])
            state_column = int(row["column"])

            policies[(reward_mode, has_key)][
                (state_row, state_column)
            ] = best_action

    wall_map = np.zeros((maze_height, maze_width))

    for row_index in range(maze_height):
        for column_index in range(maze_width):
            if maze[row_index][column_index] == "#":
                wall_map[row_index, column_index] = 1

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 10),
        constrained_layout=True
    )

    special_symbols = {"S", "K", "D", "G", "P", "T"}

    for reward_index, reward_mode in enumerate(REWARD_MODES):
        for key_index, has_key in enumerate((0, 1)):
            axis = axes[reward_index, key_index]

            axis.imshow(
                wall_map,
                cmap="Greys",
                vmin=0,
                vmax=1
            )

            key_label = "Before collecting key"

            if has_key == 1:
                key_label = "After collecting key"

            axis.set_title(
                f"{reward_mode.capitalize()} - {key_label}"
            )
            axis.set_xlabel("Column")
            axis.set_ylabel("Row")

            axis.set_xticks(range(0, maze_width, 2))
            axis.set_yticks(range(0, maze_height, 2))

            axis.set_xticks(
                np.arange(-0.5, maze_width, 1),
                minor=True
            )
            axis.set_yticks(
                np.arange(-0.5, maze_height, 1),
                minor=True
            )
            axis.grid(
                which="minor",
                color="gray",
                linewidth=0.3,
                alpha=0.5
            )
            axis.tick_params(
                which="minor",
                bottom=False,
                left=False
            )

            current_policy = policies[(reward_mode, has_key)]

            for state, best_action in current_policy.items():
                state_row, state_column = state

                axis.text(
                    state_column,
                    state_row,
                    action_arrows[best_action],
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="tab:blue",
                    fontweight="bold"
                )

            for row_index in range(maze_height):
                for column_index in range(maze_width):
                    symbol = maze[row_index][column_index]

                    if symbol in special_symbols:
                        axis.text(
                            column_index - 0.32,
                            row_index - 0.30,
                            symbol,
                            ha="left",
                            va="top",
                            fontsize=6,
                            color="darkred",
                            fontweight="bold"
                        )

    figure.suptitle(
        f"Value Iteration optimal policies "
        f"(gamma = {VALUE_ITERATION_GAMMA})"
    )

    POLICY_FIGURE_FILE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    figure.savefig(
        POLICY_FIGURE_FILE_PATH,
        dpi=200,
        bbox_inches="tight"
    )
    plt.close(figure)

    print(
        f"Value Iteration policy saved in: "
        f"{POLICY_FIGURE_FILE_PATH}"
    )

def plot_q_learning_epsilon_schedules():
    epsilon_data = {}

    for epsilon_schedule in (
        Q_LEARNING_EPSILON_SCHEDULES
    ):
        epsilon_data[epsilon_schedule] = {}

    with Q_LEARNING_EPISODE_FILE_PATH.open(
        "r",
        encoding="utf-8"
    ) as episode_file:
        reader = csv.DictReader(episode_file)

        for row in reader:
            epsilon_schedule = (
                row["epsilon_schedule"]
            )
            episode = int(row["episode"])
            epsilon = float(row["epsilon"])

            if episode not in (
                epsilon_data[epsilon_schedule]
            ):
                epsilon_data[epsilon_schedule][
                    episode
                ] = epsilon

    schedule_colors = {
        "linear": "tab:blue",
        "exponential": "tab:orange"
    }

    schedule_labels = {
        "linear": "Linear decay",
        "exponential": "Exponential decay"
    }

    figure, axis = plt.subplots(
        figsize=(10, 5.5)
    )

    for epsilon_schedule in (
        Q_LEARNING_EPSILON_SCHEDULES
    ):
        episodes = sorted(
            epsilon_data[epsilon_schedule].keys()
        )

        epsilon_values = [
            epsilon_data[epsilon_schedule][episode]
            for episode in episodes
        ]

        axis.plot(
            episodes,
            epsilon_values,
            label=schedule_labels[
                epsilon_schedule
            ],
            color=schedule_colors[
                epsilon_schedule
            ],
            linewidth=2
        )

    axis.set_title(
        "Q-Learning Epsilon Decay Schedules"
    )
    axis.set_xlabel("Episode")
    axis.set_ylabel("Epsilon")

    axis.set_ylim(0.0, 1.05)

    axis.grid(
        True,
        alpha=0.25
    )

    axis.legend()

    figure.tight_layout()

    Q_LEARNING_EPSILON_FIGURE_FILE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    figure.savefig(
        Q_LEARNING_EPSILON_FIGURE_FILE_PATH,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(figure)

    print(
        f"Q-Learning epsilon figure saved in: "
        f"{Q_LEARNING_EPSILON_FIGURE_FILE_PATH}"
    )

def load_q_learning_training_data():
    training_data = {}

    metric_names = [
        "total_reward",
        "steps",
        "success",
        "wall_collisions",
        "penalty_entries"
    ]

    for reward_mode in REWARD_MODES:
        for epsilon_schedule in (
            Q_LEARNING_EPSILON_SCHEDULES
        ):
            for seed in Q_LEARNING_SEEDS:
                result_key = (
                    reward_mode,
                    epsilon_schedule,
                    seed
                )

                training_data[result_key] = {
                    "episodes": []
                }

                for metric_name in metric_names:
                    training_data[result_key][
                        metric_name
                    ] = []

    with Q_LEARNING_EPISODE_FILE_PATH.open(
        "r",
        encoding="utf-8"
    ) as episode_file:
        reader = csv.DictReader(episode_file)

        for row in reader:
            gamma = float(row["gamma"])

            if abs(
                gamma - VALUE_ITERATION_GAMMA
            ) > 1e-9:
                continue

            reward_mode = row["reward_mode"]
            epsilon_schedule = (
                row["epsilon_schedule"]
            )
            seed = int(row["seed"])

            result_key = (
                reward_mode,
                epsilon_schedule,
                seed
            )

            training_data[result_key][
                "episodes"
            ].append(
                int(row["episode"])
            )

            training_data[result_key][
                "total_reward"
            ].append(
                float(row["total_reward"])
            )

            training_data[result_key][
                "steps"
            ].append(
                float(row["steps"])
            )

            success = 0.0

            if row["success"] == "True":
                success = 1.0

            training_data[result_key][
                "success"
            ].append(success)

            training_data[result_key][
                "wall_collisions"
            ].append(
                float(row["wall_collisions"])
            )

            training_data[result_key][
                "penalty_entries"
            ].append(
                float(row["penalty_entries"])
            )

    return training_data

def smooth_training_values(
    values,
    window_size
):
    window = np.ones(window_size) / window_size

    return np.convolve(
        values,
        window,
        mode="valid"
    )

def plot_q_learning_training_metrics():
    training_data = load_q_learning_training_data()

    metric_information = [
        (
            "total_reward",
            "Episode reward",
            "Reward"
        ),
        (
            "steps",
            "Episode steps",
            "Steps"
        ),
        (
            "success",
            "Success rate",
            "Success rate"
        ),
        (
            "wall_collisions",
            "Wall collisions",
            "Collision count"
        ),
        (
            "penalty_entries",
            "Penalty cell entries",
            "Entry count"
        )
    ]

    schedule_colors = {
        "linear": "tab:blue",
        "exponential": "tab:orange"
    }

    figure_paths = {
        "sparse": (
            Q_LEARNING_SPARSE_TRAINING_FIGURE_FILE_PATH
        ),
        "shaped": (
            Q_LEARNING_SHAPED_TRAINING_FIGURE_FILE_PATH
        )
    }

    window_size = Q_LEARNING_SUCCESS_WINDOW

    for reward_mode in REWARD_MODES:
        figure, axes = plt.subplots(
            3,
            2,
            figsize=(13, 11)
        )

        axes = axes.flatten()

        for metric_index, metric_data in enumerate(
            metric_information
        ):
            metric_name, title, y_label = metric_data
            axis = axes[metric_index]

            for epsilon_schedule in (
                Q_LEARNING_EPSILON_SCHEDULES
            ):
                seed_curves = []

                for seed in Q_LEARNING_SEEDS:
                    data_key = (
                        reward_mode,
                        epsilon_schedule,
                        seed
                    )

                    values = np.array(
                        training_data[data_key][
                            metric_name
                        ],
                        dtype=float
                    )

                    smoothed_values = (
                        smooth_training_values(
                            values=values,
                            window_size=window_size
                        )
                    )

                    seed_curves.append(
                        smoothed_values
                    )

                seed_curves = np.array(seed_curves)

                mean_values = np.mean(
                    seed_curves,
                    axis=0
                )

                standard_deviation = np.std(
                    seed_curves,
                    axis=0
                )

                episode_numbers = np.arange(
                    window_size,
                    window_size + len(mean_values)
                )

                color = schedule_colors[
                    epsilon_schedule
                ]

                axis.plot(
                    episode_numbers,
                    mean_values,
                    color=color,
                    label=epsilon_schedule.capitalize()
                )

                axis.fill_between(
                    episode_numbers,
                    mean_values - standard_deviation,
                    mean_values + standard_deviation,
                    color=color,
                    alpha=0.15
                )

            axis.set_title(title)
            axis.set_xlabel("Episode")
            axis.set_ylabel(y_label)
            axis.grid(alpha=0.3)

            if metric_name == "success":
                axis.set_ylim(0, 1.05)

        legend_axis = axes[5]
        legend_axis.axis("off")

        handles, labels = (
            axes[0].get_legend_handles_labels()
        )

        legend_axis.legend(
            handles,
            labels,
            title="Epsilon schedule",
            loc="center",
            frameon=False
        )

        figure.suptitle(
            f"Q-Learning training - "
            f"{reward_mode.capitalize()} reward - "
            f"gamma = {VALUE_ITERATION_GAMMA}\n"
            f"Mean and standard deviation across "
            f"{len(Q_LEARNING_SEEDS)} seeds"
        )

        figure.tight_layout(
            rect=(0, 0, 1, 0.94)
        )

        figure_path = figure_paths[reward_mode]

        figure_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        figure.savefig(
            figure_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(figure)

        print(
            f"Q-Learning training figure: "
            f"{figure_path}"
        )

def plot_q_learning_heatmap():
    with MAP_FILE_PATH.open(
        "r",
        encoding="utf-8"
    ) as map_file:
        maze = [
            list(line.strip())
            for line in map_file
            if line.strip()
        ]

    maze_height = len(maze)
    maze_width = len(maze[0])

    q_value_matrices = {}

    for reward_mode in REWARD_MODES:
        for has_key in (0, 1):
            q_value_matrices[
                (reward_mode, has_key)
            ] = np.full(
                (maze_height, maze_width),
                np.nan
            )

    with Q_LEARNING_MODEL_FILE_PATH.open(
        "r",
        encoding="utf-8"
    ) as model_file:
        reader = csv.DictReader(model_file)

        for row in reader:
            gamma = float(row["gamma"])

            if (
                abs(gamma - SELECTED_Q_GAMMA)
                > 1e-9
            ):
                continue

            if (
                row["epsilon_schedule"]
                != SELECTED_Q_EPSILON_SCHEDULE
            ):
                continue

            if int(row["seed"]) != SELECTED_Q_SEED:
                continue

            reward_mode = row["reward_mode"]
            has_key = int(row["has_key"])
            state_row = int(row["row"])
            state_column = int(row["column"])
            maximum_q = float(row["max_q"])

            q_value_matrices[
                (reward_mode, has_key)
            ][
                state_row,
                state_column
            ] = maximum_q

    all_q_values = []

    for matrix in q_value_matrices.values():
        valid_values = matrix[
            ~np.isnan(matrix)
        ]

        all_q_values.extend(valid_values)

    minimum_q = min(all_q_values)
    maximum_q = max(all_q_values)

    color_map = plt.get_cmap("viridis").copy()
    color_map.set_bad(color="lightgray")

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 10),
        constrained_layout=True
    )

    special_symbols = {
        "S",
        "K",
        "D",
        "G",
        "P",
        "T"
    }

    image = None

    for reward_index, reward_mode in enumerate(
        REWARD_MODES
    ):
        for key_index, has_key in enumerate(
            (0, 1)
        ):
            axis = axes[
                reward_index,
                key_index
            ]

            matrix = q_value_matrices[
                (reward_mode, has_key)
            ]

            image = axis.imshow(
                matrix,
                cmap=color_map,
                vmin=minimum_q,
                vmax=maximum_q
            )

            key_label = "Before collecting key"

            if has_key == 1:
                key_label = "After collecting key"

            axis.set_title(
                f"{reward_mode.capitalize()} - "
                f"{key_label}"
            )

            axis.set_xlabel("Column")
            axis.set_ylabel("Row")

            axis.set_xticks(
                range(0, maze_width, 2)
            )

            axis.set_yticks(
                range(0, maze_height, 2)
            )

            axis.set_xticks(
                np.arange(-0.5, maze_width, 1),
                minor=True
            )

            axis.set_yticks(
                np.arange(-0.5, maze_height, 1),
                minor=True
            )

            axis.grid(
                which="minor",
                color="white",
                linewidth=0.3,
                alpha=0.4
            )

            axis.tick_params(
                which="minor",
                bottom=False,
                left=False
            )

            for row_index in range(maze_height):
                for column_index in range(
                    maze_width
                ):
                    symbol = maze[
                        row_index
                    ][column_index]

                    if symbol in special_symbols:
                        axis.text(
                            column_index,
                            row_index,
                            symbol,
                            ha="center",
                            va="center",
                            fontsize=7,
                            fontweight="bold",
                            color="black",
                            bbox={
                                "facecolor": "white",
                                "alpha": 0.65,
                                "edgecolor": "none",
                                "pad": 0.5
                            }
                        )

    figure.suptitle(
        f"Q-Learning state values - "
        f"gamma = {SELECTED_Q_GAMMA} - "
        f"{SELECTED_Q_EPSILON_SCHEDULE} epsilon decay - "
        f"seed = {SELECTED_Q_SEED}"
    )

    color_bar = figure.colorbar(
        image,
        ax=axes.ravel().tolist(),
        shrink=0.85
    )

    color_bar.set_label(
        "Maximum action value max Q(s, a)"
    )

    Q_LEARNING_HEATMAP_FILE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    figure.savefig(
        Q_LEARNING_HEATMAP_FILE_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figure)

    print(
        f"Q-Learning heatmap saved in: "
        f"{Q_LEARNING_HEATMAP_FILE_PATH}"
    )

def plot_q_learning_policy():
    with MAP_FILE_PATH.open(
        "r",
        encoding="utf-8"
    ) as map_file:
        maze = [
            list(line.strip())
            for line in map_file
            if line.strip()
        ]

    maze_height = len(maze)
    maze_width = len(maze[0])

    policies = {}

    for reward_mode in REWARD_MODES:
        for has_key in (0, 1):
            policies[
                (reward_mode, has_key)
            ] = {}

    action_arrows = {
        "up": "↑",
        "down": "↓",
        "left": "←",
        "right": "→"
    }

    with Q_LEARNING_MODEL_FILE_PATH.open(
        "r",
        encoding="utf-8"
    ) as model_file:
        reader = csv.DictReader(model_file)

        for row in reader:
            gamma = float(row["gamma"])

            if (
                abs(gamma - SELECTED_Q_GAMMA)
                > 1e-9
            ):
                continue

            if (
                row["epsilon_schedule"]
                != SELECTED_Q_EPSILON_SCHEDULE
            ):
                continue

            if int(row["seed"]) != SELECTED_Q_SEED:
                continue

            if row["is_terminal"] == "True":
                continue

            best_action = row["best_action"]

            if best_action not in action_arrows:
                continue

            reward_mode = row["reward_mode"]
            has_key = int(row["has_key"])
            state_row = int(row["row"])
            state_column = int(row["column"])

            policies[
                (reward_mode, has_key)
            ][
                (state_row, state_column)
            ] = best_action

    wall_map = np.zeros(
        (maze_height, maze_width)
    )

    for row_index in range(maze_height):
        for column_index in range(maze_width):
            if (
                maze[row_index][column_index]
                == "#"
            ):
                wall_map[
                    row_index,
                    column_index
                ] = 1

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 10),
        constrained_layout=True
    )

    special_symbols = {
        "S",
        "K",
        "D",
        "G",
        "P",
        "T"
    }

    for reward_index, reward_mode in enumerate(
        REWARD_MODES
    ):
        for key_index, has_key in enumerate(
            (0, 1)
        ):
            axis = axes[
                reward_index,
                key_index
            ]

            axis.imshow(
                wall_map,
                cmap="Greys",
                vmin=0,
                vmax=1
            )

            key_label = "Before collecting key"

            if has_key == 1:
                key_label = "After collecting key"

            axis.set_title(
                f"{reward_mode.capitalize()} - "
                f"{key_label}"
            )

            axis.set_xlabel("Column")
            axis.set_ylabel("Row")

            axis.set_xticks(
                range(0, maze_width, 2)
            )

            axis.set_yticks(
                range(0, maze_height, 2)
            )

            axis.set_xticks(
                np.arange(
                    -0.5,
                    maze_width,
                    1
                ),
                minor=True
            )

            axis.set_yticks(
                np.arange(
                    -0.5,
                    maze_height,
                    1
                ),
                minor=True
            )

            axis.grid(
                which="minor",
                color="gray",
                linewidth=0.3,
                alpha=0.5
            )

            axis.tick_params(
                which="minor",
                bottom=False,
                left=False
            )

            current_policy = policies[
                (reward_mode, has_key)
            ]

            for state, best_action in (
                current_policy.items()
            ):
                state_row, state_column = state

                axis.text(
                    state_column,
                    state_row,
                    action_arrows[best_action],
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="tab:blue",
                    fontweight="bold"
                )

            for row_index in range(maze_height):
                for column_index in range(
                    maze_width
                ):
                    symbol = maze[
                        row_index
                    ][column_index]

                    if symbol in special_symbols:
                        axis.text(
                            column_index - 0.32,
                            row_index - 0.30,
                            symbol,
                            ha="left",
                            va="top",
                            fontsize=6,
                            color="darkred",
                            fontweight="bold"
                        )

    figure.suptitle(
        f"Q-Learning policies - "
        f"gamma = {SELECTED_Q_GAMMA} - "
        f"{SELECTED_Q_EPSILON_SCHEDULE} epsilon decay - "
        f"seed = {SELECTED_Q_SEED}"
    )

    Q_LEARNING_POLICY_FILE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    figure.savefig(
        Q_LEARNING_POLICY_FILE_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figure)

    print(
        f"Q-Learning policy saved in: "
        f"{Q_LEARNING_POLICY_FILE_PATH}"
    )

def load_sarsa_lambda_training_data():
    training_data = {}

    for reward_mode in REWARD_MODES:
        for lambda_value in (
            SARSA_LAMBDA_VALUES
        ):
            for seed in SARSA_LAMBDA_SEEDS:
                data_key = (
                    reward_mode,
                    lambda_value,
                    seed
                )

                training_data[data_key] = {
                    "episodes": [],
                    "total_reward": [],
                    "steps": [],
                    "success": [],
                    "wall_collisions": [],
                    "penalty_entries": []
                }

    with SARSA_LAMBDA_EPISODE_FILE_PATH.open(
        "r",
        encoding="utf-8"
    ) as episode_file:
        reader = csv.DictReader(
            episode_file
        )

        for row in reader:
            gamma = float(row["gamma"])

            if (
                abs(
                    gamma
                    - SARSA_LAMBDA_GAMMA
                )
                > 1e-9
            ):
                continue

            if (
                row["epsilon_schedule"]
                != SARSA_LAMBDA_EPSILON_SCHEDULE
            ):
                continue

            reward_mode = row["reward_mode"]
            lambda_value = float(
                row["lambda"]
            )
            seed = int(row["seed"])

            data_key = (
                reward_mode,
                lambda_value,
                seed
            )

            if data_key not in training_data:
                continue

            success = 0

            if row["success"] == "True":
                success = 1

            training_data[data_key][
                "episodes"
            ].append(
                int(row["episode"])
            )

            training_data[data_key][
                "total_reward"
            ].append(
                float(row["total_reward"])
            )

            training_data[data_key][
                "steps"
            ].append(
                float(row["steps"])
            )

            training_data[data_key][
                "success"
            ].append(success)

            training_data[data_key][
                "wall_collisions"
            ].append(
                float(row["wall_collisions"])
            )

            training_data[data_key][
                "penalty_entries"
            ].append(
                float(row["penalty_entries"])
            )

    return training_data

def plot_sarsa_lambda_training_metrics():
    training_data = (
        load_sarsa_lambda_training_data()
    )

    metric_information = [
        (
            "total_reward",
            "Episode reward",
            "Reward"
        ),
        (
            "steps",
            "Episode steps",
            "Steps"
        ),
        (
            "success",
            "Success rate",
            "Success rate"
        ),
        (
            "wall_collisions",
            "Wall collisions",
            "Collision count"
        ),
        (
            "penalty_entries",
            "Penalty cell entries",
            "Entry count"
        )
    ]

    lambda_colors = {
        0.0: "tab:blue",
        0.3: "tab:orange",
        0.7: "tab:green",
        0.9: "tab:red"
    }

    lambda_line_styles = {
        0.0: "-",
        0.3: "--",
        0.7: "-.",
        0.9: ":"
    }

    figure_paths = {
        "sparse": (
            SARSA_LAMBDA_SPARSE_TRAINING_FIGURE_FILE_PATH
        ),
        "shaped": (
            SARSA_LAMBDA_SHAPED_TRAINING_FIGURE_FILE_PATH
        )
    }

    window_size = (
        SARSA_LAMBDA_SUCCESS_WINDOW
    )

    for reward_mode in REWARD_MODES:
        figure, axes = plt.subplots(
            3,
            2,
            figsize=(13, 11)
        )

        axes = axes.flatten()

        for metric_index, metric_data in enumerate(
            metric_information
        ):
            metric_name, title, y_label = (
                metric_data
            )

            axis = axes[metric_index]

            for lambda_value in (
                SARSA_LAMBDA_VALUES
            ):
                seed_curves = []

                for seed in SARSA_LAMBDA_SEEDS:
                    data_key = (
                        reward_mode,
                        lambda_value,
                        seed
                    )

                    values = np.array(
                        training_data[data_key][
                            metric_name
                        ],
                        dtype=float
                    )

                    smoothed_values = (
                        smooth_training_values(
                            values=values,
                            window_size=(
                                window_size
                            )
                        )
                    )

                    seed_curves.append(
                        smoothed_values
                    )

                seed_curves = np.array(
                    seed_curves
                )

                mean_values = np.mean(
                    seed_curves,
                    axis=0
                )

                standard_deviation = np.std(
                    seed_curves,
                    axis=0
                )

                episode_numbers = np.arange(
                    window_size,
                    window_size
                    + len(mean_values)
                )

                lower_values = (
                    mean_values
                    - standard_deviation
                )

                upper_values = (
                    mean_values
                    + standard_deviation
                )

                if metric_name == "success":
                    lower_values = np.clip(
                        lower_values,
                        0,
                        1
                    )

                    upper_values = np.clip(
                        upper_values,
                        0,
                        1
                    )

                color = lambda_colors[
                    lambda_value
                ]

                axis.plot(
                    episode_numbers,
                    mean_values,
                    color=color,
                    linestyle=(
                        lambda_line_styles[
                            lambda_value
                        ]
                    ),
                    label=(
                        f"Lambda = "
                        f"{lambda_value}"
                    )
                )

                axis.fill_between(
                    episode_numbers,
                    lower_values,
                    upper_values,
                    color=color,
                    alpha=0.12
                )

            axis.set_title(title)
            axis.set_xlabel("Episode")
            axis.set_ylabel(y_label)
            axis.grid(alpha=0.3)

            if metric_name == "success":
                axis.set_ylim(0, 1.05)

        legend_axis = axes[5]
        legend_axis.axis("off")

        handles, labels = (
            axes[0].get_legend_handles_labels()
        )

        legend_axis.legend(
            handles,
            labels,
            title="Eligibility trace",
            loc="center",
            frameon=False
        )

        figure.suptitle(
            f"SARSA(lambda) training - "
            f"{reward_mode.capitalize()} reward - "
            f"gamma = {SARSA_LAMBDA_GAMMA}\n"
            f"Mean and standard deviation across "
            f"{len(SARSA_LAMBDA_SEEDS)} seeds"
        )

        figure.tight_layout(
            rect=(0, 0, 1, 0.94)
        )

        figure_path = figure_paths[
            reward_mode
        ]

        figure_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        figure.savefig(
            figure_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(figure)

        print(
            f"SARSA(lambda) training figure: "
            f"{figure_path}"
        )

def plot_sarsa_lambda_heatmap():
    with MAP_FILE_PATH.open(
        "r",
        encoding="utf-8"
    ) as map_file:
        maze = [
            list(line.strip())
            for line in map_file
            if line.strip()
        ]

    maze_height = len(maze)
    maze_width = len(maze[0])

    q_value_matrices = {}

    for reward_mode in REWARD_MODES:
        for has_key in (0, 1):
            q_value_matrices[
                (reward_mode, has_key)
            ] = np.full(
                (maze_height, maze_width),
                np.nan
            )

    with SARSA_LAMBDA_MODEL_FILE_PATH.open(
        "r",
        encoding="utf-8"
    ) as model_file:
        reader = csv.DictReader(
            model_file
        )

        for row in reader:
            gamma = float(row["gamma"])

            if (
                abs(
                    gamma
                    - SARSA_LAMBDA_GAMMA
                )
                > 1e-9
            ):
                continue

            lambda_value = float(
                row["lambda"]
            )

            if (
                abs(
                    lambda_value
                    - SELECTED_SARSA_LAMBDA
                )
                > 1e-9
            ):
                continue

            if (
                row["epsilon_schedule"]
                != SARSA_LAMBDA_EPSILON_SCHEDULE
            ):
                continue

            if (
                int(row["seed"])
                != SELECTED_SARSA_SEED
            ):
                continue

            reward_mode = row["reward_mode"]
            has_key = int(row["has_key"])
            state_row = int(row["row"])
            state_column = int(row["column"])
            maximum_q = float(row["max_q"])

            q_value_matrices[
                (reward_mode, has_key)
            ][
                state_row,
                state_column
            ] = maximum_q

    all_q_values = []

    for matrix in q_value_matrices.values():
        valid_values = matrix[
            ~np.isnan(matrix)
        ]

        all_q_values.extend(
            valid_values
        )

    minimum_q = min(all_q_values)
    maximum_q = max(all_q_values)

    color_map = plt.get_cmap(
        "viridis"
    ).copy()

    color_map.set_bad(
        color="lightgray"
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 10),
        constrained_layout=True
    )

    special_symbols = {
        "S",
        "K",
        "D",
        "G",
        "P",
        "T"
    }

    image = None

    for reward_index, reward_mode in enumerate(
        REWARD_MODES
    ):
        for key_index, has_key in enumerate(
            (0, 1)
        ):
            axis = axes[
                reward_index,
                key_index
            ]

            matrix = q_value_matrices[
                (reward_mode, has_key)
            ]

            image = axis.imshow(
                matrix,
                cmap=color_map,
                vmin=minimum_q,
                vmax=maximum_q
            )

            key_label = "Before collecting key"

            if has_key == 1:
                key_label = "After collecting key"

            axis.set_title(
                f"{reward_mode.capitalize()} - "
                f"{key_label}"
            )

            axis.set_xlabel("Column")
            axis.set_ylabel("Row")

            axis.set_xticks(
                range(0, maze_width, 2)
            )

            axis.set_yticks(
                range(0, maze_height, 2)
            )

            axis.set_xticks(
                np.arange(
                    -0.5,
                    maze_width,
                    1
                ),
                minor=True
            )

            axis.set_yticks(
                np.arange(
                    -0.5,
                    maze_height,
                    1
                ),
                minor=True
            )

            axis.grid(
                which="minor",
                color="white",
                linewidth=0.3,
                alpha=0.4
            )

            axis.tick_params(
                which="minor",
                bottom=False,
                left=False
            )

            for row_index in range(
                maze_height
            ):
                for column_index in range(
                    maze_width
                ):
                    symbol = maze[
                        row_index
                    ][column_index]

                    if symbol in special_symbols:
                        axis.text(
                            column_index,
                            row_index,
                            symbol,
                            ha="center",
                            va="center",
                            fontsize=7,
                            fontweight="bold",
                            color="black",
                            bbox={
                                "facecolor": (
                                    "white"
                                ),
                                "alpha": 0.65,
                                "edgecolor": (
                                    "none"
                                ),
                                "pad": 0.5
                            }
                        )

    figure.suptitle(
        f"SARSA(lambda) state values - "
        f"lambda = {SELECTED_SARSA_LAMBDA} - "
        f"gamma = {SARSA_LAMBDA_GAMMA} - "
        f"seed = {SELECTED_SARSA_SEED}"
    )

    color_bar = figure.colorbar(
        image,
        ax=axes.ravel().tolist(),
        shrink=0.85
    )

    color_bar.set_label(
        "Maximum action value max Q(s, a)"
    )

    SARSA_LAMBDA_HEATMAP_FILE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    figure.savefig(
        SARSA_LAMBDA_HEATMAP_FILE_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figure)

    print(
        f"SARSA(lambda) heatmap saved in: "
        f"{SARSA_LAMBDA_HEATMAP_FILE_PATH}"
    )

def plot_sarsa_lambda_policy():
    with MAP_FILE_PATH.open(
        "r",
        encoding="utf-8"
    ) as map_file:
        maze = [
            list(line.strip())
            for line in map_file
            if line.strip()
        ]

    maze_height = len(maze)
    maze_width = len(maze[0])

    policies = {}

    for reward_mode in REWARD_MODES:
        for has_key in (0, 1):
            policies[
                (reward_mode, has_key)
            ] = {}

    action_arrows = {
        "up": "↑",
        "down": "↓",
        "left": "←",
        "right": "→"
    }

    with SARSA_LAMBDA_MODEL_FILE_PATH.open(
        "r",
        encoding="utf-8"
    ) as model_file:
        reader = csv.DictReader(
            model_file
        )

        for row in reader:
            gamma = float(row["gamma"])

            if (
                abs(
                    gamma
                    - SARSA_LAMBDA_GAMMA
                )
                > 1e-9
            ):
                continue

            lambda_value = float(
                row["lambda"]
            )

            if (
                abs(
                    lambda_value
                    - SELECTED_SARSA_LAMBDA
                )
                > 1e-9
            ):
                continue

            if (
                row["epsilon_schedule"]
                != SARSA_LAMBDA_EPSILON_SCHEDULE
            ):
                continue

            if (
                int(row["seed"])
                != SELECTED_SARSA_SEED
            ):
                continue

            if row["is_terminal"] == "True":
                continue

            best_action = row["best_action"]

            if best_action not in action_arrows:
                continue

            reward_mode = row["reward_mode"]
            has_key = int(row["has_key"])
            state_row = int(row["row"])
            state_column = int(row["column"])

            policies[
                (reward_mode, has_key)
            ][
                (state_row, state_column)
            ] = best_action

    wall_map = np.zeros(
        (maze_height, maze_width)
    )

    for row_index in range(maze_height):
        for column_index in range(
            maze_width
        ):
            if (
                maze[row_index][column_index]
                == "#"
            ):
                wall_map[
                    row_index,
                    column_index
                ] = 1

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 10),
        constrained_layout=True
    )

    special_symbols = {
        "S",
        "K",
        "D",
        "G",
        "P",
        "T"
    }

    for reward_index, reward_mode in enumerate(
        REWARD_MODES
    ):
        for key_index, has_key in enumerate(
            (0, 1)
        ):
            axis = axes[
                reward_index,
                key_index
            ]

            axis.imshow(
                wall_map,
                cmap="Greys",
                vmin=0,
                vmax=1
            )

            key_label = "Before collecting key"

            if has_key == 1:
                key_label = "After collecting key"

            axis.set_title(
                f"{reward_mode.capitalize()} - "
                f"{key_label}"
            )

            axis.set_xlabel("Column")
            axis.set_ylabel("Row")

            axis.set_xticks(
                range(0, maze_width, 2)
            )

            axis.set_yticks(
                range(0, maze_height, 2)
            )

            axis.set_xticks(
                np.arange(
                    -0.5,
                    maze_width,
                    1
                ),
                minor=True
            )

            axis.set_yticks(
                np.arange(
                    -0.5,
                    maze_height,
                    1
                ),
                minor=True
            )

            axis.grid(
                which="minor",
                color="gray",
                linewidth=0.3,
                alpha=0.5
            )

            axis.tick_params(
                which="minor",
                bottom=False,
                left=False
            )

            current_policy = policies[
                (reward_mode, has_key)
            ]

            for state, best_action in (
                current_policy.items()
            ):
                state_row, state_column = (
                    state
                )

                axis.text(
                    state_column,
                    state_row,
                    action_arrows[best_action],
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="tab:blue",
                    fontweight="bold"
                )

            for row_index in range(
                maze_height
            ):
                for column_index in range(
                    maze_width
                ):
                    symbol = maze[
                        row_index
                    ][column_index]

                    if symbol in special_symbols:
                        axis.text(
                            column_index - 0.32,
                            row_index - 0.30,
                            symbol,
                            ha="left",
                            va="top",
                            fontsize=6,
                            color="darkred",
                            fontweight="bold"
                        )

    figure.suptitle(
        f"SARSA(lambda) policies - "
        f"lambda = {SELECTED_SARSA_LAMBDA} - "
        f"gamma = {SARSA_LAMBDA_GAMMA} - "
        f"seed = {SELECTED_SARSA_SEED}"
    )

    SARSA_LAMBDA_POLICY_FILE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    figure.savefig(
        SARSA_LAMBDA_POLICY_FILE_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figure)

    print(
        f"SARSA(lambda) policy saved in: "
        f"{SARSA_LAMBDA_POLICY_FILE_PATH}"
    )

def load_policy_agreement_models():
    value_iteration_policies = {}
    q_learning_policies = {}
    sarsa_lambda_policies = {}

    with VALUE_MODELS_FILE_PATH.open(
        "r",
        encoding="utf-8"
    ) as model_file:
        reader = csv.DictReader(model_file)

        for row in reader:
            gamma = float(row["gamma"])

            if abs(
                gamma - VALUE_ITERATION_GAMMA
            ) > 0.000000000001:
                continue

            reward_mode = row["reward_mode"]

            state = (
                int(row["row"]),
                int(row["column"]),
                int(row["has_key"])
            )

            is_terminal = (
                row["is_terminal"]
                .strip()
                .lower()
                == "true"
            )

            if is_terminal:
                best_action = None
            else:
                best_action = (
                    row["best_action"].strip()
                    or None
                )

            value_iteration_policies[
                (reward_mode, state)
            ] = best_action

    with Q_LEARNING_MODEL_FILE_PATH.open(
        "r",
        encoding="utf-8"
    ) as model_file:
        reader = csv.DictReader(model_file)

        for row in reader:
            gamma = float(row["gamma"])

            if abs(
                gamma - SELECTED_Q_GAMMA
            ) > 0.000000000001:
                continue

            if (
                row["epsilon_schedule"]
                != SELECTED_Q_EPSILON_SCHEDULE
            ):
                continue

            reward_mode = row["reward_mode"]
            seed = int(row["seed"])

            state = (
                int(row["row"]),
                int(row["column"]),
                int(row["has_key"])
            )

            is_terminal = (
                row["is_terminal"]
                .strip()
                .lower()
                == "true"
            )

            if is_terminal:
                best_action = None
            else:
                best_action = (
                    row["best_action"].strip()
                    or None
                )

            q_learning_policies[
                (reward_mode, seed, state)
            ] = best_action

    with SARSA_LAMBDA_MODEL_FILE_PATH.open(
        "r",
        encoding="utf-8"
    ) as model_file:
        reader = csv.DictReader(model_file)

        for row in reader:
            gamma = float(row["gamma"])
            lambda_value = float(row["lambda"])

            if abs(
                gamma - SARSA_LAMBDA_GAMMA
            ) > 0.000000000001:
                continue

            if abs(
                lambda_value
                - SELECTED_SARSA_LAMBDA
            ) > 0.000000000001:
                continue

            if (
                row["epsilon_schedule"]
                != SARSA_LAMBDA_EPSILON_SCHEDULE
            ):
                continue

            reward_mode = row["reward_mode"]
            seed = int(row["seed"])

            state = (
                int(row["row"]),
                int(row["column"]),
                int(row["has_key"])
            )

            is_terminal = (
                row["is_terminal"]
                .strip()
                .lower()
                == "true"
            )

            if is_terminal:
                best_action = None
            else:
                best_action = (
                    row["best_action"].strip()
                    or None
                )

            sarsa_lambda_policies[
                (reward_mode, seed, state)
            ] = best_action

    return {
        "value_iteration": (
            value_iteration_policies
        ),
        "q_learning": q_learning_policies,
        "sarsa_lambda": (
            sarsa_lambda_policies
        )
    }


def calculate_policy_agreement_percentage(
    reference_policy,
    learned_policy,
    has_key=None
):
    total_states = 0
    matching_states = 0

    for state, reference_action in (
        reference_policy.items()
    ):
        if reference_action is None:
            continue

        if (
            has_key is not None
            and state[2] != has_key
        ):
            continue

        total_states += 1

        learned_action = learned_policy.get(
            state
        )

        if learned_action == reference_action:
            matching_states += 1

    if total_states == 0:
        agreement_percentage = 0.0
    else:
        agreement_percentage = (
            100.0
            * matching_states
            / total_states
        )

    return (
        total_states,
        matching_states,
        agreement_percentage
    )


def analyze_policy_agreement():
    policy_data = (
        load_policy_agreement_models()
    )

    value_iteration_policies = (
        policy_data["value_iteration"]
    )

    q_learning_policies = (
        policy_data["q_learning"]
    )

    sarsa_lambda_policies = (
        policy_data["sarsa_lambda"]
    )

    q_learning_seeds = sorted({
        key[1]
        for key in q_learning_policies
    })

    sarsa_lambda_seeds = sorted({
        key[1]
        for key in sarsa_lambda_policies
    })

    algorithm_settings = [
        {
            "name": "Q-Learning",
            "policies": q_learning_policies,
            "seeds": q_learning_seeds,
            "selected_seed": SELECTED_Q_SEED,
            "gamma": SELECTED_Q_GAMMA,
            "lambda": "",
            "epsilon_schedule": (
                SELECTED_Q_EPSILON_SCHEDULE
            )
        },
        {
            "name": "SARSA(lambda)",
            "policies": sarsa_lambda_policies,
            "seeds": sarsa_lambda_seeds,
            "selected_seed": (
                SELECTED_SARSA_SEED
            ),
            "gamma": SARSA_LAMBDA_GAMMA,
            "lambda": (
                SELECTED_SARSA_LAMBDA
            ),
            "epsilon_schedule": (
                SARSA_LAMBDA_EPSILON_SCHEDULE
            )
        }
    ]

    summary_rows = []

    state_groups = [
        ("all", None),
        ("before_key", 0),
        ("after_key", 1)
    ]

    for setting in algorithm_settings:
        for reward_mode in REWARD_MODES:
            reference_policy = {
                state: best_action
                for (
                    stored_reward_mode,
                    state
                ), best_action
                in value_iteration_policies.items()
                if (
                    stored_reward_mode
                    == reward_mode
                )
            }

            for seed in setting["seeds"]:
                learned_policy = {
                    state: best_action
                    for (
                        stored_reward_mode,
                        stored_seed,
                        state
                    ), best_action
                    in setting["policies"].items()
                    if (
                        stored_reward_mode
                        == reward_mode
                        and stored_seed == seed
                    )
                }

                for (
                    state_group,
                    has_key
                ) in state_groups:
                    (
                        total_states,
                        matching_states,
                        agreement_percentage
                    ) = (
                        calculate_policy_agreement_percentage(
                            reference_policy=(
                                reference_policy
                            ),
                            learned_policy=(
                                learned_policy
                            ),
                            has_key=has_key
                        )
                    )

                    summary_rows.append({
                        "algorithm": setting["name"],
                        "reward_mode": reward_mode,
                        "gamma": setting["gamma"],
                        "lambda": setting["lambda"],
                        "epsilon_schedule": (
                            setting[
                                "epsilon_schedule"
                            ]
                        ),
                        "seed": seed,
                        "state_group": state_group,
                        "total_states": total_states,
                        "matching_states": (
                            matching_states
                        ),
                        "agreement_percentage": (
                            agreement_percentage
                        )
                    })

    POLICY_AGREEMENT_SUMMARY_FILE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with POLICY_AGREEMENT_SUMMARY_FILE_PATH.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as summary_file:
        fieldnames = [
            "algorithm",
            "reward_mode",
            "gamma",
            "lambda",
            "epsilon_schedule",
            "seed",
            "state_group",
            "total_states",
            "matching_states",
            "agreement_percentage"
        ]

        writer = csv.DictWriter(
            summary_file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    print("Policy agreement with Value Iteration")
    print("-------------------------------------")

    for setting in algorithm_settings:
        for reward_mode in REWARD_MODES:
            percentages = [
                row["agreement_percentage"]
                for row in summary_rows
                if (
                    row["algorithm"]
                    == setting["name"]
                    and row["reward_mode"]
                    == reward_mode
                    and row["state_group"]
                    == "all"
                )
            ]

            mean_percentage = float(
                np.mean(percentages)
            )

            standard_deviation = float(
                np.std(percentages)
            )

            print(
                f"{setting['name']} | "
                f"{reward_mode} | "
                f"{mean_percentage:.2f}% "
                f"+/- "
                f"{standard_deviation:.2f}%"
            )

    print()

    with MAP_FILE_PATH.open(
        "r",
        encoding="utf-8"
    ) as map_file:
        maze_rows = [
            line.rstrip("\n")
            for line in map_file
            if line.strip()
        ]

    maze_height = len(maze_rows)
    maze_width = len(maze_rows[0])

    plot_rows = [
        (setting, reward_mode)
        for setting in algorithm_settings
        for reward_mode in REWARD_MODES
    ]

    colors = [
        "#bdbdbd",
        "#d95f5f",
        "#5aae61",
        "#4c78a8"
    ]

    color_map = ListedColormap(colors)

    figure, axes = plt.subplots(
        nrows=len(plot_rows),
        ncols=2,
        figsize=(12, 20)
    )

    for plot_row_index, (
        setting,
        reward_mode
    ) in enumerate(plot_rows):
        reference_policy = {
            state: best_action
            for (
                stored_reward_mode,
                state
            ), best_action
            in value_iteration_policies.items()
            if (
                stored_reward_mode
                == reward_mode
            )
        }

        selected_seed = setting[
            "selected_seed"
        ]

        learned_policy = {
            state: best_action
            for (
                stored_reward_mode,
                stored_seed,
                state
            ), best_action
            in setting["policies"].items()
            if (
                stored_reward_mode
                == reward_mode
                and stored_seed
                == selected_seed
            )
        }

        for column_index, has_key in enumerate(
            [0, 1]
        ):
            agreement_matrix = np.full(
                (
                    maze_height,
                    maze_width
                ),
                -1.0
            )

            for (
                state,
                reference_action
            ) in reference_policy.items():
                if state[2] != has_key:
                    continue

                row_index = state[0]
                map_column_index = state[1]

                if reference_action is None:
                    agreement_matrix[
                        row_index,
                        map_column_index
                    ] = 2.0
                    continue

                learned_action = (
                    learned_policy.get(state)
                )

                if (
                    learned_action
                    == reference_action
                ):
                    agreement_matrix[
                        row_index,
                        map_column_index
                    ] = 1.0
                else:
                    agreement_matrix[
                        row_index,
                        map_column_index
                    ] = 0.0

            (
                total_states,
                matching_states,
                agreement_percentage
            ) = calculate_policy_agreement_percentage(
                reference_policy=(
                    reference_policy
                ),
                learned_policy=learned_policy,
                has_key=has_key
            )

            axis = axes[
                plot_row_index,
                column_index
            ]

            axis.imshow(
                agreement_matrix,
                cmap=color_map,
                vmin=-1.5,
                vmax=2.5,
                interpolation="nearest"
            )

            key_label = (
                "Before key"
                if has_key == 0
                else "After key"
            )

            axis.set_title(
                f"{key_label} - "
                f"{agreement_percentage:.2f}% "
                f"agreement"
            )

            axis.set_xticks(
                range(maze_width)
            )

            axis.set_yticks(
                range(maze_height)
            )

            axis.set_xticks(
                np.arange(
                    -0.5,
                    maze_width,
                    1
                ),
                minor=True
            )

            axis.set_yticks(
                np.arange(
                    -0.5,
                    maze_height,
                    1
                ),
                minor=True
            )

            axis.grid(
                which="minor",
                color="#666666",
                linewidth=0.25,
                alpha=0.35
            )

            axis.tick_params(
                which="minor",
                bottom=False,
                left=False
            )

            axis.tick_params(
                labelsize=7
            )

            for maze_row in range(
                maze_height
            ):
                for maze_column in range(
                    maze_width
                ):
                    symbol = maze_rows[
                        maze_row
                    ][maze_column]

                    if symbol not in [".", "#"]:
                        axis.text(
                            maze_column,
                            maze_row,
                            symbol,
                            horizontalalignment=(
                                "center"
                            ),
                            verticalalignment=(
                                "center"
                            ),
                            fontsize=6,
                            fontweight="bold"
                        )

            if column_index == 0:
                reward_label = (
                    reward_mode.capitalize()
                )

                axis.set_ylabel(
                    f"{setting['name']}\n"
                    f"{reward_label} reward"
                )

    legend_items = [
        Patch(
            facecolor=colors[0],
            label="Wall or invalid state"
        ),
        Patch(
            facecolor=colors[1],
            label="Different action"
        ),
        Patch(
            facecolor=colors[2],
            label="Same action"
        ),
        Patch(
            facecolor=colors[3],
            label="Terminal state"
        )
    ]

    figure.suptitle(
        "Greedy Policy Agreement with "
        "Value Iteration"
    )

    figure.legend(
        handles=legend_items,
        loc="lower center",
        ncol=4
    )

    figure.tight_layout(
        rect=(0, 0.04, 1, 0.97)
    )

    POLICY_AGREEMENT_FIGURE_FILE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    figure.savefig(
        POLICY_AGREEMENT_FIGURE_FILE_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figure)

    print(
        "Policy agreement summary saved in: "
        f"{POLICY_AGREEMENT_SUMMARY_FILE_PATH}"
    )

    print(
        "Policy agreement heatmap saved in: "
        f"{POLICY_AGREEMENT_FIGURE_FILE_PATH}"
    )

    return summary_rows

if __name__ == "__main__":
    plot_value_iteration_convergence()
    plot_value_heatmaps()
    plot_value_iteration_policy()
    plot_q_learning_epsilon_schedules()
    plot_q_learning_training_metrics()
    plot_q_learning_heatmap()
    plot_q_learning_policy()
    plot_sarsa_lambda_training_metrics()
    plot_sarsa_lambda_heatmap()
    plot_sarsa_lambda_policy()
    analyze_policy_agreement()