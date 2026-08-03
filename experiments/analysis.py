import csv
import sys
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from config import REWARD_MODES
from config import VALUE_ITERATION_GAMMAS
from config import VALUE_ITERATION_THRESHOLD
from config import BASE_SEED
from config import VALUE_ITERATION_GAMMA
from config import REWARD_MODES


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

if __name__ == "__main__":
    plot_value_iteration_convergence()
    plot_value_heatmaps()
    plot_value_iteration_policy()