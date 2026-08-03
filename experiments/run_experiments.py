import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agents.value_iteration import run_value_iteration

from config import BASE_SEED
from config import MAZE_SIZE
from config import REWARD_MODES
from config import STUDENT_ID
from config import VALUE_ITERATION_GAMMAS
from config import VALUE_ITERATION_MAX_ITERATIONS
from config import VALUE_ITERATION_THRESHOLD

from environments.maze import MazeEnvironment


MAP_FILE_PATH = (
    PROJECT_ROOT
    / "environments"
    / "maps"
    / f"maze_seed_{BASE_SEED}.txt"
)

VALUE_ITERATION_MODEL_FILE_PATH = (
    PROJECT_ROOT
    / "results"
    / "models"
    / "value_iteration_models.csv"
)

VALUE_ITERATION_CONVERGENCE_FILE_PATH = (
    PROJECT_ROOT
    / "results"
    / "raw_data"
    / "value_iteration_convergence.csv"
)

VALUE_ITERATION_SUMMARY_FILE_PATH = (
    PROJECT_ROOT
    / "results"
    / "raw_data"
    / "value_iteration_summary.csv"
)

VALUE_ITERATION_CONFIG_FILE_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "configs"
    / "value_iteration_config.json"
)

def run_value_iteration_experiments():
    print("Value Iteration experiments")
    print("---------------------------")

    experiment_results = {}

    for reward_mode in REWARD_MODES:
        for gamma in VALUE_ITERATION_GAMMAS:
            environment = MazeEnvironment(
                map_file_path=MAP_FILE_PATH,
                seed=BASE_SEED,
                reward_mode=reward_mode
            )

            result = run_value_iteration(
                environment=environment,
                gamma=gamma,
                convergence_threshold=(
                    VALUE_ITERATION_THRESHOLD
                ),
                max_iterations=(
                    VALUE_ITERATION_MAX_ITERATIONS
                )
            )

            result_key = (
                reward_mode,
                gamma
            )

            experiment_results[result_key] = result

            start_state = (
                environment.start_position[0],
                environment.start_position[1],
                0
            )

            print(f"Reward mode: {reward_mode}")
            print(f"Gamma: {gamma}")
            print(f"Converged: {result['converged']}")
            print(f"Iterations: {result['iterations']}")

            print(
                f"Execution time: "
                f"{result['execution_time']:.4f} seconds"
            )

            print(
                f"Final delta: "
                f"{result['delta_history'][-1]:.10f}"
            )

            print(
                f"Start state value: "
                f"{result['values'][start_state]:.4f}"
            )

            print(
                f"Start state action: "
                f"{result['policy'][start_state]}"
            )

            print()

    output_files = save_value_iteration_results(
        experiment_results
    )

    print("Combined output files")
    print("---------------------")
    print(f"Models: {output_files['models']}")

    print(
        f"Convergence: "
        f"{output_files['convergence']}"
    )

    print(f"Summary: {output_files['summary']}")
    print(f"Config: {output_files['config']}")
    print()

    return experiment_results

def save_value_iteration_results(
    experiment_results
):
    environment = MazeEnvironment(
        map_file_path=MAP_FILE_PATH,
        seed=BASE_SEED,
        reward_mode="sparse"
    )

    output_paths = [
        VALUE_ITERATION_MODEL_FILE_PATH,
        VALUE_ITERATION_CONVERGENCE_FILE_PATH,
        VALUE_ITERATION_SUMMARY_FILE_PATH,
        VALUE_ITERATION_CONFIG_FILE_PATH
    ]

    for output_path in output_paths:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

    valid_states = environment.get_valid_states()

    with VALUE_ITERATION_MODEL_FILE_PATH.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as model_file:
        fieldnames = [
            "reward_mode",
            "gamma",
            "row",
            "column",
            "has_key",
            "value",
            "best_action",
            "is_terminal"
        ]

        writer = csv.DictWriter(
            model_file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for reward_mode in REWARD_MODES:
            for gamma in VALUE_ITERATION_GAMMAS:
                result = experiment_results[
                    (reward_mode, gamma)
                ]

                for state in valid_states:
                    row, column, has_key = state

                    writer.writerow({
                        "reward_mode": reward_mode,
                        "gamma": gamma,
                        "row": row,
                        "column": column,
                        "has_key": has_key,
                        "value": result["values"][state],
                        "best_action": (
                            result["policy"][state]
                        ),
                        "is_terminal": (
                            environment.is_terminal_state(
                                state
                            )
                        )
                    })

    with VALUE_ITERATION_CONVERGENCE_FILE_PATH.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as convergence_file:
        fieldnames = [
            "reward_mode",
            "gamma",
            "iteration",
            "maximum_change"
        ]

        writer = csv.DictWriter(
            convergence_file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for reward_mode in REWARD_MODES:
            for gamma in VALUE_ITERATION_GAMMAS:
                result = experiment_results[
                    (reward_mode, gamma)
                ]

                for iteration, maximum_change in enumerate(
                    result["delta_history"],
                    start=1
                ):
                    writer.writerow({
                        "reward_mode": reward_mode,
                        "gamma": gamma,
                        "iteration": iteration,
                        "maximum_change": maximum_change
                    })

    start_state = (
        environment.start_position[0],
        environment.start_position[1],
        0
    )

    with VALUE_ITERATION_SUMMARY_FILE_PATH.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as summary_file:
        fieldnames = [
            "reward_mode",
            "gamma",
            "converged",
            "iterations",
            "execution_time",
            "final_delta",
            "state_count",
            "start_state_value",
            "start_state_action"
        ]

        writer = csv.DictWriter(
            summary_file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for reward_mode in REWARD_MODES:
            for gamma in VALUE_ITERATION_GAMMAS:
                result = experiment_results[
                    (reward_mode, gamma)
                ]

                writer.writerow({
                    "reward_mode": reward_mode,
                    "gamma": gamma,
                    "converged": result["converged"],
                    "iterations": result["iterations"],
                    "execution_time": (
                        result["execution_time"]
                    ),
                    "final_delta": (
                        result["delta_history"][-1]
                    ),
                    "state_count": len(
                        result["values"]
                    ),
                    "start_state_value": (
                        result["values"][start_state]
                    ),
                    "start_state_action": (
                        result["policy"][start_state]
                    )
                })

    config_data = {
        "student_id": STUDENT_ID,
        "base_seed": BASE_SEED,
        "maze_size": MAZE_SIZE,
        "map_file": (
            MAP_FILE_PATH
            .relative_to(PROJECT_ROOT)
            .as_posix()
        ),
        "reward_modes": REWARD_MODES,
        "gammas": VALUE_ITERATION_GAMMAS,
        "convergence_threshold": (
            VALUE_ITERATION_THRESHOLD
        ),
        "max_iterations": (
            VALUE_ITERATION_MAX_ITERATIONS
        ),
        "state_count": len(valid_states),
        "experiment_count": (
            len(REWARD_MODES)
            * len(VALUE_ITERATION_GAMMAS)
        )
    }

    with VALUE_ITERATION_CONFIG_FILE_PATH.open(
        "w",
        encoding="utf-8"
    ) as config_file:
        json.dump(
            config_data,
            config_file,
            indent=4
        )

    return {
        "models": VALUE_ITERATION_MODEL_FILE_PATH,
        "convergence": (
            VALUE_ITERATION_CONVERGENCE_FILE_PATH
        ),
        "summary": VALUE_ITERATION_SUMMARY_FILE_PATH,
        "config": VALUE_ITERATION_CONFIG_FILE_PATH
    }

if __name__ == "__main__":
    run_value_iteration_experiments()