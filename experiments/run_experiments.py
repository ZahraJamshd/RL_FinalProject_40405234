import csv
import json
import sys
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agents.value_iteration import run_value_iteration

from agents.q_learning import extract_q_learning_policy
from agents.q_learning import train_q_learning

from config import BASE_SEED
from config import MAZE_SIZE
from config import REWARD_MODES
from config import STUDENT_ID
from config import VALUE_ITERATION_GAMMAS
from config import VALUE_ITERATION_MAX_ITERATIONS
from config import VALUE_ITERATION_THRESHOLD
from config import Q_LEARNING_ALPHA
from config import Q_LEARNING_EPISODES
from config import Q_LEARNING_EPSILON_DECAY_EPISODES
from config import Q_LEARNING_EPSILON_END
from config import Q_LEARNING_EPSILON_SCHEDULES
from config import Q_LEARNING_EPSILON_START
from config import Q_LEARNING_GAMMAS
from config import Q_LEARNING_SEEDS
from config import Q_LEARNING_SUCCESS_WINDOW
from config import Q_LEARNING_UPDATE_LOG_EPISODE
from config import VALUE_ITERATION_GAMMA

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

Q_LEARNING_MODEL_FILE_PATH = (
    PROJECT_ROOT
    / "results"
    / "models"
    / "q_learning_models.csv"
)

Q_LEARNING_EPISODE_FILE_PATH = (
    PROJECT_ROOT
    / "results"
    / "raw_data"
    / "q_learning_episode_metrics.csv"
)

Q_LEARNING_UPDATE_LOG_FILE_PATH = (
    PROJECT_ROOT
    / "results"
    / "raw_data"
    / "q_learning_update_log.csv"
)

Q_LEARNING_SUMMARY_FILE_PATH = (
    PROJECT_ROOT
    / "results"
    / "raw_data"
    / "q_learning_summary.csv"
)

Q_LEARNING_CONFIG_FILE_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "configs"
    / "q_learning_config.json"
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

    print("Value Iteration output files")
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

def run_q_learning_experiments(
    number_of_episodes=Q_LEARNING_EPISODES
):
    print("Q-Learning experiments")
    print("----------------------")

    experiment_results = {}

    total_experiments = (
        len(REWARD_MODES)
        * len(Q_LEARNING_GAMMAS)
        * len(Q_LEARNING_EPSILON_SCHEDULES)
        * len(Q_LEARNING_SEEDS)
    )

    completed_experiments = 0

    for reward_mode in REWARD_MODES:
        for gamma in Q_LEARNING_GAMMAS:
            for epsilon_schedule in (
                Q_LEARNING_EPSILON_SCHEDULES
            ):
                for seed in Q_LEARNING_SEEDS:
                    completed_experiments += 1

                    environment = MazeEnvironment(
                        map_file_path=MAP_FILE_PATH,
                        seed=seed,
                        reward_mode=reward_mode
                    )

                    is_update_log_experiment = (
                        reward_mode == "sparse"
                        and gamma
                        == VALUE_ITERATION_GAMMA
                        and epsilon_schedule
                        == "linear"
                        and seed == BASE_SEED
                    )

                    record_update_episode = None

                    if is_update_log_experiment:
                        record_update_episode = min(
                            Q_LEARNING_UPDATE_LOG_EPISODE,
                            number_of_episodes
                        )

                    start_time = time.perf_counter()

                    result = train_q_learning(
                        environment=environment,
                        number_of_episodes=(
                            number_of_episodes
                        ),
                        alpha=Q_LEARNING_ALPHA,
                        gamma=gamma,
                        epsilon_schedule=(
                            epsilon_schedule
                        ),
                        epsilon_start=(
                            Q_LEARNING_EPSILON_START
                        ),
                        epsilon_end=(
                            Q_LEARNING_EPSILON_END
                        ),
                        epsilon_decay_episodes=(
                            Q_LEARNING_EPSILON_DECAY_EPISODES
                        ),
                        success_window=(
                            Q_LEARNING_SUCCESS_WINDOW
                        ),
                        seed=seed,
                        record_update_episode=(
                            record_update_episode
                        )
                    )

                    result["execution_time"] = (
                        time.perf_counter()
                        - start_time
                    )

                    result["policy"] = (
                        extract_q_learning_policy(
                            environment=environment,
                            q_table=result["q_table"]
                        )
                    )

                    result["reward_mode"] = (
                        reward_mode
                    )
                    result["gamma"] = gamma
                    result["epsilon_schedule"] = (
                        epsilon_schedule
                    )
                    result["seed"] = seed
                    result["number_of_episodes"] = (
                        number_of_episodes
                    )

                    result_key = (
                        reward_mode,
                        gamma,
                        epsilon_schedule,
                        seed
                    )

                    experiment_results[result_key] = (
                        result
                    )

                    final_record = (
                        result["episode_records"][-1]
                    )

                    print(
                        f"[{completed_experiments}"
                        f"/{total_experiments}] "
                        f"Reward: {reward_mode} | "
                        f"Gamma: {gamma} | "
                        f"Schedule: "
                        f"{epsilon_schedule} | "
                        f"Seed: {seed}"
                    )

                    print(
                        f"    Final epsilon: "
                        f"{final_record['epsilon']:.4f} | "
                        f"Success rate: "
                        f"{final_record['success_rate']:.2f} | "
                        f"Time: "
                        f"{result['execution_time']:.4f}s"
                    )

    output_files = save_q_learning_results(
        experiment_results
    )

    print()
    print("Q-Learning output files")
    print("-----------------------")

    print(
        f"Episode metrics: "
        f"{output_files['episodes']}"
    )

    print(
        f"Models: "
        f"{output_files['models']}"
    )

    print(
        f"Update log: "
        f"{output_files['updates']}"
    )

    print(
        f"Summary: "
        f"{output_files['summary']}"
    )

    print(
        f"Config: "
        f"{output_files['config']}"
    )
    print()

    return experiment_results

def save_q_learning_results(
    experiment_results
):
    output_paths = [
        Q_LEARNING_MODEL_FILE_PATH,
        Q_LEARNING_EPISODE_FILE_PATH,
        Q_LEARNING_UPDATE_LOG_FILE_PATH,
        Q_LEARNING_SUMMARY_FILE_PATH,
        Q_LEARNING_CONFIG_FILE_PATH
    ]

    for output_path in output_paths:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

    fieldnames = [
        "reward_mode",
        "gamma",
        "epsilon_schedule",
        "seed",
        "episode",
        "alpha",
        "epsilon",
        "total_reward",
        "steps",
        "success",
        "success_rate",
        "wall_collisions",
        "penalty_entries",
        "final_event"
    ]

    with Q_LEARNING_EPISODE_FILE_PATH.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as episode_file:
        writer = csv.DictWriter(
            episode_file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for result_key, result in (
            experiment_results.items()
        ):
            (
                reward_mode,
                gamma,
                epsilon_schedule,
                seed
            ) = result_key

            for episode_record in (
                result["episode_records"]
            ):
                writer.writerow({
                    "reward_mode": reward_mode,
                    "gamma": gamma,
                    "epsilon_schedule": (
                        epsilon_schedule
                    ),
                    "seed": seed,
                    "episode": (
                        episode_record["episode"]
                    ),
                    "alpha": (
                        episode_record["alpha"]
                    ),
                    "epsilon": (
                        episode_record["epsilon"]
                    ),
                    "total_reward": (
                        episode_record["total_reward"]
                    ),
                    "steps": (
                        episode_record["steps"]
                    ),
                    "success": (
                        episode_record["success"]
                    ),
                    "success_rate": (
                        episode_record["success_rate"]
                    ),
                    "wall_collisions": (
                        episode_record[
                            "wall_collisions"
                        ]
                    ),
                    "penalty_entries": (
                        episode_record[
                            "penalty_entries"
                        ]
                    ),
                    "final_event": (
                        episode_record["final_event"]
                    )
                })

    environment = MazeEnvironment(
        map_file_path=MAP_FILE_PATH,
        seed=BASE_SEED,
        reward_mode="sparse"
    )

    valid_states = environment.get_valid_states()

    with Q_LEARNING_MODEL_FILE_PATH.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as model_file:
        fieldnames = [
            "reward_mode",
            "gamma",
            "epsilon_schedule",
            "seed",
            "row",
            "column",
            "has_key",
            "q_up",
            "q_down",
            "q_left",
            "q_right",
            "max_q",
            "best_action",
            "is_terminal"
        ]

        writer = csv.DictWriter(
            model_file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for result_key, result in (
            experiment_results.items()
        ):
            (
                reward_mode,
                gamma,
                epsilon_schedule,
                seed
            ) = result_key

            q_table = result["q_table"]
            policy = result["policy"]

            for state in valid_states:
                row, column, has_key = state
                action_values = q_table[state]

                writer.writerow({
                    "reward_mode": reward_mode,
                    "gamma": gamma,
                    "epsilon_schedule": (
                        epsilon_schedule
                    ),
                    "seed": seed,
                    "row": row,
                    "column": column,
                    "has_key": has_key,
                    "q_up": action_values["up"],
                    "q_down": (
                        action_values["down"]
                    ),
                    "q_left": (
                        action_values["left"]
                    ),
                    "q_right": (
                        action_values["right"]
                    ),
                    "max_q": max(
                        action_values.values()
                    ),
                    "best_action": policy[state],
                    "is_terminal": (
                        environment.is_terminal_state(
                            state
                        )
                    )
                })

    with Q_LEARNING_UPDATE_LOG_FILE_PATH.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as update_log_file:
        fieldnames = [
            "reward_mode",
            "gamma",
            "epsilon_schedule",
            "seed",
            "episode",
            "step",
            "state_row",
            "state_column",
            "has_key_before",
            "selected_action",
            "executed_action",
            "reward",
            "next_row",
            "next_column",
            "has_key_after",
            "done",
            "event",
            "epsilon",
            "alpha",
            "old_q",
            "next_max_q",
            "td_target",
            "td_error",
            "new_q"
        ]

        writer = csv.DictWriter(
            update_log_file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for result_key, result in (
            experiment_results.items()
        ):
            (
                reward_mode,
                gamma,
                epsilon_schedule,
                seed
            ) = result_key

            for update_record in (
                result["update_records"]
            ):
                writer.writerow({
                    "reward_mode": reward_mode,
                    "gamma": gamma,
                    "epsilon_schedule": (
                        epsilon_schedule
                    ),
                    "seed": seed,
                    "episode": (
                        update_record["episode"]
                    ),
                    "step": update_record["step"],
                    "state_row": (
                        update_record["state_row"]
                    ),
                    "state_column": (
                        update_record[
                            "state_column"
                        ]
                    ),
                    "has_key_before": (
                        update_record[
                            "has_key_before"
                        ]
                    ),
                    "selected_action": (
                        update_record[
                            "selected_action"
                        ]
                    ),
                    "executed_action": (
                        update_record[
                            "executed_action"
                        ]
                    ),
                    "reward": (
                        update_record["reward"]
                    ),
                    "next_row": (
                        update_record["next_row"]
                    ),
                    "next_column": (
                        update_record[
                            "next_column"
                        ]
                    ),
                    "has_key_after": (
                        update_record[
                            "has_key_after"
                        ]
                    ),
                    "done": update_record["done"],
                    "event": update_record["event"],
                    "epsilon": (
                        update_record["epsilon"]
                    ),
                    "alpha": (
                        update_record["alpha"]
                    ),
                    "old_q": (
                        update_record["old_q"]
                    ),
                    "next_max_q": (
                        update_record[
                            "next_max_q"
                        ]
                    ),
                    "td_target": (
                        update_record[
                            "td_target"
                        ]
                    ),
                    "td_error": (
                        update_record["td_error"]
                    ),
                    "new_q": (
                        update_record["new_q"]
                    )
                })

    with Q_LEARNING_SUMMARY_FILE_PATH.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as summary_file:
        fieldnames = [
            "reward_mode",
            "gamma",
            "epsilon_schedule",
            "seed",
            "alpha",
            "episodes",
            "execution_time",
            "final_epsilon",
            "successful_episodes",
            "overall_success_rate",
            "final_window_success_rate",
            "mean_reward",
            "mean_reward_last_window",
            "mean_steps",
            "mean_steps_last_window",
            "total_wall_collisions",
            "total_penalty_entries",
            "state_count",
        ]

        writer = csv.DictWriter(
            summary_file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for result_key, result in (
            experiment_results.items()
        ):
            (
                reward_mode,
                gamma,
                epsilon_schedule,
                seed
            ) = result_key

            episode_records = (
                result["episode_records"]
            )

            recent_records = episode_records[
                -Q_LEARNING_SUCCESS_WINDOW:
            ]

            successful_episodes = sum(
                1
                for record in episode_records
                if record["success"]
            )

            overall_success_rate = (
                successful_episodes
                / len(episode_records)
            )

            final_window_success_rate = (
                sum(
                    1
                    for record in recent_records
                    if record["success"]
                )
                / len(recent_records)
            )

            mean_reward = (
                sum(
                    record["total_reward"]
                    for record in episode_records
                )
                / len(episode_records)
            )

            mean_reward_last_window = (
                sum(
                    record["total_reward"]
                    for record in recent_records
                )
                / len(recent_records)
            )

            mean_steps = (
                sum(
                    record["steps"]
                    for record in episode_records
                )
                / len(episode_records)
            )

            mean_steps_last_window = (
                sum(
                    record["steps"]
                    for record in recent_records
                )
                / len(recent_records)
            )

            total_wall_collisions = sum(
                record["wall_collisions"]
                for record in episode_records
            )

            total_penalty_entries = sum(
                record["penalty_entries"]
                for record in episode_records
            )

            writer.writerow({
                "reward_mode": reward_mode,
                "gamma": gamma,
                "epsilon_schedule": (
                    epsilon_schedule
                ),
                "seed": seed,
                "alpha": Q_LEARNING_ALPHA,
                "episodes": len(
                    episode_records
                ),
                "execution_time": (
                    result["execution_time"]
                ),
                "final_epsilon": (
                    episode_records[-1]["epsilon"]
                ),
                "successful_episodes": (
                    successful_episodes
                ),
                "overall_success_rate": (
                    overall_success_rate
                ),
                "final_window_success_rate": (
                    final_window_success_rate
                ),
                "mean_reward": mean_reward,
                "mean_reward_last_window": (
                    mean_reward_last_window
                ),
                "mean_steps": mean_steps,
                "mean_steps_last_window": (
                    mean_steps_last_window
                ),
                "total_wall_collisions": (
                    total_wall_collisions
                ),
                "total_penalty_entries": (
                    total_penalty_entries
                ),
                "state_count": len(
                    result["q_table"]
                )
            })

    example_result = next(
        iter(experiment_results.values())
    )

    actual_number_of_episodes = (
        example_result["number_of_episodes"]
    )

    actual_update_log_episode = min(
        Q_LEARNING_UPDATE_LOG_EPISODE,
        actual_number_of_episodes
    )

    config_data = {
        "algorithm": "Q-Learning",
        "learning_type": "off-policy",
        "behavior_policy": "epsilon-greedy",
        "student_id": STUDENT_ID,
        "base_seed": BASE_SEED,
        "maze_size": MAZE_SIZE,
        "map_file": (
            MAP_FILE_PATH
            .relative_to(PROJECT_ROOT)
            .as_posix()
        ),
        "reward_modes": REWARD_MODES,
        "alpha": Q_LEARNING_ALPHA,
        "gammas": Q_LEARNING_GAMMAS,
        "episodes": actual_number_of_episodes,
        "epsilon_start": (
            Q_LEARNING_EPSILON_START
        ),
        "epsilon_end": (
            Q_LEARNING_EPSILON_END
        ),
        "epsilon_decay_episodes": (
            Q_LEARNING_EPSILON_DECAY_EPISODES
        ),
        "epsilon_schedules": (
            Q_LEARNING_EPSILON_SCHEDULES
        ),
        "success_window": (
            Q_LEARNING_SUCCESS_WINDOW
        ),
        "training_seeds": Q_LEARNING_SEEDS,
        "update_log_reward_mode": "sparse",
        "update_log_gamma": (
            VALUE_ITERATION_GAMMA
        ),
        "update_log_epsilon_schedule": (
            "linear"
        ),
        "update_log_seed": BASE_SEED,
        "update_log_episode": (
            actual_update_log_episode
        ),
        "state_count": len(
            example_result["q_table"]
        ),
         "configuration_count": (
            len(REWARD_MODES)
            * len(Q_LEARNING_GAMMAS)
            * len(Q_LEARNING_EPSILON_SCHEDULES)
        ),
        "runs_per_configuration": len(
            Q_LEARNING_SEEDS
        ),
        "training_run_count": len(
            experiment_results
        )
    }

    with Q_LEARNING_CONFIG_FILE_PATH.open(
        "w",
        encoding="utf-8"
    ) as config_file:
        json.dump(
            config_data,
            config_file,
            indent=4
        )

    return {
        "episodes": Q_LEARNING_EPISODE_FILE_PATH,
        "models": Q_LEARNING_MODEL_FILE_PATH,
        "updates": Q_LEARNING_UPDATE_LOG_FILE_PATH,
        "summary": Q_LEARNING_SUMMARY_FILE_PATH,
        "config": Q_LEARNING_CONFIG_FILE_PATH
    }

if __name__ == "__main__":
    run_value_iteration_experiments()
    run_q_learning_experiments()