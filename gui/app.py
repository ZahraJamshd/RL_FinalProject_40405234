import csv
import sys
import tkinter as tk

from pathlib import Path
from tkinter import messagebox
from tkinter import ttk


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


from config import BASE_SEED
from config import SARSA_LAMBDA_EPSILON_SCHEDULE
from config import SARSA_LAMBDA_GAMMA
from config import VALUE_ITERATION_GAMMA

from environments.maze import MazeEnvironment

from renderer import CELL_SIZE
from renderer import draw_maze


MAP_FILE_PATH = (
    PROJECT_ROOT
    / "environments"
    / "maps"
    / f"maze_seed_{BASE_SEED}.txt"
)

VALUE_MODELS_FILE_PATH = (
    PROJECT_ROOT
    / "results"
    / "models"
    / "value_iteration_models.csv"
)

Q_LEARNING_MODEL_FILE_PATH = (
    PROJECT_ROOT
    / "results"
    / "models"
    / "q_learning_models.csv"
)

SARSA_LAMBDA_MODEL_FILE_PATH = (
    PROJECT_ROOT
    / "results"
    / "models"
    / "sarsa_lambda_models.csv"
)

SELECTED_Q_GAMMA = VALUE_ITERATION_GAMMA
SELECTED_Q_EPSILON_SCHEDULE = "exponential"
SELECTED_Q_SEED = BASE_SEED

SELECTED_SARSA_LAMBDA = 0.3
SELECTED_SARSA_SEED = BASE_SEED
ANIMATION_DELAY = 250

class MazeApplication:
    def __init__(self, root):
        self.root = root

        self.root.title(
            "Reinforcement Learning Maze"
        )

        self.root.resizable(
            False,
            False
        )

        self.environment = MazeEnvironment(
            map_file_path=MAP_FILE_PATH,
            seed=BASE_SEED,
            reward_mode="sparse"
        )

        self.policy = {}
        self.current_event = "ready"

        self.is_running = False
        self.animation_job = None

        self.algorithm_value = tk.StringVar(
            value="Value Iteration"
        )

        self.reward_mode_value = tk.StringVar(
            value="sparse"
        )

        self.configuration_text = tk.StringVar()
        self.position_text = tk.StringVar()
        self.key_text = tk.StringVar()
        self.step_text = tk.StringVar()
        self.reward_text = tk.StringVar()
        self.action_text = tk.StringVar()
        self.event_text = tk.StringVar()

        self.create_widgets()
        self.load_policy()

    def create_widgets(self):
        main_frame = ttk.Frame(
            self.root,
            padding=10
        )

        main_frame.pack()

        maze_height = len(
            self.environment.maze
        )

        maze_width = len(
            self.environment.maze[0]
        )

        self.canvas = tk.Canvas(
            main_frame,
            width=maze_width * CELL_SIZE,
            height=maze_height * CELL_SIZE,
            highlightthickness=1
        )

        self.canvas.grid(
            row=0,
            column=0,
            sticky="n"
        )

        information_frame = ttk.Frame(
            main_frame,
            padding=(15, 0, 0, 0)
        )

        information_frame.grid(
            row=0,
            column=1,
            sticky="nw"
        )

        ttk.Label(
            information_frame,
            text="Policy Selection",
            font=("Arial", 13, "bold")
        ).pack(
            anchor="w",
            pady=(0, 8)
        )

        ttk.Label(
            information_frame,
            text="Algorithm"
        ).pack(
            anchor="w"
        )

        algorithm_box = ttk.Combobox(
            information_frame,
            textvariable=(
                self.algorithm_value
            ),
            values=[
                "Value Iteration",
                "Q-Learning",
                "SARSA(lambda)"
            ],
            state="readonly",
            width=24
        )

        algorithm_box.pack(
            fill="x",
            pady=(2, 8)
        )

        ttk.Label(
            information_frame,
            text="Reward mode"
        ).pack(
            anchor="w"
        )

        reward_mode_box = ttk.Combobox(
            information_frame,
            textvariable=(
                self.reward_mode_value
            ),
            values=[
                "sparse",
                "shaped"
            ],
            state="readonly",
            width=24
        )

        reward_mode_box.pack(
            fill="x",
            pady=(2, 8)
        )

        ttk.Button(
            information_frame,
            text="Load Policy",
            command=self.load_policy
        ).pack(
            fill="x"
        )

        ttk.Label(
            information_frame,
            textvariable=(
                self.configuration_text
            ),
            wraplength=220
        ).pack(
            anchor="w",
            pady=(8, 0)
        )

        ttk.Separator(
            information_frame,
            orient="horizontal"
        ).pack(
            fill="x",
            pady=10
        )

        ttk.Label(
            information_frame,
            text="Environment Status",
            font=("Arial", 12, "bold")
        ).pack(
            anchor="w",
            pady=(0, 6)
        )

        ttk.Label(
            information_frame,
            textvariable=self.position_text
        ).pack(
            anchor="w",
            pady=1
        )

        ttk.Label(
            information_frame,
            textvariable=self.key_text
        ).pack(
            anchor="w",
            pady=1
        )

        ttk.Label(
            information_frame,
            textvariable=self.step_text
        ).pack(
            anchor="w",
            pady=1
        )

        ttk.Label(
            information_frame,
            textvariable=self.reward_text
        ).pack(
            anchor="w",
            pady=1
        )

        ttk.Label(
            information_frame,
            textvariable=self.action_text
        ).pack(
            anchor="w",
            pady=1
        )

        ttk.Label(
            information_frame,
            textvariable=self.event_text,
            wraplength=220
        ).pack(
            anchor="w",
            pady=1
        )

        ttk.Separator(
            information_frame,
            orient="horizontal"
        ).pack(
            fill="x",
            pady=10
        )

        ttk.Label(
            information_frame,
            text=(
                "S: Start   K: Key   D: Door\n"
                "G: Goal   P: Penalty   "
                "T: Teleporter"
            ),
            wraplength=220
        ).pack(
            anchor="w"
        )

        control_frame = ttk.Frame(
            information_frame
        )

        control_frame.pack(
            fill="x",
            pady=(12, 4)
        )

        ttk.Button(
            control_frame,
            text="Start",
            command=self.start_episode
        ).pack(
            side="left",
            expand=True,
            fill="x",
            padx=(0, 2)
        )

        ttk.Button(
            control_frame,
            text="Pause",
            command=self.pause_episode
        ).pack(
            side="left",
            expand=True,
            fill="x",
            padx=(2, 0)
        )

        ttk.Button(
            information_frame,
            text="Step",
            command=self.run_single_step
        ).pack(
            fill="x",
            pady=(0, 4)
        )

        ttk.Button(
            information_frame,
            text="Reset Environment",
            command=self.reset_environment
        ).pack(
            fill="x"
        )

    def load_policy(self):

        self.stop_animation()

        algorithm = self.algorithm_value.get()
        reward_mode = (
            self.reward_mode_value.get()
        )

        self.environment = MazeEnvironment(
            map_file_path=MAP_FILE_PATH,
            seed=BASE_SEED,
            reward_mode=reward_mode
        )

        self.policy = {}

        if algorithm == "Value Iteration":
            model_file_path = (
                VALUE_MODELS_FILE_PATH
            )

            configuration_description = (
                f"gamma = "
                f"{VALUE_ITERATION_GAMMA}"
            )

        elif algorithm == "Q-Learning":
            model_file_path = (
                Q_LEARNING_MODEL_FILE_PATH
            )

            configuration_description = (
                f"gamma = {SELECTED_Q_GAMMA}, "
                f"epsilon = "
                f"{SELECTED_Q_EPSILON_SCHEDULE}, "
                f"seed = {SELECTED_Q_SEED}"
            )

        else:
            model_file_path = (
                SARSA_LAMBDA_MODEL_FILE_PATH
            )

            configuration_description = (
                f"gamma = {SARSA_LAMBDA_GAMMA}, "
                f"lambda = "
                f"{SELECTED_SARSA_LAMBDA}, "
                f"epsilon = "
                f"{SARSA_LAMBDA_EPSILON_SCHEDULE}, "
                f"seed = {SELECTED_SARSA_SEED}"
            )

        with model_file_path.open(
            "r",
            encoding="utf-8"
        ) as model_file:
            reader = csv.DictReader(
                model_file
            )

            for row in reader:
                if (
                    row["reward_mode"]
                    != reward_mode
                ):
                    continue

                if algorithm == "Value Iteration":
                    if abs(
                        float(row["gamma"])
                        - VALUE_ITERATION_GAMMA
                    ) > 0.000000000001:
                        continue

                elif algorithm == "Q-Learning":
                    if abs(
                        float(row["gamma"])
                        - SELECTED_Q_GAMMA
                    ) > 0.000000000001:
                        continue

                    if (
                        row["epsilon_schedule"]
                        != (
                            SELECTED_Q_EPSILON_SCHEDULE
                        )
                    ):
                        continue

                    if (
                        int(row["seed"])
                        != SELECTED_Q_SEED
                    ):
                        continue

                else:
                    if abs(
                        float(row["gamma"])
                        - SARSA_LAMBDA_GAMMA
                    ) > 0.000000000001:
                        continue

                    if abs(
                        float(row["lambda"])
                        - SELECTED_SARSA_LAMBDA
                    ) > 0.000000000001:
                        continue

                    if (
                        row["epsilon_schedule"]
                        != (
                            SARSA_LAMBDA_EPSILON_SCHEDULE
                        )
                    ):
                        continue

                    if (
                        int(row["seed"])
                        != SELECTED_SARSA_SEED
                    ):
                        continue

                state = (
                    int(row["row"]),
                    int(row["column"]),
                    int(row["has_key"])
                )

                best_action = (
                    row["best_action"].strip()
                    or None
                )

                self.policy[state] = best_action

        if not self.policy:
            messagebox.showerror(
                "Policy Error",
                "No matching policy was found."
            )

            return

        self.current_event = (
            f"{algorithm} policy loaded"
        )

        self.configuration_text.set(
            "Configuration: "
            f"{configuration_description}"
        )

        self.update_display()

    def perform_policy_step(self):
        if self.environment.done:
            self.current_event = (
                "episode already finished"
            )

            self.update_display()
            return False

        current_state = (
            self.environment.get_state()
        )

        selected_action = self.policy.get(
            current_state
        )

        if selected_action is None:
            self.current_event = (
                "no action for current state"
            )

            self.update_display()
            return False

        (
            next_state,
            reward,
            done,
            event,
            executed_action
        ) = self.environment.move(
            selected_action
        )

        self.current_event = (
            f"{event} | selected: "
            f"{selected_action} | executed: "
            f"{executed_action}"
        )

        self.update_display()

        return True

    def run_single_step(self):
        self.stop_animation()
        self.perform_policy_step()

    def start_episode(self):
        if self.is_running:
            return

        if self.environment.done:
            self.environment.reset()

        self.is_running = True
        self.current_event = "running"
        self.update_display()
        self.run_automatic_step()

    def run_automatic_step(self):
        self.animation_job = None

        if not self.is_running:
            return

        step_completed = (
            self.perform_policy_step()
        )

        if (
            not step_completed
            or self.environment.done
        ):
            self.is_running = False
            return

        self.animation_job = self.root.after(
            ANIMATION_DELAY,
            self.run_automatic_step
        )

    def pause_episode(self):
        if not self.is_running:
            return

        self.stop_animation()
        self.current_event = "paused"
        self.update_display()

    def stop_animation(self):
        self.is_running = False

        if self.animation_job is not None:
            self.root.after_cancel(
                self.animation_job
            )

            self.animation_job = None

    def reset_environment(self):
        self.stop_animation()
        self.environment.reset()
        self.current_event = "ready"
        self.update_display()

    def update_display(self):
        draw_maze(
            canvas=self.canvas,
            maze=self.environment.maze,
            agent_position=(
                self.environment.agent_position
            ),
            has_key=self.environment.has_key
        )

        current_state = (
            self.environment.get_state()
        )

        policy_action = self.policy.get(
            current_state
        )

        self.position_text.set(
            "Position: "
            f"{self.environment.agent_position}"
        )

        self.key_text.set(
            "Has key: "
            f"{self.environment.has_key}"
        )

        self.step_text.set(
            "Steps: "
            f"{self.environment.step_count}"
            f" / {self.environment.max_steps}"
        )

        self.reward_text.set(
            "Total reward: "
            f"{self.environment.total_reward:.2f}"
        )

        self.action_text.set(
            "Policy action: "
            f"{policy_action}"
        )

        self.event_text.set(
            "Current event: "
            f"{self.current_event}"
        )


if __name__ == "__main__":
    root = tk.Tk()

    application = MazeApplication(root)

    root.mainloop()