import random
import time

from agents.q_learning import calculate_epsilon
from agents.q_learning import initialize_q_table
from agents.q_learning import select_epsilon_greedy_action

from config import ACTIONS

def initialize_eligibility_traces():
    return {}


def get_eligibility_trace(
    eligibility_traces,
    state,
    action
):
    state_action = (
        state,
        action
    )

    return eligibility_traces.get(
        state_action,
        0.0
    )


def set_replacing_trace(
    eligibility_traces,
    state,
    action
):
    traces_to_remove = []

    for state_action in eligibility_traces:
        trace_state, trace_action = (
            state_action
        )

        if (
            trace_state == state
            and trace_action != action
        ):
            traces_to_remove.append(
                state_action
            )

    for state_action in traces_to_remove:
        del eligibility_traces[state_action]

    current_state_action = (
        state,
        action
    )

    eligibility_traces[
        current_state_action
    ] = 1.0

def decay_eligibility_traces(
    eligibility_traces,
    gamma,
    lambda_value,
    trace_threshold
):
    traces_to_remove = []

    for state_action in eligibility_traces:
        eligibility_traces[state_action] *= (
            gamma * lambda_value
        )

        if (
            eligibility_traces[state_action]
            < trace_threshold
        ):
            traces_to_remove.append(
                state_action
            )

    for state_action in traces_to_remove:
        del eligibility_traces[state_action]

def calculate_sarsa_td_error(
    q_table,
    state,
    action,
    reward,
    next_state,
    next_action,
    done,
    gamma
):
    old_q = q_table[state][action]

    next_q = 0.0

    if not done:
        next_q = q_table[
            next_state
        ][next_action]

    td_target = reward

    if not done:
        td_target += gamma * next_q

    td_error = td_target - old_q

    return {
        "old_q": old_q,
        "next_q": next_q,
        "td_target": td_target,
        "td_error": td_error
    }

def update_q_values_with_traces(
    q_table,
    eligibility_traces,
    alpha,
    td_error,
    record_trace_updates=False
):
    trace_update_records = []

    for state_action, eligibility in (
        eligibility_traces.items()
    ):
        trace_state, trace_action = (
            state_action
        )

        old_q = q_table[
            trace_state
        ][trace_action]

        update_amount = (
            alpha
            * td_error
            * eligibility
        )

        new_q = old_q + update_amount

        q_table[
            trace_state
        ][trace_action] = new_q

        if record_trace_updates:
            trace_update_records.append({
                "trace_state": trace_state,
                "trace_action": trace_action,
                "eligibility": eligibility,
                "old_q": old_q,
                "update_amount": update_amount,
                "new_q": new_q
            })

    return trace_update_records

def train_sarsa_lambda_episode(
    environment,
    q_table,
    epsilon,
    alpha,
    gamma,
    lambda_value,
    trace_threshold,
    random_generator,
    record_trace_updates=False
):
    eligibility_traces = (
        initialize_eligibility_traces()
    )

    state = environment.reset()

    action = select_epsilon_greedy_action(
        q_table=q_table,
        state=state,
        epsilon=epsilon,
        random_generator=random_generator
    )

    total_reward = 0.0
    steps = 0
    success = False
    wall_collisions = 0
    penalty_entries = 0
    final_event = None

    trace_update_records = []

    done = False

    while not done:
        steps += 1

        (
            next_state,
            reward,
            done,
            event,
            executed_action
        ) = environment.move(action)

        total_reward += reward
        final_event = event

        if event == "goal_reached":
            success = True

        if event == "wall_collision":
            wall_collisions += 1

        if event == "penalty_cell":
            penalty_entries += 1

        next_action = None

        if not done:
            next_action = (
                select_epsilon_greedy_action(
                    q_table=q_table,
                    state=next_state,
                    epsilon=epsilon,
                    random_generator=(
                        random_generator
                    )
                )
            )

        td_values = calculate_sarsa_td_error(
            q_table=q_table,
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            next_action=next_action,
            done=done,
            gamma=gamma
        )

        set_replacing_trace(
            eligibility_traces=(
                eligibility_traces
            ),
            state=state,
            action=action
        )

        active_traces_before_decay = len(
            eligibility_traces
        )

        step_trace_records = (
            update_q_values_with_traces(
                q_table=q_table,
                eligibility_traces=(
                    eligibility_traces
                ),
                alpha=alpha,
                td_error=(
                    td_values["td_error"]
                ),
                record_trace_updates=(
                    record_trace_updates
                )
            )
        )

        decay_eligibility_traces(
            eligibility_traces=(
                eligibility_traces
            ),
            gamma=gamma,
            lambda_value=lambda_value,
            trace_threshold=trace_threshold
        )

        active_traces_after_decay = len(
            eligibility_traces
        )

        if record_trace_updates:
            for trace_record in (
                step_trace_records
            ):
                trace_state = trace_record[
                    "trace_state"
                ]

                trace_action = trace_record[
                    "trace_action"
                ]

                trace_record.update({
                    "step": steps,
                    "state": state,
                    "selected_action": action,
                    "executed_action": (
                        executed_action
                    ),
                    "reward": reward,
                    "next_state": next_state,
                    "next_action": next_action,
                    "done": done,
                    "event": event,
                    "epsilon": epsilon,
                    "alpha": alpha,
                    "gamma": gamma,
                    "lambda": lambda_value,
                    "current_old_q": (
                        td_values["old_q"]
                    ),
                    "next_action_q": (
                        td_values["next_q"]
                    ),
                    "td_target": (
                        td_values["td_target"]
                    ),
                    "td_error": (
                        td_values["td_error"]
                    ),
                    "eligibility_after_decay": (
                        get_eligibility_trace(
                            eligibility_traces,
                            trace_state,
                            trace_action
                        )
                    ),
                    "active_traces_before_decay": (
                        active_traces_before_decay
                    ),
                    "active_traces_after_decay": (
                        active_traces_after_decay
                    )
                })

                trace_update_records.append(
                    trace_record
                )

        if not done:
            state = next_state
            action = next_action

    return {
        "total_reward": total_reward,
        "steps": steps,
        "success": success,
        "wall_collisions": wall_collisions,
        "penalty_entries": penalty_entries,
        "final_event": final_event,
        "trace_update_records": (
            trace_update_records
        )
    }

def train_sarsa_lambda(
    environment,
    number_of_episodes,
    alpha,
    gamma,
    lambda_value,
    epsilon_schedule,
    epsilon_start,
    epsilon_end,
    epsilon_decay_episodes,
    success_window,
    trace_threshold,
    seed,
    record_trace_episode=None
):
    q_table = initialize_q_table(
        environment
    )

    random_generator = random.Random(seed)

    episode_records = []
    recent_successes = []
    trace_update_records = []

    start_time = time.perf_counter()

    for episode_index in range(
        number_of_episodes
    ):
        episode_number = episode_index + 1

        epsilon = calculate_epsilon(
            episode_index=episode_index,
            epsilon_schedule=(
                epsilon_schedule
            ),
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            decay_episodes=(
                epsilon_decay_episodes
            )
        )

        should_record_trace = (
            record_trace_episode is not None
            and episode_number
            == record_trace_episode
        )

        episode_result = (
            train_sarsa_lambda_episode(
                environment=environment,
                q_table=q_table,
                epsilon=epsilon,
                alpha=alpha,
                gamma=gamma,
                lambda_value=lambda_value,
                trace_threshold=(
                    trace_threshold
                ),
                random_generator=(
                    random_generator
                ),
                record_trace_updates=(
                    should_record_trace
                )
            )
        )

        success_value = int(
            episode_result["success"]
        )

        recent_successes.append(
            success_value
        )

        if (
            len(recent_successes)
            > success_window
        ):
            recent_successes.pop(0)

        success_rate = (
            sum(recent_successes)
            / len(recent_successes)
        )

        episode_records.append({
            "episode": episode_number,
            "alpha": alpha,
            "gamma": gamma,
            "lambda": lambda_value,
            "epsilon": epsilon,
            "total_reward": (
                episode_result[
                    "total_reward"
                ]
            ),
            "steps": episode_result["steps"],
            "success": (
                episode_result["success"]
            ),
            "success_rate": success_rate,
            "wall_collisions": (
                episode_result[
                    "wall_collisions"
                ]
            ),
            "penalty_entries": (
                episode_result[
                    "penalty_entries"
                ]
            ),
            "final_event": (
                episode_result[
                    "final_event"
                ]
            )
        })

        if should_record_trace:
            for trace_record in (
                episode_result[
                    "trace_update_records"
                ]
            ):
                trace_record["episode"] = (
                    episode_number
                )

                trace_record["seed"] = seed

                trace_record[
                    "epsilon_schedule"
                ] = epsilon_schedule

                trace_update_records.append(
                    trace_record
                )

    execution_time = (
        time.perf_counter() - start_time
    )

    return {
        "q_table": q_table,
        "episode_records": episode_records,
        "trace_update_records": (
            trace_update_records
        ),
        "execution_time": execution_time
    }

def extract_sarsa_lambda_policy(
    environment,
    q_table
):
    policy = {}

    for state, action_values in (
        q_table.items()
    ):
        if environment.is_terminal_state(
            state
        ):
            policy[state] = None
            continue

        best_action = ACTIONS[0]

        best_q_value = action_values[
            best_action
        ]

        for action in ACTIONS[1:]:
            q_value = action_values[action]

            if q_value > best_q_value:
                best_action = action
                best_q_value = q_value

        policy[state] = best_action

    return policy