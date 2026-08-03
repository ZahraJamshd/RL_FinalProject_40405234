import random

from config import ACTIONS


def initialize_q_table(environment):
    q_table = {}

    valid_states = environment.get_valid_states()

    for state in valid_states:
        q_table[state] = {}

        for action in ACTIONS:
            q_table[state][action] = 0.0

    return q_table

def select_epsilon_greedy_action(
    q_table,
    state,
    epsilon,
    random_generator
):
    if epsilon < 0.0 or epsilon > 1.0:
        raise ValueError(
            "Epsilon must be between 0 and 1."
        )

    random_number = random_generator.random()

    if random_number < epsilon:
        return random_generator.choice(ACTIONS)

    action_values = q_table[state]
    maximum_q_value = max(action_values.values())

    best_actions = []

    for action, q_value in action_values.items():
        if q_value == maximum_q_value:
            best_actions.append(action)

    return random_generator.choice(best_actions)

def calculate_linear_epsilon(
    episode_index,
    epsilon_start,
    epsilon_end,
    decay_episodes
):
    if decay_episodes <= 0:
        raise ValueError(
            "Decay episodes must be greater than zero."
        )

    if episode_index <= 0:
        return epsilon_start

    if episode_index >= decay_episodes:
        return epsilon_end

    limited_episode = max(
        0,
        min(episode_index, decay_episodes)
    )

    progress = limited_episode / decay_episodes

    epsilon = (
        epsilon_start
        + (epsilon_end - epsilon_start) * progress
    )

    return max(epsilon_end, epsilon)


def calculate_exponential_epsilon(
    episode_index,
    epsilon_start,
    epsilon_end,
    decay_episodes
):
    if decay_episodes <= 0:
        raise ValueError(
            "Decay episodes must be greater than zero."
        )

    if epsilon_start <= 0.0 or epsilon_end <= 0.0:
        raise ValueError(
            "Epsilon values must be greater than zero."
        )

    if episode_index <= 0:
        return epsilon_start

    if episode_index >= decay_episodes:
        return epsilon_end

    limited_episode = max(
        0,
        min(episode_index, decay_episodes)
    )

    progress = limited_episode / decay_episodes

    decay_ratio = epsilon_end / epsilon_start

    epsilon = epsilon_start * (
        decay_ratio ** progress
    )

    return max(epsilon_end, epsilon)

def calculate_epsilon(
    episode_index,
    epsilon_schedule,
    epsilon_start,
    epsilon_end,
    decay_episodes
):
    if epsilon_schedule == "linear":
        return calculate_linear_epsilon(
            episode_index=episode_index,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            decay_episodes=decay_episodes
        )

    if epsilon_schedule == "exponential":
        return calculate_exponential_epsilon(
            episode_index=episode_index,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            decay_episodes=decay_episodes
        )

    raise ValueError(
        f"Unknown epsilon schedule: {epsilon_schedule}"
    )

def update_q_value(
    q_table,
    state,
    action,
    reward,
    next_state,
    done,
    alpha,
    gamma
):
    old_q = q_table[state][action]

    if done:
        next_max_q = 0.0
    else:
        next_max_q = max(
            q_table[next_state].values()
        )

    td_target = (
        reward
        + gamma * next_max_q
    )

    td_error = td_target - old_q

    new_q = (
        old_q
        + alpha * td_error
    )

    q_table[state][action] = new_q

    update_details = {
        "old_q": old_q,
        "next_max_q": next_max_q,
        "td_target": td_target,
        "td_error": td_error,
        "new_q": new_q
    }

    return update_details

def train_q_learning_episode(
    environment,
    q_table,
    epsilon,
    alpha,
    gamma,
    random_generator,
    record_updates=False,
):
    state = environment.reset()

    total_reward = 0.0
    steps = 0
    wall_collisions = 0
    penalty_entries = 0
    success = False
    final_event = None

    update_records = []

    while True:
        selected_action = (
            select_epsilon_greedy_action(
                q_table=q_table,
                state=state,
                epsilon=epsilon,
                random_generator=random_generator
            )
        )

        (
            next_state,
            reward,
            done,
            event,
            executed_action
        ) = environment.move(selected_action)

        update_details = update_q_value(
            q_table=q_table,
            state=state,
            action=selected_action,
            reward=reward,
            next_state=next_state,
            done=done,
            alpha=alpha,
            gamma=gamma
        )

        if record_updates:
            update_record = {
                "step": steps + 1,
                "state_row": state[0],
                "state_column": state[1],
                "has_key_before": state[2],
                "selected_action": selected_action,
                "executed_action": executed_action,
                "reward": reward,
                "next_row": next_state[0],
                "next_column": next_state[1],
                "has_key_after": next_state[2],
                "done": done,
                "event": event,
                "epsilon": epsilon,
                "alpha": alpha,
                "gamma": gamma,
                "old_q": update_details["old_q"],
                "next_max_q": (
                    update_details["next_max_q"]
                ),
                "td_target": (
                    update_details["td_target"]
                ),
                "td_error": (
                    update_details["td_error"]
                ),
                "new_q": update_details["new_q"]
            }

            update_records.append(update_record)

        total_reward += reward
        steps += 1
        final_event = event

        if event == "wall_collision":
            wall_collisions += 1

        if event == "penalty_cell":
            penalty_entries += 1

        if event == "goal_reached":
            success = True

        state = next_state

        if done:
            break

    episode_result = {
        "total_reward": total_reward,
        "steps": steps,
        "success": success,
        "wall_collisions": wall_collisions,
        "penalty_entries": penalty_entries,
        "final_event": final_event,
        "update_records": update_records
    }

    return episode_result

def train_q_learning(
    environment,
    number_of_episodes,
    alpha,
    gamma,
    epsilon_schedule,
    epsilon_start,
    epsilon_end,
    epsilon_decay_episodes,
    success_window,
    seed,
    record_update_episode=None
):
    q_table = initialize_q_table(environment)

    random_generator = random.Random(seed)

    episode_records = []
    recent_successes = []
    recorded_updates = []

    for episode_index in range(number_of_episodes):
        epsilon = calculate_epsilon(
            episode_index=episode_index,
            epsilon_schedule=epsilon_schedule,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            decay_episodes=epsilon_decay_episodes
        )

        should_record_episode_updates = (
            record_update_episode is not None
            and episode_index + 1
            == record_update_episode
        )

        episode_result = train_q_learning_episode(
            environment=environment,
            q_table=q_table,
            epsilon=epsilon,
            alpha=alpha,
            gamma=gamma,
            random_generator=random_generator,
            record_updates=should_record_episode_updates
        )

        if should_record_episode_updates:
            for update_record in (
                episode_result["update_records"]
            ):
                update_record["episode"] = (
                    episode_index + 1
                )
                update_record["seed"] = seed
                update_record["epsilon_schedule"] = (
                    epsilon_schedule
                )

                recorded_updates.append(
                    update_record
                )

        if episode_result["success"]:
            recent_successes.append(1)
        else:
            recent_successes.append(0)

        if len(recent_successes) > success_window:
            recent_successes.pop(0)

        success_rate = (
            sum(recent_successes)
            / len(recent_successes)
        )

        episode_record = {
            "episode": episode_index + 1,
            "seed": seed,
            "alpha": alpha,
            "gamma": gamma,
            "epsilon_schedule": epsilon_schedule,
            "epsilon": epsilon,
            "total_reward": (
                episode_result["total_reward"]
            ),
            "steps": episode_result["steps"],
            "success": episode_result["success"],
            "success_rate": success_rate,
            "wall_collisions": (
                episode_result["wall_collisions"]
            ),
            "penalty_entries": (
                episode_result["penalty_entries"]
            ),
            "final_event": (
                episode_result["final_event"]
            )
        }

        episode_records.append(episode_record)

    training_result = {
        "q_table": q_table,
        "episode_records": episode_records,
        "update_records": recorded_updates,
    }

    return training_result

def extract_q_learning_policy(
    environment,
    q_table
):
    policy = {}

    for state, action_values in q_table.items():
        if environment.is_terminal_state(state):
            policy[state] = None
            continue

        best_action = ACTIONS[0]
        best_q_value = action_values[best_action]

        for action in ACTIONS[1:]:
            q_value = action_values[action]

            if q_value > best_q_value:
                best_action = action
                best_q_value = q_value

        policy[state] = best_action

    return policy
