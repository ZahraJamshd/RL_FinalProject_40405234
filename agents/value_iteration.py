from time import perf_counter

def initialize_values(environment):
    values = {}

    valid_states = environment.get_valid_states()

    for state in valid_states:
        values[state] = 0.0

    return values

def calculate_action_value(
    environment,
    state,
    action,
    values,
    gamma
):
    if not 0 <= gamma < 1:
        raise ValueError(
            "Gamma must be between 0 and 1."
        )

    transition_outcomes = (
        environment.get_transition_outcomes(
            state=state,
            selected_action=action
        )
    )

    action_value = 0.0

    for outcome in transition_outcomes:
        probability = outcome["probability"]
        next_state = outcome["next_state"]
        reward = outcome["reward"]
        done = outcome["done"]

        if done:
            future_value = 0.0
        else:
            future_value = values[next_state]

        transition_value = (
            reward
            + gamma * future_value
        )

        action_value += (
            probability
            * transition_value
        )

    return action_value

def perform_value_iteration_sweep(
    environment,
    values,
    gamma
):
    new_values = {}
    maximum_change = 0.0

    valid_states = environment.get_valid_states()

    for state in valid_states:
        if environment.is_terminal_state(state):
            new_state_value = 0.0

        else:
            available_actions = (
                environment.get_available_actions(state)
            )

            action_values = []

            for action in available_actions:
                action_value = calculate_action_value(
                    environment=environment,
                    state=state,
                    action=action,
                    values=values,
                    gamma=gamma
                )

                action_values.append(action_value)

            new_state_value = max(action_values)

        new_values[state] = new_state_value

        state_change = abs(
            new_state_value - values[state]
        )

        if state_change > maximum_change:
            maximum_change = state_change

    return new_values, maximum_change

def extract_greedy_policy(
    environment,
    values,
    gamma
):
    policy = {}

    valid_states = environment.get_valid_states()

    for state in valid_states:
        if environment.is_terminal_state(state):
            policy[state] = None
            continue

        available_actions = (
            environment.get_available_actions(state)
        )

        best_action = None
        best_action_value = float("-inf")

        for action in available_actions:
            action_value = calculate_action_value(
                environment=environment,
                state=state,
                action=action,
                values=values,
                gamma=gamma
            )

            if action_value > best_action_value:
                best_action_value = action_value
                best_action = action

        policy[state] = best_action

    return policy

def run_value_iteration(
    environment,
    gamma,
    convergence_threshold,
    max_iterations
):
    if convergence_threshold <= 0:
        raise ValueError(
            "Convergence threshold must be positive."
        )

    if max_iterations <= 0:
        raise ValueError(
            "Maximum iterations must be positive."
        )

    values = initialize_values(environment)

    delta_history = []
    converged = False

    start_time = perf_counter()

    for iteration_number in range(
        1,
        max_iterations + 1
    ):
        values, maximum_change = (
            perform_value_iteration_sweep(
                environment=environment,
                values=values,
                gamma=gamma
            )
        )

        delta_history.append(maximum_change)

        if maximum_change < convergence_threshold:
            converged = True
            break

    policy = extract_greedy_policy(
        environment=environment,
        values=values,
        gamma=gamma
    )
    execution_time = perf_counter() - start_time

    result = {
        "values": values,
        "policy": policy,
        "iterations": iteration_number,
        "delta_history": delta_history,
        "converged": converged,
        "execution_time": execution_time,
        "gamma": gamma,
        "convergence_threshold": convergence_threshold,
        "max_iterations": max_iterations,
        "reward_mode": environment.reward_mode
    }

    return result