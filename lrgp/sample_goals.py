import os
import numpy as np

from gym_simple_minigrid.minigrid import SimpleMiniGridEnv
from typing import Callable, Tuple
import torch

from .high import HighPolicy
from .low import LowPolicy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Sample_goal:
    def __init__(self, env: SimpleMiniGridEnv):
        self.env = env
        self.low = LowPolicy(env)
        self.high = HighPolicy(env)
        self.logs = list()

        self.possible_path = []
        self.q_possible_path = []
        self.idx_possible_path = 0
        self.success_flag = False
        self.scan_in_level = []


    def train(self, n_sample_low: int, n_episodes: int, low_h: int, high_h: int, test_each: int, n_episodes_test: int,
              update_each: int, n_updates: int, batch_size: int, epsilon_f: Callable, **kwargs):

        self.low_policy_learning(n_sample_low, low_h, update_each, n_updates, batch_size, epsilon_f)
        for episode in range(n_episodes):
            # Noise and epsilon for this episode
            epsilon = epsilon_f(episode)

            # Init episode variables
            subgoals_proposed = 0
            max_env_steps = False

            # Generate env initialization
            state, ep_goal = self.env.reset()
            goal_stack = [ep_goal]

            # Start LRGP
            while True:
                goal = goal_stack[-1]

                # Check if reachable
                reachable = self.low.is_reachable(state, goal, epsilon)

                if not reachable:
                    # Check if more proposals available
                    subgoals_proposed += 1
                    if subgoals_proposed > high_h:
                        break  # Too many proposals. Break and move to another episode

                    # Ask for a new subgoal
                    new_goal = self.high.select_action(state, goal)

                    # Bad proposals --> Same state, same goal or forbidden goal
                    # Penalize this proposal and avoid adding it to stack
                    if not self.low.is_allowed(new_goal, epsilon) or \
                            np.array_equal(new_goal, goal) or \
                            np.array_equal(new_goal, self.env.state_goal_mapper(state)):
                        self.high.add_penalization((state, new_goal, -high_h, state, goal, True))  # ns not used
                    else:
                        goal_stack.append(new_goal)

                else:
                    # Reachable. Apply a run of max low_h low actions
                    # Store run's initial state
                    state_high = state

                    # Init run variables
                    achieved = self._goal_achived(state, goal)
                    low_fwd = 0
                    low_steps = 0

                    # Add state to compute reachable pairs
                    self.low.add_run_step(state)
                    # Add current position as allowed goal to overcome the incomplete goal space problem
                    self.low.add_allowed_goal(self.env.state_goal_mapper(state))

                    # Apply steps
                    while low_fwd < low_h and low_steps < 2 * low_h and not achieved:
                        action = self.low.select_action(state, goal, epsilon)
                        next_state, reward, done, info = self.env.step(action)
                        # Check if last subgoal is achieved (not episode's goal)
                        achieved = self._goal_achived(next_state, goal)
                        self.low.add_transition((state, action, int(achieved) - 1, next_state, goal, achieved))

                        state = next_state

                        # Add info to reachable and allowed buffers
                        self.low.add_run_step(state)
                        self.low.add_allowed_goal(self.env.state_goal_mapper(state))

                        # Don't count turns
                        if action == SimpleMiniGridEnv.Actions.forward:
                            low_fwd += 1
                        # Max steps to avoid getting stuck
                        low_steps += 1

                        # Max env steps
                        if done and len(info) > 0:
                            max_env_steps = True
                            break

                    # Run's final state
                    next_state_high = state

                    # Create reachable transitions from run info
                    self.low.create_reachable_transitions(goal, achieved)

                    # We enforce a goal to be different from current state or previous goal, the agent MUST have moved
                    assert low_steps != 0

                    # Add run info for high agent to create transitions
                    if not np.array_equal(state_high, next_state_high):
                        self.high.add_run_info((state_high, goal, next_state_high))

                    # Update goal stack
                    while len(goal_stack) > 0 and self._goal_achived(next_state_high, goal_stack[-1]):
                        goal_stack.pop()

                    # Check episode completed successfully
                    if len(goal_stack) == 0:
                        break

                    # Check episode completed due to Max Env Steps
                    elif max_env_steps:
                        break

            # Perform end-of-episode actions (Compute transitions for high level and HER for low one)
            self.high.on_episode_end()
            self.low.on_episode_end()

            # Update networks / policies
            if (episode + 1) % update_each == 0:
                self.high.update(n_updates, batch_size)
                self.low.update(n_updates, batch_size)

            # Test to validate training
            if (episode + 1) % test_each == 0:
                subg, subg_a, steps, steps_a, max_subg, sr, low_sr = self._test(n_episodes_test, low_h, high_h)
                print(f"Episode {episode + 1:5d}: {100 * sr:5.1f}% Achieved")
                self.logs.append([episode, subg, subg_a, steps, steps_a, max_subg, sr, low_sr,
                                  len(self.high.replay_buffer), len(self.low.replay_buffer),
                                  len(self.low.reachable_buffer), len(self.low.allowed_buffer)])

    def _test(self, n_episodes: int, low_h: int, high_h: int) -> Tuple[np.ndarray, ...]:

        # Log metrics
        log_proposals = list()
        log_proposals_a = list()
        log_steps = list()
        log_steps_a = list()
        log_success = list()
        log_low_success = list()
        log_max_proposals = list()

        for episode in range(n_episodes):
            # Init episode variables
            subgoals_proposed = 0
            low_steps_ep = 0
            max_env_steps = max_subgoals_proposed = low_stuck = add_noise = False

            # Generate env initialization
            state, ep_goal = self.env.reset()
            goal_stack = [ep_goal]

            # Start LRGP
            while True:
                goal = goal_stack[-1]

                # Check if reachable
                reachable = self.low.is_reachable(state, goal, 0)

                if not reachable:
                    # Check if more proposals available
                    subgoals_proposed += 1
                    if subgoals_proposed > high_h:
                        max_subgoals_proposed = True
                        break  # Too many proposals. Break and move to another episode

                    # Ask for a new subgoal
                    new_goal = self.high.select_action_test(state, goal, add_noise)

                    # If not allowed, add noise to generate an adjacent goal
                    if not self.low.is_allowed(new_goal, 0):
                        add_noise = True
                    else:
                        goal_stack.append(new_goal)
                        add_noise = False

                else:
                    # Reachable. Apply a run of max low_h low actions
                    # Store run's initial state
                    state_high = state

                    # Init run variables
                    achieved = self._goal_achived(state, goal)
                    low_fwd = 0
                    low_steps = 0

                    # Apply steps
                    while low_fwd < low_h and low_steps < 2 * low_h and not achieved:
                        action = self.low.select_action(state, goal, 0)
                        next_state, reward, done, info = self.env.step(action)
                        achieved = self._goal_achived(next_state, goal)

                        state = next_state

                        # Don't count turns
                        if action == SimpleMiniGridEnv.Actions.forward:
                            low_fwd += 1
                        # Max steps to avoid getting stuck
                        low_steps += 1
                        low_steps_ep += 1  # To log performance

                        # Max env steps
                        if done and len(info) > 0:
                            max_env_steps = True
                            break

                    # Run's final state
                    next_state_high = state

                    log_low_success.append(achieved)

                    # Update goal stack
                    while len(goal_stack) > 0 and self._goal_achived(next_state_high, goal_stack[-1]):
                        goal_stack.pop()

                    # Check episode completed successfully
                    if len(goal_stack) == 0:
                        break

                    # Check episode completed due to bad low policy
                    elif np.array_equal(state_high, next_state_high):
                        low_stuck = True
                        break

                    # Check episode completed due to Max Env Steps
                    elif max_env_steps:
                        break

            # Log metrics
            episode_achieved = not max_subgoals_proposed and not max_env_steps and not low_stuck
            log_success.append(episode_achieved)
            log_max_proposals.append(max_subgoals_proposed)
            log_proposals.append(min(subgoals_proposed, high_h))
            log_steps.append(low_steps_ep)
            if episode_achieved:
                log_proposals_a.append(min(subgoals_proposed, high_h))
                log_steps_a.append(low_steps_ep)

        # Avoid taking the mean of an empty array
        if len(log_proposals_a) == 0:
            log_proposals_a = [0]
            log_steps_a = [0]

        return np.array(log_proposals).mean(), np.array(log_proposals_a).mean(), np.array(log_steps).mean(), \
               np.array(log_steps_a).mean(), np.array(log_max_proposals).mean(), np.array(log_success).mean(), \
               np.array(log_low_success).mean()

    def low_policy_learning(self, n_samples: int, low_h: int, update_each: int, n_updates: int, batch_size: int,
                            epsilon_f: Callable):
        for sample in range(n_samples):
            epsilon = epsilon_f(sample)
            state, ep_goal = self.env.reset()
            goal = ep_goal
            solution = [tuple(state)]
            achieved = self._goal_achived(state, goal)
            if not achieved:
                for run_iter in range(5):
                    last_state, max_env_steps = self.run_setps(state, goal, low_h, epsilon)
                    self.low.create_reachable_transitions(goal, achieved)
                    goal = self.env.state_goal_mapper(state)
                    state = last_state
                    solution.append(tuple(last_state))

            self.low.on_episode_end()
            n = len(solution)
            solution.reverse()
            for i, element in enumerate(solution):
                goal_1dim = self.env.location_to_number(self.env.state_goal_mapper(element))
                for j in range(1, len(solution)-i):
                    if self.env.state_goal_mapper(element) != self.env.state_goal_mapper(solution[i+j]):
                        self.high.goal_list[goal_1dim].add(self.env.state_goal_mapper(solution[i+j]))
                        curr_state_1dim = self.env.location_to_number(solution[i+j])
                        self.high.goal_list[curr_state_1dim].add(self.env.state_goal_mapper(element))
                    if j >= low_h:
                        break

            # Update networks / policies
            if (sample + 1) % update_each == 0:
                # self.high.update(n_updates, batch_size)
                self.low.update(n_updates, batch_size)

            if (sample + 1) % 50 == 0:
                print("low sampling target " + str(sample + 1))

    def store_path(self, idx_best):
        path = self.possible_path[idx_best].copy()
        path.reverse()
        for i in range(len(path) - 2):
            q_val = self.calc_q_vals((*path[i], 0), path[i + 1], path[i + 2])
            self.high.replay_buffer.add((*path[i], 0),  # state
                                        path[i + 1],  # action <-> proposed goal
                                        q_val,  # reward <-> - N runs
                                        (*path[i + 1], 0),  # (NOT USED) next_state
                                        path[i + 2],  # goal
                                        True)
        self.possible_path = list()
        self.q_possible_path = list()
        self.idx_possible_path = 0

    def recursive_intersect(self, state, goal, path, limit):
        if limit == 0:
            # path.pop()
            self.scan_in_level[limit].add(goal)
            return False, path
        state_1dim = self.env.location_to_number(state)
        goal_1dim = self.env.location_to_number(goal)
        if len(self.high.goal_list[goal_1dim]) == 0:
            # path.pop()
            self.scan_in_level[limit].add(goal)
            return False, path
        intersect = self.high.goal_list[state_1dim].intersection(self.high.goal_list[goal_1dim])
        if tuple(goal) in self.scan_in_level[limit]:
            return False, path
        path.append(tuple(goal))
        if len(intersect) == 0:  # keep on searching for state closer to current state
            res = False
            if limit == 1:
                self.scan_in_level[limit].add(goal)
                path.pop()
                return False, path
            set_to_check = self.high.goal_list[goal_1dim] - self.scan_in_level[limit]
            if len(set_to_check) == 0:
                self.scan_in_level[limit].add(goal)
                return False, path
            for subgoal in set_to_check:
                if tuple(subgoal) not in path: # and tuple(subgoal) not in self.scan_in_level[limit+1]:  #add flag at least one true return true
                    res, path = self.recursive_intersect(state, subgoal, path, limit - 1)
                    if res:
                        self.success_flag = True # At least one path was found
            self.scan_in_level[limit].add(tuple(goal))
            path.pop()
            return res, path
        else:  # found intersection with current state vicinity
            # store path and q-val of path
            v_val_tot = 0
            for i in range(len(path)-1):
                v_val = self.calc_v_vals((*path[i+1], 0), path[i])
                v_val_tot += v_val
            for i in range(len(intersect)):
                exp = list(intersect)[i]
                self.possible_path.append(None)
                self.q_possible_path.append(None)
                self.possible_path[self.idx_possible_path] = path.copy()
                self.q_possible_path[self.idx_possible_path] = v_val_tot
                self.possible_path[self.idx_possible_path].append(exp)
                q_val = self.calc_v_vals(state, exp)
                self.q_possible_path[self.idx_possible_path] += q_val
                self.idx_possible_path += 1
            # self.idx_possible_path += i
            path.pop()
            return True, path

    def _goal_achived(self, state: np.ndarray, goal: np.ndarray) -> bool:
        return np.array_equal(self.env.state_goal_mapper(state), goal)

    def calc_q_vals(self, state, action, goal):
        state_tensor = torch.FloatTensor(state).to(device)
        action_tensor_2dim = torch.FloatTensor(action).to(device)
        action_tensor_3dim = torch.FloatTensor((*action, 0)).to(device)
        goal_tensor = torch.FloatTensor(goal).to(device)
        state_action = torch.cat([state_tensor, action_tensor_2dim], dim=-1)
        action_goal = torch.cat([action_tensor_3dim, goal_tensor], dim=-1)
        with torch.no_grad():
            q_val = self.low.alg.value_network(state_action).numpy()[0] + \
                    self.low.alg.value_network(action_goal).numpy()[0]
        return q_val

    def calc_v_vals(self, state, goal):
        state_tensor = torch.FloatTensor(state).to(device)
        goal_tensor = torch.FloatTensor(goal).to(device)
        state_goal = torch.cat([state_tensor, goal_tensor], dim=-1)
        with torch.no_grad():
            v_val = self.low.alg.value_network(state_goal).numpy()[0]
        return v_val

    def run_setps(self, state: np.ndarray, goal: np.ndarray, low_h: int, epsilon: float):
        low_steps = low_fwd = 0
        max_env_steps = False
        achieved = self._goal_achived(state, goal)
        while low_fwd < low_h and low_steps < 2 * low_h and not achieved:
            action = self.low.select_action(state, self.env.state_goal_mapper(goal), epsilon)
            next_state, reward, done, info = self.env.step(action)
            # Check if last subgoal is achieved (not episode's goal)
            achieved = self._goal_achived(next_state, goal)
            self.low.add_transition(
                (state, action, int(achieved) - 1, next_state, self.env.state_goal_mapper(goal), achieved))

            state = next_state

            # Add info to reachable and allowed buffers
            self.low.add_run_step(state)
            self.low.add_allowed_goal(self.env.state_goal_mapper(state))

            # Don't count turns
            if action == SimpleMiniGridEnv.Actions.forward:
                low_fwd += 1
            # Max steps to avoid getting stuck
            low_steps += 1

            # Max env steps
            if done and len(info) > 0:
                max_env_steps = True
                return state, max_env_steps

        return state, max_env_steps

    def save(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path)
        self.high.save(path)
        self.low.save(path)
        with open(os.path.join(path, f"logs.npy"), 'wb') as f:
            np.save(f, np.array(self.logs))

    def load(self, path: str):
        self.high.load(path)
        self.low.load(path)
