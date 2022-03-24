import os
import numpy as np

from gym_simple_minigrid.minigrid import SimpleMiniGridEnv
from typing import Callable, Tuple

from .high import HighPolicy
from .low import LowPolicy


class Hierarchy:
    def __init__(self, env: SimpleMiniGridEnv):
        self.env = env
        self.low = LowPolicy(env)
        self.high = HighPolicy(env)

        self.logs = list()

    def train(self, n_episodes: int, low_h: int, high_h: int, test_each: int, n_episodes_test: int,
              update_each: int, n_updates: int, batch_size: int, epsilon_f: Callable, **kwargs):

        for episode in range(n_episodes):
            # Noise and epsilon for this episode
            epsilon = epsilon_f(episode)

            # Init episode variables
            subgoals_proposed = 0
            max_env_steps = False

            # Generate env initialization
            s, g = self.env.reset()
            g = np.concatenate((g, np.array([4]))) #having orientation of 4 means it is the ultimate goal
            starting_state_list = [g, s]

            # Start LRGP
            while True:
                css = starting_state_list[1]  # current_starting_point
                cgs = starting_state_list[0]
                # Check if reachable
                reachable = self.low.is_reachable(css, cgs, epsilon)

                if not reachable:
                    # Check if more proposals available
                    subgoals_proposed += 1
                    if subgoals_proposed > high_h:
                        break  # Too many proposals. Break and move to another episode

                    # Ask for a new starting point
                    new_ss = self.high.select_action(css, cgs)
                    new_ss_loc = self.env.state_goal_mapper(new_ss)
                    # Bad proposals --> Same state, same goal or forbidden goal
                    # Penalize this proposal and avoid adding it to stack
                    if not self.low.is_allowed(new_ss_loc, epsilon) or \
                            np.array_equal(new_ss_loc, self.env.state_goal_mapper(cgs)) or \
                            np.array_equal(new_ss_loc, self.env.state_goal_mapper(css)):
                        self.high.add_penalization((css, new_ss, -high_h, css, cgs, True))  # ns not used
                    else:
                        # Adding the new goal in between cgs ans css
                        starting_state_list.insert(1, new_ss)

                else:
                    # Reachable. Apply a run of max low_h low actions
                    # Store run's initial state
                    state_high = css

                    # Init run variables
                    achieved = self._goal_achived(css, cgs)
                    low_fwd = 0
                    low_steps = 0

                    # Add state to compute reachable pairs
                    self.low.add_run_step(css)
                    # Add current position as allowed goal to overcome the incomplete goal space problem
                    self.low.add_allowed_goal(self.env.state_goal_mapper(css))

                    # Apply steps

                    while low_fwd < low_h and low_steps < 2 * low_h and not achieved:
                        action = self.low.select_action(css, cgs, epsilon)
                        next_state, reward, done, info = self.env.step(action, css)
                        # Check if last subgoal is achieved (not episode's goal)
                        achieved = self._goal_achived(next_state, cgs)
                        self.low.add_transition((css, action, int(achieved) - 1, next_state, cgs, achieved))

                        css = next_state

                        # Add info to reachable and allowed buffers
                        self.low.add_run_step(css)
                        self.low.add_allowed_goal(self.env.state_goal_mapper(css))

                        # Don't count turns
                        if action == SimpleMiniGridEnv.Actions.forward:
                            low_fwd += 1
                        # Max steps to avoid getting stuck
                        low_steps += 1

                        # Max env steps
                        if done and len(info) > 0:
                            max_env_steps = True
                            break

                        if np.array_equal(self.env.state_goal_mapper(css), self.env.state_goal_mapper(cgs)):
                            starting_state_list = starting_state_list[1:]
                            break

                    # Run's final state
                    next_state_high = css

                    # Create reachable transitions from run info
                    self.low.create_reachable_transitions(self.env.state_goal_mapper(cgs), achieved)

                    # We enforce a goal to be different from current state or previous goal, the agent MUST have moved
                    assert low_steps != 0

                    # Add run info for high agent to create transitions
                    if not np.array_equal(self.env.state_goal_mapper(state_high), self.env.state_goal_mapper(next_state_high)):
                        self.high.add_run_info((state_high, self.env.state_goal_mapper(cgs), next_state_high))

                    # Update goal stack
                    # while len(goal_stack) > 0 and self._goal_achived(next_state_high, goal_stack[-1]):
                    #     goal_stack.pop()

                    # Check episode completed successfully
                    if len(starting_state_list) == 1:
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

    def test(self, n_episodes: int, low_h: int, high_h: int, render: bool = False, **kwargs) -> Tuple[np.ndarray, ...]:
        if render:
            return self._test_render(n_episodes, low_h, high_h)
        else:
            return self._test(n_episodes, low_h, high_h)

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
            ep_goal = np.concatenate((ep_goal, np.array([4])))
            starting_state_list = [ep_goal, state]

            # Start LRGP
            while True:
                css = starting_state_list[1]  # current_starting_point
                cgs = starting_state_list[0]

                # Check if reachable
                reachable = self.low.is_reachable(css, cgs, 0)

                if not reachable:
                    # Check if more proposals available
                    subgoals_proposed += 1
                    if subgoals_proposed > high_h:
                        max_subgoals_proposed = True
                        break  # Too many proposals. Break and move to another episode

                    # Ask for a new subgoal
                    new_ss = self.high.select_action(css, cgs)
                    new_ss_loc = self.env.state_goal_mapper(new_ss)

                    # If not allowed, add noise to generate an adjacent goal
                    if not self.low.is_allowed(new_ss_loc, 0):
                        add_noise = True
                    else:
                        starting_state_list.insert(1, new_ss)
                        add_noise = False

                else:
                    # Reachable. Apply a run of max low_h low actions
                    # Store run's initial state
                    state_high = css

                    # Init run variables
                    achieved = self._goal_achived(css, cgs)
                    low_fwd = 0
                    low_steps = 0

                    # Apply steps
                    while low_fwd < low_h and low_steps < 2 * low_h and not achieved:
                        action = self.low.select_action(css, cgs, 0)
                        next_state, reward, done, info = self.env.step(action)
                        achieved = self._goal_achived(next_state, cgs)

                        css = next_state

                        # Don't count turns
                        if action == SimpleMiniGridEnv.Actions.forward:
                            low_fwd += 1
                        # Max steps to avoid getting stuck
                        low_steps += 1
                        low_steps_ep += 1  # To log performance

                        if achieved:
                            starting_state_list = starting_state_list[1:]
                            break

                        # Max env steps
                        if done and len(info) > 0:
                            max_env_steps = True
                            break

                    # Run's final state
                    next_state_high = css

                    log_low_success.append(achieved)

                    # Update goal stack
                    # while len(goal_stack) > 0 and self._goal_achived(next_state_high, goal_stack[-1]):
                    #     goal_stack.pop()

                    # Check episode completed successfully
                    if len(starting_state_list) == 1:
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

    def _test_render(self, n_episodes: int, low_h: int, high_h: int) -> Tuple[np.ndarray, ...]:

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
            self.env.render()
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
                        self.env.add_goal(new_goal)
                        self.env.render()
                        self.env.remove_goal()
                        self.env.render()
                    else:
                        goal_stack.append(new_goal)
                        self.env.add_goal(new_goal)
                        self.env.render()
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
                        self.env.render()
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

                    # Update stack
                    while len(goal_stack) > 0 and self._goal_achived(next_state_high, goal_stack[-1]):
                        goal_stack.pop()
                        self.env.remove_goal()
                    self.env.render()

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

    def _goal_achived(self, state: np.ndarray, goal: np.ndarray) -> bool:
        return np.array_equal(self.env.state_goal_mapper(state), self.env.state_goal_mapper(goal))

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
