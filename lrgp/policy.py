import os
import numpy as np

from gym_simple_minigrid.minigrid import SimpleMiniGridEnv
from typing import Callable, Tuple
from .baselines import DDQN_agent, DDQN_HER
from .high import HighPolicy
from .low import LowPolicy
import time


class Policy:
    def __init__(self, env: SimpleMiniGridEnv):
        self.env = env
        self.policy = DDQN_agent
        self.logs = list()

    def _goal_achived(self, state: np.ndarray, goal: np.ndarray) -> bool:
        return np.array_equal(state, goal)

    def _test(self, n_episodes: int, low_h: int, **kwargs) -> Tuple[np.ndarray, ...]:

        # Log metrics
        log_steps = list()
        log_steps_a = list()
        log_low_success = list()

        for episode in range(n_episodes):
            # Init episode variables
            low_steps_ep = 0
            max_env_steps = low_stuck = add_noise = False

            # Generate env initialization
            state, ep_goal = self.env.reset()
            goal = np.concatenate((ep_goal, np.random.randint(0, 3, 1)))

            achieved = self._goal_achived(state, goal)
            low_fwd = 0
            low_steps = 0

            # Apply steps
            while not achieved and low_steps_ep < low_h:
                action = self.policy.select_action(state, goal, 0)
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

            log_low_success.append(achieved)

            # Check episode completed due to Max Env Steps

            # Log metrics
            log_steps.append(low_steps_ep)

        return np.array(log_steps).mean(), np.array(log_steps_a).mean(), \
               np.array(log_low_success).mean()

    def save(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path)
        self.policy.save(path)
        with open(os.path.join(path, f"logs.npy"), 'wb') as f:
            np.save(f, np.array(self.logs))

    def load(self, path: str):
        self.policy.load(path)


class Policy_ddqn(Policy):
    def __init__(self, env: SimpleMiniGridEnv):
        super().__init__(env)
        self.policy = DDQN_agent(env)
        self.env = env
        self.logs = list()

    def train(self, n_episodes: int, low_h: int, test_each: int, n_episodes_test: int,
              update_each: int, n_updates: int, batch_size: int, epsilon_f: Callable, **kwargs):
        start_time = time.time()
        for episode in range(n_episodes):
            # Noise and epsilon for this episode
            epsilon = epsilon_f(episode)
            # Generate env initialization
            state, ep_goal = self.env.reset()
            goal = np.concatenate((ep_goal, np.random.randint(0, 3, 1)))
            achieved = self._goal_achived(state, goal)
            low_steps = 0
            while not achieved and low_steps < low_h:
                # Init run variables
                low_fwd = 0
                low_steps = 0

                # Add state to compute reachable pairs
                self.policy.add_run_step(state)
                action = self.policy.select_action(state, goal, epsilon)
                next_state, reward, done, info = self.env.step(action)

                # Check if last subgoal is achieved (not episode's goal)
                achieved = self._goal_achived(next_state, goal)
                self.policy.add_transition((state, action, int(achieved) - 1, next_state, goal, achieved))
                state = next_state

                # Add info to reachable and allowed buffers
                self.policy.add_run_step(state)
                # Don't count turns
                if action == SimpleMiniGridEnv.Actions.forward:
                    low_fwd += 1
                # Max steps to avoid getting stuck
                low_steps += 1

                # Max env steps
                if done and len(info) > 0:
                    max_env_steps = True
                    break

            # Create reachable transitions from run info
            self.policy.empty_run_steps(goal, achieved)

            # We enforce a goal to be different from current state or previous goal, the agent MUST have moved
            assert low_steps != 0

            # Perform end-of-episode actions (Compute transitions for high level and HER for low one)
            # self.policy.on_episode_end()

            # Update networks / policies
            if (episode + 1) % update_each == 0:
                self.policy.update(n_updates, batch_size)

            # Test to validate training
            if (episode + 1) % test_each == 0:
                steps, steps_a, low_sr = self._test(n_episodes_test, low_h)
                curr_time = (time.time() - start_time) / 60
                print(f"Episode {episode + 1:5d}: {100 * low_sr:5.1f}% Achieved")
                self.logs.append([episode, steps, steps_a, low_sr,
                                  len(self.policy.replay_buffer), curr_time])
                self.save(os.path.join('logs', kwargs['job_name']))

class Policy_ddqn_her(Policy):
    def __init__(self, env: SimpleMiniGridEnv):
        super().__init__(env)
        self.policy = DDQN_HER(env)
        self.env = env
        self.logs = list()

    def train(self, n_episodes: int, low_h: int, test_each: int, n_episodes_test: int,
              update_each: int, n_updates: int, batch_size: int, epsilon_f: Callable, **kwargs):
        start_time = time.time()
        for episode in range(n_episodes):
            # Noise and epsilon for this episode
            epsilon = epsilon_f(episode)
            # Generate env initialization
            state, ep_goal = self.env.reset()
            goal = np.concatenate((ep_goal, np.random.randint(0, 3, 1)))
            achieved = self._goal_achived(state, goal)
            low_steps = 0
            while not achieved and low_steps < low_h:
                # Init run variables
                low_fwd = 0
                low_steps = 0

                # Add state to compute reachable pairs
                self.policy.add_run_step(state)
                action = self.policy.select_action(state, goal, epsilon)
                next_state, reward, done, info = self.env.step(action)

                # Check if last subgoal is achieved (not episode's goal)
                achieved = self._goal_achived(next_state, goal)
                self.policy.add_transition((state, action, int(achieved) - 1, next_state, goal, achieved))
                state = next_state

                # Add info to reachable and allowed buffers
                self.policy.add_run_step(state)

                # Don't count turns
                if action == SimpleMiniGridEnv.Actions.forward:
                    low_fwd += 1
                # Max steps to avoid getting stuck
                low_steps += 1

                # Max env steps
                if done and len(info) > 0:
                    max_env_steps = True
                    break

            # Create reachable transitions from run info
            self.policy.empty_run_steps(goal, achieved)

            # We enforce a goal to be different from current state or previous goal, the agent MUST have moved
            assert low_steps != 0

            # Perform end-of-episode actions (Compute transitions for high level and HER for low one)
            self.policy.on_episode_end()

            # Update networks / policies
            if (episode + 1) % update_each == 0:
                self.policy.update(n_updates, batch_size)

            # Test to validate training
            if (episode + 1) % test_each == 0:
                steps, steps_a, low_sr = self._test(n_episodes_test, low_h)
                curr_time = (time.time() - start_time) / 60
                print(f"Episode {episode + 1:5d}: {100 * low_sr:5.1f}% Achieved")
                self.logs.append([episode, steps, steps_a, low_sr,
                                  len(self.policy.replay_buffer), curr_time])
                self.save(os.path.join('logs', kwargs['job_name']))
