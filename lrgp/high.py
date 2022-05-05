import random
import torch

import numpy as np
from gym_simple_minigrid.minigrid import SimpleMiniGridEnv

from .rl_algs.sac import SACStateGoal
from .utils.utils import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HighPolicy:
    def __init__(self, env: SimpleMiniGridEnv, gamma: float = 1., tau: float = 0.005, br_size: int = 1e6):
        self.env = env

        state_shape = env.observation_space.shape[0]
        goal_shape = env.state_goal_mapper(env.observation_space.sample()).shape[0]
        # High agent proposes goals --> action space === goal space
        action_shape = state_shape

        # Compute action bounds to convert SAC's action in the correct range
        action_low = env.state_goal_mapper(env.observation_space.low)
        action_high = env.state_goal_mapper(env.observation_space.high)
        action_high_corrected = 1 + action_high  # To adapt discrete env into SAC (continuous actions)

        self.action_bound = (action_high_corrected - action_low) / 2
        self.action_offset = (action_high_corrected + action_low) / 2
        self.action_bound = np.concatenate((self.action_bound, np.array([2])))
        self.action_offset = np.concatenate((self.action_offset, np.array([2])))

        # Init SAC algorithm, base learner for high agent
        self.alg = SACStateGoal(state_shape, action_shape, goal_shape, self.action_bound, self.action_offset, gamma, tau)

        self.clip_low = np.concatenate((action_low, np.array([0])))
        self.clip_high = np.concatenate((action_high, np.array([3])))

        self.replay_buffer = ReplayBuffer(br_size)
        self.episode_runs = list()
        self.solution = list()

    def select_action(self, state: np.ndarray, goal: np.ndarray) -> np.ndarray:
        if self.replay_buffer.__len__() == 0:  # for the first steps, high buffer still empty
            action = np.random.uniform(-1, 1, size=(3, 1))
            # action = np.concatenate((action, np.array([np.random.randint(0, 3)])))
            action = action * self.action_bound + self.action_offset
            return action.astype(np.int)[0]
        possible_suggestions = []
        q_vals = []
        state = torch.FloatTensor(state).to(device)
        goal = torch.FloatTensor(goal).to(device)
        for exp in self.replay_buffer.buffer:
            if tuple(exp[4]) == tuple(goal):  #TODO: add limitation of horizon:
                action = exp[1]
                possible_suggestions.append(action)
                action = torch.FloatTensor(action).to(device)
                action_as_goal = torch.FloatTensor(action[:2]).to(device)
                state_action = torch.cat([state, action_as_goal], dim=-1)
                action_goal = torch.cat([action, goal], dim=-1)
                with torch.no_grad():
                    q_value = self.alg.value(state_action) + self.alg.value(action_goal)  # direct estimation of the Q value.
                    q_vals.append(q_value)
        if len(q_vals) == 0:
            idx = np.random.randint(0, len(self.replay_buffer.buffer))
            return self.replay_buffer.buffer[idx][1]
        max_idx = np.argmax(np.array(q_vals))
        return possible_suggestions[max_idx]

        # SAC action is continuous [low, high + 1]
        # action = self.alg.select_action(state, goal, False)
        # # Discretize using floor --> discrete [low, high + 1]
        # action = np.floor(action)
        # # In case action was exactly high + 1, it is out of bounds. Clip
        # action = np.clip(action, self.clip_low, self.clip_high)

    def select_action_test(self, state: np.ndarray, goal: np.ndarray, add_noise: bool = False) -> np.ndarray:
        action = self.select_action(state, goal)  # , True)

        # noise = 0
        # if add_noise:
        #     # Add small noise so we choose an adjacent goal position
        #     noise = np.random.randint(-1, 2, action.shape)
        # action = action + noise
        #
        # # Discretize and clip
        # action = np.floor(action)
        # action = np.clip(action, self.clip_low, self.clip_high)

        return action

    def add_run_info(self, info: tuple):
        self.episode_runs.append(info)

    def add_penalization(self, transition: tuple):
        self.replay_buffer.add(*transition)

    def on_episode_end(self):
        # Create MonteCarlo-based transitions from episode runs
        # Hindsight goals --> Next state as proposed goal (as if low level acts optimally)

        # We do not create transitions between states and goals that are reachable within one run.
        # We do not need to learn them because no subgoals are required

        for i, (state_1, _, next_state_1) in enumerate(self.episode_runs):
            for j, (_, _, next_state_3) in enumerate(self.episode_runs[i + 1:], i + 1):
                # Used as final goal
                hindsight_goal_3 = self.env.state_goal_mapper(next_state_3)
                for k, (_, _, next_state_2) in enumerate(self.episode_runs[i:j], i):
                    # Used as intermediate goal or proposed action
                    hindsight_goal_2 = self.env.state_goal_mapper(next_state_2)
                    state = torch.FloatTensor(state_1).to(device)
                    action_3dim = torch.FloatTensor(next_state_2).to(device)
                    action = torch.FloatTensor(hindsight_goal_2).to(device)
                    goal = torch.FloatTensor(hindsight_goal_3).to(device)
                    state_action = torch.cat([state, action], dim=-1)
                    action_goal = torch.cat([action_3dim, goal], dim=-1)
                    state_goal = torch.cat([state, goal], dim=-1)
                    with torch.no_grad():
                        q1 = self.alg.value(state_action)
                        q2 = self.alg.value(action_goal)
                        q3 = self.alg.value(state_goal)
                        if q3 < q1 + q2:
                            self.replay_buffer.add(tuple(state_1),  # state
                                                   next_state_2,  # action <-> proposed goal
                                                   # -(j - i + 1),  # reward <-> - N runs
                                                   (q1 + q2).numpy()[0],
                                                   next_state_1,  # (NOT USED) next_state
                                                   hindsight_goal_3,  # goal
                                                   True)  # done --> Q-value = Reward (no bootstrap / Bellman eq)
        self.episode_runs = list()

        # for ni in range(len(self.solution)):
        #     i = self.solution[ni]
        #     for nj in range(ni+1, len(self.solution)):
        #         j = self.solution[nj]
        #         for nk in range(nj+1, len(self.solution)):
        #             k = self.solution[nk]
        #             q1 = self.alg.value(i, j)
        #             q2 = self.alg.value(j, k)
        #             q3 = self.alg.value(i, k)
        #             if q3 > q1 + q2:
        #                 self.replay_buffer.add(i,j,q1+q2,k, True)  # (state, subgoal, cost, goal)

        # self.solution = list()

    def update(self, n_updates: int, batch_size: int):
        if len(self.replay_buffer) > 0:
            for _ in range(n_updates):
                self.alg.update(self.replay_buffer, batch_size)

    def save(self, path: str):
        self.alg.save(path, "high")

    def load(self, path: str):
        self.alg.load(path, "high")
