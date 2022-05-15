import numpy as np
from gym_simple_minigrid.minigrid import SimpleMiniGridEnv
import torch
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
        self.goal_list = [set() for _ in range(self.env.height*self.env.width)]
        self.low_h = 0
        self.episode_runs = list()

    def select_action(self, state: np.ndarray, goal: np.ndarray, epsilon=0) -> np.ndarray:
        if self.replay_buffer.__len__() == 0 or np.random.random() < epsilon:  # for the first steps, high buffer still empty
            # action = np.random.uniform(-1, 1, size=(1, 3))
            # action = np.multiply(action, self.action_bound) + self.action_offset
            # SAC action is continuous [low, high + 1]
            action = self.alg.select_action(state, goal, False)
            # Discretize using floor --> discrete [low, high + 1]
            action = np.floor(action)
            # In case action was exactly high + 1, it is out of bounds. Clip
            action = np.clip(action, self.clip_low, self.clip_high)
            return action.astype(np.int)  # [0]
        # if np.random.random() < epsilon:
        #     action = self.env.observation_space.sample()
        #     return action
        possible_suggestions = []
        q_vals = []
        current_1d_goal = self.env.location_to_number(goal)
        if bool(self.goal_list[current_1d_goal]):
            for exp in self.goal_list[current_1d_goal]:
                if type(exp) == tuple:
                    action = exp
                else:
                    action = exp[0]
                possible_suggestions.append(action)
                state_tensor = torch.FloatTensor(state).to(device)
                goal_tensor = torch.FloatTensor(goal).to(device)
                action_tensor = torch.FloatTensor(action).to(device)
                action2d_tensor = torch.FloatTensor(self.env.state_goal_mapper(action)).to(device)
                state_action = torch.cat([state_tensor, action2d_tensor], dim=-1)
                action_goal = torch.cat([action_tensor, goal_tensor], dim=-1)
                state_goal = torch.cat([state_tensor, goal_tensor], dim=-1)
                with torch.no_grad():
                    q_value = self.alg.value(state_action).numpy()[0] + self.alg.value(action_goal).numpy()[0]
                    # q_value = self.alg.value2(state_goal, action_tensor).numpy()[0] #  + self.alg.value(action_goal).numpy()[0]  # direct estimation of the Q value.
                    q_vals.append(q_value)
        else:
            # SAC action is continuous [low, high + 1]
            action = self.alg.select_action(state, goal, False)
            # Discretize using floor --> discrete [low, high + 1]
            action = np.floor(action)
            # In case action was exactly high + 1, it is out of bounds. Clip
            action = np.clip(action, self.clip_low, self.clip_high)
            return action.astype(np.int)  # [0]
            # idx = np.random.randint(0, len(self.replay_buffer.buffer))
            # return list(self.replay_buffer.buffer)[idx][1]
        max_idx = np.argmax(np.array(q_vals))
        return possible_suggestions[max_idx]


    def select_action_test(self, state: np.ndarray, goal: np.ndarray, add_noise: bool = False) -> np.ndarray:
        # action = self.alg.select_action(state, goal, True)
        action = self.select_action(state, goal)
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
        # return action.astype(np.int)

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
                    self.replay_buffer.add(state_1,             # state
                                           next_state_2,    # action <-> proposed goal
                                           -(j - i + 1),        # reward <-> - N runs
                                           next_state_1,        # (NOT USED) next_state
                                           hindsight_goal_3,    # goal
                                           True)                # done --> Q-value = Reward (no bootstrap / Bellman eq)

                    if tuple(state_1) != tuple(next_state_1):
                        goal_1d = self.env.location_to_number(self.env.state_goal_mapper(next_state_1))
                        self.goal_list[goal_1d].add(tuple(state_1))
                    if (j - i + 1) <= 6:  # TODO: make this adjustable
                        if tuple(state_1) != tuple(next_state_3):
                            goal_1d = self.env.location_to_number(hindsight_goal_3)
                            self.goal_list[goal_1d].add(tuple(state_1))
        self.episode_runs = list()

    def update(self, n_updates: int, batch_size: int):
        if len(self.replay_buffer) > 0:
            for _ in range(n_updates):
                self.alg.update(self.replay_buffer, batch_size)

    def save(self, path: str):
        self.alg.save(path, "high")

    def load(self, path: str):
        self.alg.load(path, "high")
