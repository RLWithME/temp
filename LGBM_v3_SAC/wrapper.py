from collections import deque
import gym


class DelayedEnv(gym.Wrapper):
    def __init__(self, env, seed, delay_step):
        super(DelayedEnv, self).__init__(env)
        assert delay_step > 0
        self.env.seed(seed)
        self.env.action_space.seed(seed)

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.delay_step = delay_step
        self._max_episode_steps = self.env._max_episode_steps

        self.obs_buffer = deque(maxlen=delay_step)
        self.reward_buffer = deque(maxlen=delay_step)
        self.done_buffer = deque(maxlen=delay_step)

    def reset(self):
        init_state = self.env.reset()
        for _ in range(self.delay_step):
            self.obs_buffer.append(init_state)
            self.reward_buffer.append(0)
            self.done_buffer.append(False)
        return init_state

    def step(self, action):
        current_obs, current_reward, current_done, _ = self.env.step(action)

        delayed_obs = self.obs_buffer.popleft()
        delayed_reward = self.reward_buffer.popleft()
        delayed_done = self.done_buffer.popleft()

        self.obs_buffer.append(current_obs)
        self.reward_buffer.append(current_reward)
        self.done_buffer.append(current_done)

        return delayed_obs, delayed_reward, delayed_done, None




