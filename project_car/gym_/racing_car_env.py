import numpy as np
from gym import spaces, Env
from racing_car_game import CarGame2D


class RacingCarEnv(Env):
    def __init__(self) -> None:
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(np.array([0, 0, 0, 0, 0]), np.array([30, 30, 30, 30, 30]), dtype=np.float32)
        self.car_game = CarGame2D()

    def reset(self):
        self.car_game.reset()
        obs = self.car_game.observe()
        return obs

    def step(self, action):
        self.car_game.action(action)
        obs = self.car_game.observe()
        reward = self.car_game.evaluate()
        done = self.car_game.is_done()

        return obs, reward, done, {}

    def render(self, mode="human", close = False):
        self.car_game.view()

    # def close(self) -> None:
    #     self.car_game.close()