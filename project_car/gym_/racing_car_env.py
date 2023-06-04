import numpy as np
from gym import spaces, Env
from racing_car_game import CarGame2D
import numpy as np


class RacingCarEnv(Env):
    def __init__(self) -> None:
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(np.array([0, 0, 0, 0, 0]), np.array([50, 50, 50, 50, 50]), dtype=np.float32)
        self.car_game : CarGame2D = None

    def reset(self):
        if self.car_game is None:
            self.car_game = CarGame2D()

        self.car_game.reset()
        obs = self.car_game.observe()
        return np.array(obs)

    def step(self, action):
        self.car_game.action(action)
        obs = self.car_game.observe()
        reward = self.car_game.evaluate()
        done = self.car_game.is_done()

        return np.array(obs), reward, done, {}

    def render(self, mode="human", close = False):
        self.car_game.start_visualization()
        self.car_game.view()

    def close(self) -> None:
        self.car_game.close()
        del self.car_game
        self.car_game = None