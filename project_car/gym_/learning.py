from racing_car_env import RacingCarEnv
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

env = RacingCarEnv()

# print(env.action_space.sample())

# print(env.observation_space.sample())

# episodes = 5
# for episode in range(1, episodes + 1):
#     observation = env.reset()
#     done = False
#     score = 0

#     while not done:
#         env.render()
#         action = env.action_space.sample()     # Using model here
#         observation, reward, done, info = env.step(action)
#         score += reward
#     print("Episode {} Score {}".format(episode, score))
# env.close()


# log_path = os.path.join('Training', 'Logs')
# model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

# model.learn(total_timesteps=200000)

shower_path = os.path.join('Training', 'Saved_models', 'Shower_model_ppo')
# model.save(shower_path)
model = PPO.load(shower_path)

evaluate_policy(model, env, n_eval_episodes=10, render=True)