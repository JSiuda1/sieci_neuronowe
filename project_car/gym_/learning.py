from racing_car_env import RacingCarEnv
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers.legacy import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy  # policy based rf
from rl.memory import SequentialMemory

env = RacingCarEnv()

# print(env.action_space.sample())

# print(env.observation_space.sample())

# episodes = 2
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

# print("test")
# log_path = os.path.join('Training', 'Logs')
# model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

# model.learn(total_timesteps=5000)

# shower_path = os.path.join('Training', 'Saved_models', 'Shower_model_ppo')
# model.save(shower_path)
# model = PPO.load(shower_path)

# evaluate_policy(model, env, n_eval_episodes=2, render=True)

def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1, states)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model
states = env.observation_space.shape[0]
actions = env.action_space.n
model = build_model(states, actions)
model.summary()

def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=5000, window_length = 1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)

    return dqn

dqn = build_agent(model, actions)
dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=30000, visualize=False, verbose=1)

scores = dqn.test(env, nb_episodes=2, visualize=True)
print(np.mean(scores.history['episode_reward']))
