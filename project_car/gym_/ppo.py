# CODE BASE ON KERAS REINFORCEMENT LEARNING EXAMPLES PPO
# https://keras.io/examples/rl/ppo_cartpole/

import numpy as np
import scipy.signal
from gym import spaces, Env
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import racing_car_env


class Buffer:
    # Buffer for storing trajectories
    def __init__(self, observation_dimensions, size, gamma=0.99, lam=0.95):
        # Buffer initialization
        self.observation_buffer = np.zeros(
            (size, observation_dimensions), dtype=np.float32
        )
        self.action_buffer = np.zeros(size, dtype=np.int32)
        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.return_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.logprobability_buffer = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.pointer, self.trajectory_start_index = 0, 0

    def store(self, observation, action, reward, value, logprobability):
        # Append one step of agent-environment interaction
        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.logprobability_buffer[self.pointer] = logprobability
        self.pointer += 1

    def discounted_cumulative_sums(self, x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def finish_trajectory(self, last_value=0):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantage_buffer[path_slice] = self.discounted_cumulative_sums(
            deltas, self.gamma * self.lam
        )
        self.return_buffer[path_slice] = self.discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]

        self.trajectory_start_index = self.pointer

    def get(self):
        # Get all data of the buffer and normalize the advantages
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean, advantage_std = (
            np.mean(self.advantage_buffer),
            np.std(self.advantage_buffer),
        )
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_std
        return (
            self.observation_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_buffer,
            self.logprobability_buffer,
        )


class PPO():
    def __init__(self, enviroment: Env,
                 gamma = 0.99, lam = 0.97, clip_ratio = 0.2, policy_learning_rate = 3e-4,
                 value_learning_rate = 3e-4, train_policy_iterations = 80, train_value_iterations = 80,
                 target_kl = 0.01, hidden_sizes = (64, 64)) -> None:
        self._env = enviroment
        self._gamma = gamma
        self._lam = lam
        self._clip_ratio = clip_ratio
        self._policy_learning_rate = policy_learning_rate
        self._train_policy_iterations = train_policy_iterations
        self._train_value_iterations = train_value_iterations
        self._target_kl = target_kl

        self._actor, self._critic = self.create_models(hidden_sizes)
        self._policy_optymizer = keras.optimizers.Adam(learning_rate=policy_learning_rate)
        self._value_optimizer = keras.optimizers.Adam(learning_rate=value_learning_rate)

    def create_feedforward_nn(self, x, sizes, activation=tf.tanh, output_activation=None):
        for size in sizes[:-1]:
            x = layers.Dense(units=size, activation=activation)(x)
        return layers.Dense(units=sizes[-1], activation=output_activation)(x)

    def create_models(self, hidden_sizes) -> tuple[keras.Model]:
        obs_space = self._env.observation_space.shape[0]
        num_actions = self._env.action_space.n
        input = keras.Input(shape=(obs_space,),dtype=tf.float32)
        actor_out = self.create_feedforward_nn(input, list(hidden_sizes) + [num_actions], tf.tanh, None)
        actor = keras.Model(inputs = input, outputs = actor_out)

        critic_out = self.create_feedforward_nn(input, list(hidden_sizes) + [1], tf.tanh, None)
        critic = keras.Model(inputs = input, outputs = critic_out)

        return actor, critic

    def _sample_action(self, observation):
        logits = self._actor(observation) # get log output  observation
        action = tf.squeeze(tf.random.categorical(logits, 1), axis=1) # get sample action
        return logits, action

    def _logprobabilities(self, logits, a):
        # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
        logprobabilities_all = tf.nn.log_softmax(logits)
        logprobability = tf.reduce_sum(
            tf.one_hot(a, self._env.action_space.n) * logprobabilities_all, axis=1
        )
        return logprobability

    # Train policy, maximizing the PPO-Clip objective
    def _train_policy( self,
        observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
    ):
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            ratio = tf.exp(
                self._logprobabilities(self._actor(observation_buffer), action_buffer)
                - logprobability_buffer
            )
            min_advantage = tf.where(
                advantage_buffer > 0,
                (1 + self._clip_ratio) * advantage_buffer,
                (1 - self._clip_ratio) * advantage_buffer,
            )

            policy_loss = -tf.reduce_mean(
                tf.minimum(ratio * advantage_buffer, min_advantage)
            )
        policy_grads = tape.gradient(policy_loss, self._actor.trainable_variables)

        # apply gradients for policy optymizer
        self._policy_optymizer.apply_gradients(zip(policy_grads, self._actor.trainable_variables))

        kl = tf.reduce_mean(
            logprobability_buffer
            - self._logprobabilities(self._actor(observation_buffer), action_buffer)
        )
        kl = tf.reduce_sum(kl)

        return kl


    # Fit value function by regression on mean-square error
    def _train_value_function(self, observation_buffer, return_buffer):
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            value_loss = tf.reduce_mean((return_buffer - self._critic(observation_buffer)) ** 2)
        value_grads = tape.gradient(value_loss, self._critic.trainable_variables)

        #apply gradients for value oprimizer
        self._value_optimizer.apply_gradients(zip(value_grads, self._critic.trainable_variables))



    def learn(self, epochs = 10, steps_per_epoch = 4000, render = False):
        self._buffer = Buffer(self._env.observation_space.shape[0], steps_per_epoch)
        # initalize observation episodes and length
        observation, episode_return, episode_length = self._env.reset(), 0, 0

        for epoch in range(epochs):
            sum_ret = 0
            sum_length = 0
            num_episodes = 0

            # Collect set of data
            for t in range(steps_per_epoch):
                if render:
                    self._env.render()

                # make step
                observation = observation.reshape(1, -1)
                logits, action = self._sample_action(observation)
                observation_new, reward, done, _ = self._env.step(action[0].numpy())
                episode_return += reward
                episode_length += 1

                # get value and log-probablility of the action
                value_crit = self._critic(observation)
                logprob_crit = self._logprobabilities(logits, action)

                # store data
                self._buffer.store(observation, action, reward, value_crit, logprob_crit)

                # udpate observation
                observation = observation_new

                terminal = done
                if terminal or (t == steps_per_epoch - 1):
                    last_value = 0 if done else self._critic(observation.reshape(1, -1))
                    self._buffer.finish_trajectory(last_value)
                    sum_ret += episode_return
                    sum_length += episode_length
                    num_episodes += 1
                    observation, episode_return, episode_length = self._env.reset(), 0, 0

            (
            observation_buffer,
            action_buffer,
            advantage_buffer,
            return_buffer,
            logprobability_buffer,
            ) = self._buffer.get()

            # Update the policy and implement early stopping using KL divergence
            for _ in range(self._train_policy_iterations):
                kl = self._train_policy(
                    observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
                )

                # early stopping
                if kl > 1.5 * self._target_kl:
                    break

            # Update the value function
            for _ in range(self._train_value_iterations):
                self._train_value_function(observation_buffer, return_buffer)

            print(
                f" Epoch: {epoch}. Mean Return: {sum_ret / num_episodes}. Mean Length: {sum_length / num_episodes}"
            )

    def save_weights(self, actor_path = "ppo_training/ppo_actor", critic_path = "ppo_training/ppo_critic"):
        self._actor.save_weights(actor_path)
        self._critic.save_weights(critic_path)

    def load_weights(self, actor_path = "ppo_training/ppo_actor", critic_path = "ppo_training/ppo_critic"):
        self._actor.load_weights(actor_path)
        self._critic.load_weights(critic_path)

    def evaluate(self, n_eval_epoch = 10, render = False):

        for i in range(n_eval_epoch):
            observation = self._env.reset()
            done = False
            episode_return = 0

            while not done:
                observation = observation.reshape(1, -1)
                logits, action = self._sample_action(observation)
                observation_new, reward, done, _ = self._env.step(action[0].numpy())
                episode_return += reward
                observation = observation_new

                if render:
                    self._env.render()

            print(f"Epoch {i}, reward {episode_return}")


if __name__ == "__main__":
    gym = racing_car_env.RacingCarEnv()
    test = PPO(gym)
    test.learn(epochs=20)
    test.save_weights()
    test.evaluate()