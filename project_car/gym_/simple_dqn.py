# Simple dqn reinfocement lerning
# Made with https://www.youtube.com/watch?v=5fHngyN8Qhw

from keras.layers import Dense, Activation
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import numpy as np

class DqnBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, discrete = False) -> None:
        self._mem_size = max_size
        self._discrete = discrete
        self._state_memory = np.zeros((self._mem_size, input_shape))
        self._new_state_memory = np.zeros((self._mem_size, input_shape))
        dtype = np.int8 if self._discrete else np.float32
        self._action_memory = np.zeros((self._mem_size, n_actions), dtype=dtype)
        self._reward_memory = np.zeros(self._mem_size)
        self._terminal_memory = np.zeros(self._mem_size, dtype=np.float32)
        self.mem_counter = 0

    # store one episode
    def store(self, state, action, reward, new_state, done):
        index = self.mem_counter % self._mem_size
        self._state_memory[index] = state
        self._new_state_memory[index] = new_state
        self._reward_memory[index] = reward
        self._terminal_memory[index] = 1 - int(done)

        if self._discrete:
            actions = np.zeros(self._action_memory.shape[1])
            actions[action] = 1.0
            self._action_memory[index] = actions
        else:
            self._action_memory[index] = action

        self.mem_counter += 1

    # get one random episode
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self._mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self._state_memory[batch]
        new_states = self._new_state_memory[batch]
        rewards = self._reward_memory[batch]
        actions = self._action_memory[batch]
        terminal = self._terminal_memory[batch]

        return states, actions, rewards, new_states, terminal

def build_dqn(learning_rate, n_actions, input_dims, fc1_dims, fc2_dims):
    model = Sequential([
        Dense(fc1_dims, input_shape = (input_dims,)),
        Activation('relu'),
        Dense(fc2_dims),
        Activation("relu"),
        Dense(n_actions)
    ])

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    model.summary()
    return model


class Agent(object):
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size,
                 input_dims, epsilon_dec = 0.996, epsiolon_end = 0.01,
                 mem_size = 10_000_00, fname='dqn_model.h5') -> None:
        """_summary_

        Args:
            alpha (_type_): learining rate
            gamma (_type_): discount factor of reward
            n_actions (_type_): number of enviroment actions
            epsilon (_type_): random factor, if random number is less than epsilon we will take random action instead of nn action
            batch_size (_type_): batch size
            input_dims (_type_): input dims
            epsilon_dec (float, optional): . Defaults to 0.996.
            epsiolon_end (float, optional): minimum value for epsilon. Defaults to 0.01.
            mem_size (_type_, optional): size of memory. Defaults to 10_000_00.
            fname (str, optional): file to save model. Defaults to 'dqn_model.h5'.
        """
        self._action_space = [i for i in range(n_actions)]
        self._n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsiolon_end
        self.batch_size = batch_size
        self.model_file = fname

        self.memory = DqnBuffer(mem_size, input_dims, n_actions, discrete=True)
        self.q_eval = build_dqn(alpha, n_actions, input_dims, 256, 256)

    def rememeber(self, state, action, reward, new_state, done):
        self.memory.store(state, action, reward, new_state, done)

    def choosen_action(self, state):
        state = state[np.newaxis, :]
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self._action_space)
        else:
            actions = self.q_eval.predict(state, verbose=0)
            action = np.argmax(actions)

        return action

    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return

        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        action_values = np.array(self._action_space, dtype=np.int8)
        action_indices = np.dot(action, action_values)  # intiger representation of an action

        q_eval = self.q_eval.predict(state, verbose=0)
        q_next = self.q_eval.predict(new_state, verbose=0)

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target[batch_index, action_indices] = reward + \
            self.gamma*np.max(q_next, axis=1) * done

        _ = self.q_eval.fit(state, q_target, verbose=0)

        self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > self.epsilon_end else self.epsilon_end

    def save_model(self):
        self.q_eval.save(self.model_file)

    def load_model(self):
        self.q_eval.load_model(self.model_file)