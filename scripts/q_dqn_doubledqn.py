import numpy as np
import random
import time
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers
import gymnasium as gym
import ale_py
import logging

#DQN 
class DeepQNetwork(tf.keras.Model):
    def __init__(self, action_space_size):
        super(DeepQNetwork, self).__init__()
        self.first_hidden_layer = layers.Dense(128, activation='relu')
        self.second_hidden_layer = layers.Dense(128, activation='relu')
        self.output_layer = layers.Dense(action_space_size, activation=None)

    def call(self, input_tensor):
        first_hidden_output = self.first_hidden_layer(input_tensor)
        second_hidden_output = self.second_hidden_layer(first_hidden_output)
        return self.output_layer(second_hidden_output)

#Double DQN 
class DoubleDeepQNetwork(tf.keras.Model):
    def __init__(self, action_space_size):
        super(DoubleDeepQNetwork, self).__init__()
        self.first_hidden_layer = layers.Dense(128, activation='relu')
        self.second_hidden_layer = layers.Dense(128, activation='relu')
        self.output_layer = layers.Dense(action_space_size, activation=None)

    def call(self, input_tensor):
        first_hidden_output = self.first_hidden_layer(input_tensor)
        second_hidden_output = self.second_hidden_layer(first_hidden_output)
        return self.output_layer(second_hidden_output)

#Experience memory for DQN & Double DQN
class ExperienceReplayBuffer:
    def __init__(self, buffer_max_size, batch_sample_size):
        self.experience_buffer = deque(maxlen=buffer_max_size)
        self.batch_sample_size = batch_sample_size

    def store_experience(self, state, action, reward, next_state, done):
        self.experience_buffer.append((state, action, reward, next_state, done))

    def sample_experience(self):
        experience_batch = random.sample(self.experience_buffer, self.batch_sample_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*experience_batch))
        return states, actions, rewards, next_states, dones

    def buffer_size(self):
        return len(self.experience_buffer)

#Preprocess image frame from environment for DQN & Double DQN agents
def preprocess_image_frame(image_frame):
    grayscale_image_frame = np.mean(image_frame, axis=2).astype(np.uint8)  
    downsampled_image_frame = grayscale_image_frame[::2, ::2]  
    return downsampled_image_frame

#Discretize observation state into buckets for DQN & Double DQN agents
def discretize_observation_state(observation_state, bucket_thresholds):
    flattened_state = observation_state.flatten()  
    bucket_indices = np.digitize(flattened_state, bucket_thresholds) - 1  
    max_index = len(bucket_thresholds) - 1
    bucket_indices = np.clip(bucket_indices, 0, max_index)
    return bucket_indices

class DeepQNetworkAgent:
    def __init__(self, state_shape, action_space_size, replay_buffer_size=10000, batch_size=64, discount_factor=0.99, learning_rate=0.001, exploration_rate=1.0, exploration_decay=0.995, exploration_min=0.01):
        self.state_shape = state_shape
        self.action_space_size = action_space_size
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.exploration_min = exploration_min
        self.batch_size = batch_size

        self.policy_network = DeepQNetwork(action_space_size)
        self.target_network = DeepQNetwork(action_space_size)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.replay_buffer = ExperienceReplayBuffer(replay_buffer_size, batch_size)
        self.update_target_network()

    def update_target_network(self):
        self.target_network.set_weights(self.policy_network.get_weights())

    def choose_action(self, current_state):
        if np.random.rand() < self.exploration_rate:
            return np.random.choice(self.action_space_size)
        q_values = self.policy_network(tf.convert_to_tensor([current_state], dtype=tf.float32))
        return np.argmax(q_values.numpy())

    def learn_from_experience(self):
        if self.replay_buffer.buffer_size() < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample_experience()
        target_q_values = self.target_network(tf.convert_to_tensor(next_states, dtype=tf.float32))
        max_target_q_values = np.max(target_q_values.numpy(), axis=1)
        target_values = rewards + (1 - dones) * self.discount_factor * max_target_q_values

        with tf.GradientTape() as gradient_tape:
            q_values = self.policy_network(tf.convert_to_tensor(states, dtype=tf.float32))
            indices = np.arange(self.batch_size)
            action_q_values = tf.convert_to_tensor([q_values[i][actions[i]] for i in indices])
            loss = tf.keras.losses.MSE(action_q_values, target_values)

        gradients = gradient_tape.gradient(loss, self.policy_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.policy_network.trainable_variables))
        self.exploration_rate = max(self.exploration_min, self.exploration_rate * self.exploration_decay)

    def store_experience_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.store_experience(state, action, reward, next_state, done)

    def save_trained_model(self, filepath):
        self.policy_network.save_weights(filepath)

    def load_trained_model(self, filepath):
        self.policy_network.load_weights(filepath)
        self.update_target_network()

class DoubleDeepQNetworkAgent:
    def __init__(self, state_shape, action_space_size, replay_buffer_size=10000, batch_size=64, discount_factor=0.99, learning_rate=0.001, exploration_rate=1.0, exploration_decay=0.995, exploration_min=0.01):
        self.state_shape = state_shape
        self.action_space_size = action_space_size
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.exploration_min = exploration_min
        self.batch_size = batch_size

        self.policy_network = DoubleDeepQNetwork(action_space_size)
        self.target_network = DoubleDeepQNetwork(action_space_size)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.replay_buffer = ExperienceReplayBuffer(replay_buffer_size, batch_size)
        self.update_target_network()

    def update_target_network(self):
        self.target_network.set_weights(self.policy_network.get_weights())

    def choose_action(self, current_state):
        if np.random.rand() < self.exploration_rate:
            return np.random.choice(self.action_space_size)
        q_values = self.policy_network(tf.convert_to_tensor([current_state], dtype=tf.float32))
        return np.argmax(q_values.numpy())

    def learn_from_experience(self):
        if self.replay_buffer.buffer_size() < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample_experience()
        next_actions = np.argmax(self.policy_network(tf.convert_to_tensor(next_states, dtype=tf.float32)).numpy(), axis=1)
        target_q_values = self.target_network(tf.convert_to_tensor(next_states, dtype=tf.float32))
        double_q_values = target_q_values.numpy()[np.arange(self.batch_size), next_actions]
        target_values = rewards + (1 - dones) * self.discount_factor * double_q_values

        with tf.GradientTape() as gradient_tape:
            q_values = self.policy_network(tf.convert_to_tensor(states, dtype=tf.float32))
            indices = np.arange(self.batch_size)
            action_q_values = tf.convert_to_tensor([q_values[i][actions[i]] for i in indices])
            loss = tf.keras.losses.MSE(action_q_values, target_values)

        gradients = gradient_tape.gradient(loss, self.policy_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.policy_network.trainable_variables))
        self.exploration_rate = max(self.exploration_min, self.exploration_rate * self.exploration_decay)

    def store_experience_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.store_experience(state, action, reward, next_state, done)

    def save_trained_model(self, filepath):
        self.policy_network.save_weights(filepath)

    def load_trained_model(self, filepath):
        self.policy_network.load_weights(filepath)
        self.update_target_network()

#Qlearning
class QLearningAgent:
    def __init__(self, state_space_size, action_space_size, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995, exploration_min=0.01):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.exploration_min = exploration_min
        self.q_table = np.zeros((state_space_size, action_space_size))

    def choose_action(self, current_state):
        if np.random.rand() < self.exploration_rate:
            return np.random.choice(self.action_space_size)
        return np.argmax(self.q_table[current_state])

    def learn_from_experience(self, current_state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        temporal_difference_target = reward + self.discount_factor * self.q_table[next_state, best_next_action]
        temporal_difference_error = temporal_difference_target - self.q_table[current_state, action]
        self.q_table[current_state, action] += self.learning_rate * temporal_difference_error
        self.exploration_rate = max(self.exploration_min, self.exploration_rate * self.exploration_decay)

    def save_q_table(self, filepath):
        with open(filepath, 'wb') as f:
            np.save(f, self.q_table)

    def load_q_table(self, filepath):
        with open(filepath, 'rb') as f:
            self.q_table = np.load(f)

def create_environment():
    environment = gym.make('ALE/MsPacman-v5') #with environment pop-up
    #environment = gym.make('ALE/MsPacman-v5', render_mode='human') #with environment pop-up
    return environment

def train_deep_q_network():
    print("Running: train_deep_q_network()")
    environment = create_environment()
    state_shape = (84, 84, 1)
    agent = DeepQNetworkAgent(state_shape=state_shape, action_space_size=environment.action_space.n)
    number_of_episodes = 1000
    target_network_update_frequency = 10
    model_save_frequency = 100
    state_buckets = np.linspace(0, 255, 10)

    total_rewards = []
    start_time = time.time()
    for episode_number in range(number_of_episodes):
        current_state, _ = environment.reset()
        current_state = preprocess_image_frame(current_state)
        current_state = discretize_observation_state(current_state, state_buckets)
        total_episode_reward = 0

        while True:
            chosen_action = agent.choose_action(current_state)
            next_state, reward, done, truncated, info = environment.step(chosen_action)
            next_state = preprocess_image_frame(next_state)
            next_state = discretize_observation_state(next_state, state_buckets)
            agent.store_experience_transition(current_state, chosen_action, reward, next_state, done)
            agent.learn_from_experience()
            current_state = next_state
            total_episode_reward += reward
            if done or truncated:
                break
        total_rewards.append(total_episode_reward)
        if episode_number % target_network_update_frequency == 0:
            agent.update_target_network()
        if episode_number % model_save_frequency == 0:
            agent.save_trained_model(f'dqn_model_episode_{episode_number}.weights.h5')

    end_time = time.time()
    training_time = end_time - start_time
    agent.save_trained_model('dqn_model_final.weights.h5')
    environment.close()
    convergence_rate = compute_convergence_rate(total_rewards)
    stability = compute_stability(total_rewards)
    print("Finished: train_deep_q_network()")
    return convergence_rate, stability, training_time

def compute_convergence_rate(total_rewards, threshold=0.01):
    mean_rewards = np.convolve(total_rewards, np.ones((100,))/100, mode='valid')
    if len(mean_rewards) == 0:
        return len(total_rewards)
    convergence_episode = np.where(np.abs(np.diff(mean_rewards)) < threshold)[0]
    return convergence_episode[0] if len(convergence_episode) > 0 else len(total_rewards)

def compute_stability(total_rewards):
    mean_rewards = np.convolve(total_rewards, np.ones((100,))/100, mode='valid')
    return np.std(mean_rewards)

def train_deep_q_network_with_hyperparameters():
    print("Running: train_deep_q_network_with_hyperparameters()")
    environment = create_environment()
    state_shape = (84, 84, 1)
    agent = DeepQNetworkAgent(state_shape=state_shape, action_space_size=environment.action_space.n, replay_buffer_size=50000, batch_size=128, learning_rate=0.0005)
    number_of_episodes = 1000
    target_network_update_frequency = 10
    model_save_frequency = 100
    state_buckets = np.linspace(0, 255, 10)

    total_rewards = []
    start_time = time.time()
    for episode_number in range(number_of_episodes):
        current_state, _ = environment.reset()
        current_state = preprocess_image_frame(current_state)
        current_state = discretize_observation_state(current_state, state_buckets)
        total_episode_reward = 0
        while True:
            chosen_action = agent.choose_action(current_state)
            next_state, reward, done, truncated, info = environment.step(chosen_action)
            next_state = preprocess_image_frame(next_state)
            next_state = discretize_observation_state(next_state, state_buckets)
            agent.store_experience_transition(current_state, chosen_action, reward, next_state, done)
            agent.learn_from_experience()
            current_state = next_state
            total_episode_reward += reward
            if done or truncated:
                break
        total_rewards.append(total_episode_reward)
        if episode_number % target_network_update_frequency == 0:
            agent.update_target_network()
        if episode_number % model_save_frequency == 0:
            agent.save_trained_model(f'dqn_model_hyperparameters_episode_{episode_number}.weights.h5')

    end_time = time.time()
    training_time = end_time - start_time
    agent.save_trained_model('dqn_model_hyperparameters_final.weights.h5')
    environment.close()
    convergence_rate = compute_convergence_rate(total_rewards)
    stability = compute_stability(total_rewards)
    print("Finished: train_deep_q_network_with_hyperparameters()")
    return convergence_rate, stability, training_time

def train_deep_q_network_for_exploitation():
    print("Running: train_deep_q_network_for_exploitation()")
    environment = create_environment()
    state_shape = (84, 84, 1)
    agent = DeepQNetworkAgent(state_shape=state_shape, action_space_size=environment.action_space.n)
    agent.exploration_rate = 0  #0 for pure exploitation, no exploration
    number_of_episodes = 1000
    target_network_update_frequency = 10
    model_save_frequency = 100
    state_buckets = np.linspace(0, 255, 10)

    total_rewards = []
    start_time = time.time()
    for episode_number in range(number_of_episodes):
        current_state, _ = environment.reset()
        current_state = preprocess_image_frame(current_state)
        current_state = discretize_observation_state(current_state, state_buckets)
        total_episode_reward = 0
        while True:
            chosen_action = agent.choose_action(current_state)
            next_state, reward, done, truncated, info = environment.step(chosen_action)
            next_state = preprocess_image_frame(next_state)
            next_state = discretize_observation_state(next_state, state_buckets)
            agent.store_experience_transition(current_state, chosen_action, reward, next_state, done)
            agent.learn_from_experience()
            current_state = next_state
            total_episode_reward += reward
            if done or truncated:
                break
        total_rewards.append(total_episode_reward)
        if episode_number % target_network_update_frequency == 0:
            agent.update_target_network()
        if episode_number % model_save_frequency == 0:
            agent.save_trained_model(f'dqn_model_exploitation_episode_{episode_number}.weights.h5')

    end_time = time.time()
    training_time = end_time - start_time
    agent.save_trained_model('dqn_model_exploitation_final.weights.h5')
    environment.close()
    convergence_rate = compute_convergence_rate(total_rewards)
    stability = compute_stability(total_rewards)
    print("Finished: train_deep_q_network_for_exploitation()")
    return convergence_rate, stability, training_time

def train_deep_q_network_for_exploration():
    print("Running: train_deep_q_network_for_exploration()")
    environment = create_environment()
    state_shape = (84, 84, 1)
    agent = DeepQNetworkAgent(state_shape=state_shape, action_space_size=environment.action_space.n)
    agent.exploration_rate = 1  #1 for pure exploration, no exploitation
    number_of_episodes = 1000
    target_network_update_frequency = 10
    model_save_frequency = 100
    state_buckets = np.linspace(0, 255, 10)

    total_rewards = []
    start_time = time.time()
    for episode_number in range(number_of_episodes):
        current_state, _ = environment.reset()
        current_state = preprocess_image_frame(current_state)
        current_state = discretize_observation_state(current_state, state_buckets)
        total_episode_reward = 0
        while True:
            chosen_action = agent.choose_action(current_state)
            next_state, reward, done, truncated, info = environment.step(chosen_action)
            next_state = preprocess_image_frame(next_state)
            next_state = discretize_observation_state(next_state, state_buckets)
            agent.store_experience_transition(current_state, chosen_action, reward, next_state, done)
            agent.learn_from_experience()
            current_state = next_state
            total_episode_reward += reward
            if done or truncated:
                break
        total_rewards.append(total_episode_reward)
        if episode_number % target_network_update_frequency == 0:
            agent.update_target_network()
        if episode_number % model_save_frequency == 0:
            agent.save_trained_model(f'dqn_model_exploration_episode_{episode_number}.weights.h5')

    end_time = time.time()
    training_time = end_time - start_time
    agent.save_trained_model('dqn_model_exploration_final.weights.h5')
    environment.close()
    convergence_rate = compute_convergence_rate(total_rewards)
    stability = compute_stability(total_rewards)
    print("Finished: train_deep_q_network_for_exploration()")
    return convergence_rate, stability, training_time

def train_double_deep_q_network():
    print("Running: train_double_deep_q_network()")
    environment = create_environment()
    state_shape = (84, 84, 1)
    agent = DoubleDeepQNetworkAgent(state_shape=state_shape, action_space_size=environment.action_space.n)
    number_of_episodes = 1000
    target_network_update_frequency = 10
    model_save_frequency = 100
    state_buckets = np.linspace(0, 255, 10)

    total_rewards = []
    start_time = time.time()
    for episode_number in range(number_of_episodes):
        current_state, _ = environment.reset()
        current_state = preprocess_image_frame(current_state)
        current_state = discretize_observation_state(current_state, state_buckets)
        total_episode_reward = 0
        while True:
            chosen_action = agent.choose_action(current_state)
            next_state, reward, done, truncated, info = environment.step(chosen_action)
            next_state = preprocess_image_frame(next_state)
            next_state = discretize_observation_state(next_state, state_buckets)
            agent.store_experience_transition(current_state, chosen_action, reward, next_state, done)
            agent.learn_from_experience()
            current_state = next_state
            total_episode_reward += reward
            if done or truncated:
                break
        total_rewards.append(total_episode_reward)
        if episode_number % target_network_update_frequency == 0:
            agent.update_target_network()
        if episode_number % model_save_frequency == 0:
            agent.save_trained_model(f'double_dqn_model_episode_{episode_number}.weights.h5')

    end_time = time.time()
    training_time = end_time - start_time
    agent.save_trained_model('double_dqn_model_final.weights.h5')
    environment.close()
    convergence_rate = compute_convergence_rate(total_rewards)
    stability = compute_stability(total_rewards)
    print("Finished: train_double_deep_q_network()")
    return convergence_rate, stability, training_time

def train_q_learning_agent():
    print("Running: train_q_learning_agent()")
    environment = create_environment()
    state_buckets = np.linspace(0, 255, 10)
    state_space_size = len(state_buckets) ** 2  
    agent = QLearningAgent(state_space_size=state_space_size, action_space_size=environment.action_space.n)
    number_of_episodes = 1000
    model_save_frequency = 100

    total_rewards = []
    start_time = time.time()
    for episode_number in range(number_of_episodes):
        current_state, _ = environment.reset()
        current_state = preprocess_image_frame(current_state)
        current_state = discretize_observation_state(current_state, state_buckets)
        current_state = np.ravel_multi_index(current_state[:2], (len(state_buckets), len(state_buckets)))
        total_episode_reward = 0
        while True:
            chosen_action = agent.choose_action(current_state)
            next_state, reward, done, truncated, info = environment.step(chosen_action)
            next_state = preprocess_image_frame(next_state)
            next_state = discretize_observation_state(next_state, state_buckets)
            next_state = np.ravel_multi_index(next_state[:2], (len(state_buckets), len(state_buckets)))
            agent.learn_from_experience(current_state, chosen_action, reward, next_state)
            current_state = next_state
            total_episode_reward += reward
            if done or truncated:
                break
        total_rewards.append(total_episode_reward)
        if episode_number % model_save_frequency == 0:
            agent.save_q_table(f'q_learning_model_episode_{episode_number}.weights.h5')

    end_time = time.time()
    training_time = end_time - start_time
    agent.save_q_table('q_learning_model_final.weights.h5')
    environment.close()
    convergence_rate = compute_convergence_rate(total_rewards)
    stability = compute_stability(total_rewards)
    print("Finished: train_q_learning_agent()")
    return convergence_rate, stability, training_time

if __name__ == "__main__":
    ###### ATTENTION: comment out the model you want to train one at a time######
    
    #DQN Model-Free
    # convergence_rate, stability, training_time = train_deep_q_network()
    # with open("DQN_training_metrics.csv", "w") as f:
    #     f.write("Model,Convergence Rate,Stability,Training Time\n")
    #     f.write(f"DQN,{convergence_rate},{stability},{training_time}\n")
    
    #DQN Hyperparameters
    # convergence_rate, stability, training_time = train_deep_q_network_with_hyperparameters()
    # with open("DQNhyperparameters_training_metrics.csv", "w") as f:
    #     f.write("Model,Convergence Rate,Stability,Training Time\n")
    #     f.write(f"DQNhyperparameters,{convergence_rate},{stability},{training_time}\n")
    
    #DQN Exploitation
    # convergence_rate, stability, training_time = train_deep_q_network_for_exploitation()
    # with open("DQNexploitation_training_metrics.csv", "w") as f:
    #     f.write("Model,Convergence Rate,Stability,Training Time\n")
    #     f.write(f"DQNexploitation,{convergence_rate},{stability},{training_time}\n")
    
    #DQN Exploration
    # convergence_rate, stability, training_time = train_deep_q_network_for_exploration()
    # with open("DQNexploration_training_metrics.csv", "w") as f:
    #     f.write("Model,Convergence Rate,Stability,Training Time\n")
    #     f.write(f"DQNexploration,{convergence_rate},{stability},{training_time}\n")
    
    #Double DQN
    # convergence_rate, stability, training_time = train_double_deep_q_network()
    # with open("DoubleDQN_training_metrics.csv", "w") as f:
    #     f.write("Model,Convergence Rate,Stability,Training Time\n")
    #     f.write(f"DoubleDQN,{convergence_rate},{stability},{training_time}\n")
    
    #Q-Learning 
    # convergence_rate, stability, training_time = train_q_learning_agent()
    # with open("QLearning_training_metrics.csv", "w") as f:
    #     f.write("Model,Convergence Rate,Stability,Training Time\n")
    #     f.write(f"QLearning,{convergence_rate},{stability},{training_time}\n")
        
    pass

