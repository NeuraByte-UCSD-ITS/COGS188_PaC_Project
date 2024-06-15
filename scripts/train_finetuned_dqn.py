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
        self.first_hidden_layer = layers.Dense(256, activation='relu')  #increased hidden layer size for fine tuning iteration 1
        self.second_hidden_layer = layers.Dense(256, activation='relu') #increased hidden layer size for fine tuning iteration 1
        self.output_layer = layers.Dense(action_space_size, activation=None)

    def call(self, input_tensor):
        first_hidden_output = self.first_hidden_layer(input_tensor)
        second_hidden_output = self.second_hidden_layer(first_hidden_output)
        return self.output_layer(second_hidden_output)

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

def preprocess_image_frame(image_frame):
    grayscale_image_frame = np.mean(image_frame, axis=2).astype(np.uint8)
    downsampled_image_frame = grayscale_image_frame[::2, ::2]
    return downsampled_image_frame

def discretize_observation_state(observation_state, bucket_thresholds):
    flattened_state = observation_state.flatten()
    bucket_indices = np.digitize(flattened_state, bucket_thresholds) - 1
    max_index = len(bucket_thresholds) - 1
    bucket_indices = np.clip(bucket_indices, 0, max_index)
    return bucket_indices

#increased replay buffer size, batch size, and decreased learning rate for fine tuning iteration 1
class DeepQNetworkAgent:
    def __init__(self, state_shape, action_space_size, replay_buffer_size=20000, batch_size=128, discount_factor=0.99, learning_rate=0.0005, exploration_rate=1.0, exploration_decay=0.995, exploration_min=0.01):
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

#Registering all Atari environments, but comment out once registered to avoid re-registering/computation
# ale_py.register_v0_v4_envs() 

# print("Available environments after registration:", list(gym.envs.registry.keys()))

def create_environment():
    environment = gym.make('ALE/MsPacman-v5')
    return environment

def train_deep_q_network():
    print("Running: train_deep_q_network()")
    environment = create_environment()
    state_shape = (84, 84, 1)
    agent = DeepQNetworkAgent(state_shape=state_shape, action_space_size=environment.action_space.n)
    number_of_episodes = 1000
    target_network_update_frequency = 5  #decreased for more frequent target network updates for fine tuning iteration 1
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

def train_deep_q_network():
    print("Running: train_deep_q_network()")
    environment = create_environment()
    state_shape = (84, 84, 1)
    agent = DeepQNetworkAgent(state_shape=state_shape, action_space_size=environment.action_space.n)
    number_of_episodes = 1000
    target_network_update_frequency = 5  #decreased for more frequent target network updates for fine tuning iteration 1
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

if __name__ == "__main__":
    convergence_rate, stability, training_time = train_deep_q_network()
    with open("DQN_training_metrics_finetuned.csv", "w") as f:
        f.write("Model,Convergence Rate,Stability,Training Time\n")
        f.write(f"DQN_FineTuned,{convergence_rate},{stability},{training_time}\n")

