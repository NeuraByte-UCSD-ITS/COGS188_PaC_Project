import gymnasium as gym
import numpy as np
import pandas as pd
import time
from dqn_agent import DQNAgent
from q_dqn_doubledqn import QLearningAgent, preprocess_image_frame, discretize_observation_state
import ale_py

#Atari environments registry
ale_py.register_v0_v4_envs() #comment after 1st run to prevent re-registering

def evaluate(agent, env, buckets, num_episodes):
    total_rewards = []
    start_time = time.time()

    for episode in range(num_episodes):
        state, placeholder_state = env.reset()
        state = preprocess_image_frame(state)
        state = discretize_observation_state(state, buckets)
        total_reward = 0
        while True:
            action = agent.choose_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            next_state = preprocess_image_frame(next_state)
            next_state = discretize_observation_state(next_state, buckets)
            state = next_state
            total_reward += reward
            if done or truncated:
                break
        total_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

    end_time = time.time()
    avg_reward = np.mean(total_rewards)
    stability = np.std(total_rewards)
    computation_time = end_time - start_time
    print(f"Average Reward over {num_episodes} episodes: {avg_reward}")
    return avg_reward, stability, computation_time

def main():
    env = gym.make('ALE/MsPacman-v5', render_mode='human')
    state_shape = (84, 84, 1)
    buckets = np.linspace(0, 255, 10)
    num_episodes = 500

    #Evaluation results DF
    evaluation_results = pd.DataFrame(columns=['Model', 'Average Reward', 'Stability', 'Computation Time'])

    #DQN Model-Free evaluation
    print("DQN Model-Free evaluation:")
    agent_dqn = DQNAgent(state_shape=state_shape, action_space_size=env.action_space.n)
    agent_dqn.load_model('dqn_model_final.weights.h5')
    avg_reward_dqn, stability_dqn, comp_time_dqn = evaluate(agent_dqn, env, buckets, num_episodes)
    evaluation_results = pd.concat([evaluation_results, pd.DataFrame({'Model': ['DQN Model-Free'], 'Average Reward': [avg_reward_dqn], 'Stability': [stability_dqn], 'Computation Time': [comp_time_dqn]})], ignore_index=True)

    #DQN Exploitation evaluation
    print("DQN Exploitation evaluation:")
    agent_exploitation = DQNAgent(state_shape=state_shape, action_space_size=env.action_space.n)
    agent_exploitation.load_model('dqn_model_exploitation_final.weights.h5')
    avg_reward_exploitation, stability_exploitation, comp_time_exploitation = evaluate(agent_exploitation, env, buckets, num_episodes)
    evaluation_results = pd.concat([evaluation_results, pd.DataFrame({'Model': ['DQN Exploitation'], 'Average Reward': [avg_reward_exploitation], 'Stability': [stability_exploitation], 'Computation Time': [comp_time_exploitation]})], ignore_index=True)

    #DQN Exploration evaluation
    print("DQN Exploration evaluation:")
    agent_exploration = DQNAgent(state_shape=state_shape, action_space_size=env.action_space.n)
    agent_exploration.load_model('dqn_model_exploration_final.weights.h5')
    avg_reward_exploration, stability_exploration, comp_time_exploration = evaluate(agent_exploration, env, buckets, num_episodes)
    evaluation_results = pd.concat([evaluation_results, pd.DataFrame({'Model': ['DQN Exploration'], 'Average Reward': [avg_reward_exploration], 'Stability': [stability_exploration], 'Computation Time': [comp_time_exploration]})], ignore_index=True)

    #DQN Hyperparameters evaluation
    print("DQN Hyperparameters evaluation:")
    agent_hyperparameters = DQNAgent(state_shape=state_shape, action_space_size=env.action_space.n)
    agent_hyperparameters.load_model('dqn_model_hyperparameters_final.weights.h5')
    avg_reward_hyperparameters, stability_hyperparameters, comp_time_hyperparameters = evaluate(agent_hyperparameters, env, buckets, num_episodes)
    evaluation_results = pd.concat([evaluation_results, pd.DataFrame({'Model': ['DQN Hyperparameters'], 'Average Reward': [avg_reward_hyperparameters], 'Stability': [stability_hyperparameters], 'Computation Time': [comp_time_hyperparameters]})], ignore_index=True)

    #Double DQN evaluation
    print("Double DQN evaluation:")
    agent_double_dqn = DQNAgent(state_shape=state_shape, action_space_size=env.action_space.n)
    agent_double_dqn.load_model('double_dqn_model_final.weights.h5')
    avg_reward_double_dqn, stability_double_dqn, comp_time_double_dqn = evaluate(agent_double_dqn, env, buckets, num_episodes)
    evaluation_results = pd.concat([evaluation_results, pd.DataFrame({'Model': ['Double DQN'], 'Average Reward': [avg_reward_double_dqn], 'Stability': [stability_double_dqn], 'Computation Time': [comp_time_double_dqn]})], ignore_index=True)

    #Q-learning Model-Free evaluation
    print("Q-learning Model-Free evaluation:")
    state_space_size = len(buckets) ** 2  
    agent_q_learning = QLearningAgent(state_space_size=state_space_size, action_space_size=env.action_space.n)
    agent_q_learning.load_q_table('q_learning_model_final.weights.h5')
    avg_reward_q_learning, stability_q_learning, comp_time_q_learning = evaluate(agent_q_learning, env, buckets, num_episodes)
    evaluation_results = pd.concat([evaluation_results, pd.DataFrame({'Model': ['Q-learning Model-Free'], 'Average Reward': [avg_reward_q_learning], 'Stability': [stability_q_learning], 'Computation Time': [comp_time_q_learning]})], ignore_index=True)

    env.close()

    #saving results to .csv file
    evaluation_results.to_csv('evaluation_results.csv', index=False)
    print("Evaluation saved to evaluation_results.csv")

if __name__ == "__main__":
    main()

