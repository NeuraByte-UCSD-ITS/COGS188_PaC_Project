import gymnasium as gym
import numpy as np
import pandas as pd
import time
from q_dqn_doubledqn import DeepQNetworkAgent, preprocess_image_frame, discretize_observation_state

def evaluate(agent, env, buckets, num_episodes):
    total_rewards = []
    start_time = time.time()

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = preprocess_image_frame(state)
        state = discretize_observation_state(state, buckets)
        total_reward = 0
        while True:
            action = agent.choose_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
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
    env = gym.make('ALE/MsPacman-v5', render_mode='human') #with environment pop-up
    # env = gym.make('ALE/MsPacman-v5') #without environment pop-up
    state_shape = (84, 84, 1)
    buckets = np.linspace(0, 255, 10)
    num_episodes = 500

    evaluation_results = pd.DataFrame(columns=['Model', 'Average Reward', 'Stability', 'Computation Time'])
    print("Fine-Tuned DQN evaluation:")
    agent_finetuned = DeepQNetworkAgent(state_shape=state_shape, action_space_size=env.action_space.n)
    agent_finetuned.load_trained_model('dqn_model_final.weights.h5')
    avg_reward_finetuned, stability_finetuned, comp_time_finetuned = evaluate(agent_finetuned, env, buckets, num_episodes)
    evaluation_results = pd.concat([evaluation_results, pd.DataFrame({'Model': ['DQN Fine-Tuned'], 'Average Reward': [avg_reward_finetuned], 'Stability': [stability_finetuned], 'Computation Time': [comp_time_finetuned]})], ignore_index=True)

    env.close()
    evaluation_results.to_csv('finetuned_evaluation_results.csv', index=False)
    print("Evaluation saved to finetuned_evaluation_results.csv")

if __name__ == "__main__":
    main()
