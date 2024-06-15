# PaC - COGNITIVE SCIENCE 188 FINAL PROJECT

# Instructions to run code


### 1. install all packages/dependencies 

- #### 1.a) open terminal and execute the following commands in the root directory of the project to create a virtual environment with required project dependencies
	- `python -m venv finalproject`
	- `source finalproject/bin/activate`
	- `pip install -r requirements.txt`

### 2. cd into `scripts` folder and execute the training script for all 6 algorithms to generate the first 3 metrics results, and weights for all models, one by one
    - uncomment lines 490-493 for DQN Model-Free model
    - Execute script:
    		- `python q_dqn_doubledqn.py`
    - After script is finished, comment back lines 490-493 and uncomment lines 496-499 for DQN Hyperparameters model
    - Execute script:
    		- `python q_dqn_doubledqn.py`
    - repeat this procedure for remaining models until all 3 metrics and weights are generated for all models: DQN Exploitation, DQN Exploration, Double DQN, and Q-learning models

### 3. Execute the evaluation script to generate average reward metric results and all model evaluations 
    - python evaluate_dqn.py

### 4. Execute the training script for the first fine tuned DQN model to generate its first 3 metrics results, and weights
	- python train_finetuned_dqn.py

### 5. Execute the evaluation script to generate average reward metric results and evaluations for first fine tuned model
	- python finetuned_evaluation.py

### 6. Execute the training script for the second fine tuned DQN model to generate its first 3 metrics results, and weights
	- python train_finetuned_dqn.py

### 7. Execute the evaluation script to generate average reward metric results and evaluations for second fine tuned model
	- python finetuned_evaluation2.py

### Note: this requires large computation resources, I ran the script for several days in both local and google doc and it resulted in a killed file `zsh: killed python train_finetuned_dqn.py` locally and runtime stop on google doc

### 9. run evaluation script to generate average reward metric results and all (A2C, PPO, DQN) model evaluations for new algorithms and environment implementation
    - python train_finetuned_dqn2.py

# References

1. [DQN — Stable Baselines 2.10.3a0 documentation](https://stable-baselines.readthedocs.io/en/master/modules/dqn.html)
2. [The Deep Q-Learning Algorithm](https://huggingface.co/learn/deep-rl-course/en/unit3/deep-q-algorithm)
3. [Reinforcement Learning (DQN) Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
4. [Deep Q Learning Examples](https://github.com/rajibhossen/dqn-examples)
5. [Deep-Q-Network](https://github.com/topics/deep-q-network)
6. [17.3. Q-Learning](https://d2l.ai/chapter_reinforcement-learning/qlearning.html)
7. [Introducing Q-Learning](https://huggingface.co/learn/deep-rl-course/en/unit2/q-learning)
8. [Reinforcement Learning](https://ml-cheatsheet.readthedocs.io/en/latest/reinforcement_learning.html)
9. [How to Train Ms-Pacman with Reinforcement Learning](https://medium.com/analytics-vidhya/how-to-train-ms-pacman-with-reinforcement-learning-dea714a2365e)