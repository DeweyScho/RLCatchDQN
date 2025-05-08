# RLCatchDQN

A Deep Q-Learning (DQN) implementation in PyTorch for a reflex-based Catch game environment.
The agent learns to catch falling objects using pixel input and convolutional Q-networks.

Files:
- RLCatchDQN_main.py — Main training script
- DQNetwork.py — Convolutional neural network for Q-value estimation
- PlayCatchTrainedRL.py — Run trained model to play Catch
- CatchGame.py — Game logic
- CatchGameEnv.py — Environment wrapper (Gym-style)

- checkpoint/ — Folder to store saved models
- output/ — Optional output logs/visuals
- pytorch2x/ — Additional PyTorch tools (if used)

create checkpoint folder if training from scratch

Install dependencies:
pip install torch numpy matplotlib
