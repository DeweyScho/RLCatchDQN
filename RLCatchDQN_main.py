import sys
import os
from CatchGameEnv import CatchGameEnv
import matplotlib.pyplot as plt
import collections
from PIL import Image
from DQNetwork import DQNetwork
import cv2
import numpy as np
import torch
import torch.optim as optim

# Hyperparameters
GAMMA = 0.99
INITIAL_EPSILON = 0.4
FINAL_EPSILON = 0.0001
MEMORY_SIZE = 50000
NUM_EPOCHS_OBSERVE = 50
NUM_EPOCHS_TRAIN = 8000
NUM_ACTIONS = 3
BATCH_SIZE = 32
NUM_EPOCHS = NUM_EPOCHS_OBSERVE + NUM_EPOCHS_TRAIN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_model(model, i, optim, fname):
    print("----------saving model-----------------")
    checkpoint_data = {
        'epoch': i,
        'state_dict': model.state_dict(),
        'optimizer': optim.state_dict()
    }
    ckpt_path = os.path.join("checkpoint", fname)
    torch.save(checkpoint_data, ckpt_path)
    model.train()

def convert_frame_to2darray(images):
    x_t = images[0]
    img = Image.fromarray(x_t)
    x_t = img.resize(size=(100, 100))
    x_t = np.array(x_t).T
    x_t = x_t.astype('float') / np.max(x_t)
    return x_t.reshape(1, 100, 100)

def get_next_batch(experience, model, num_actions, gamma, batch_size):
    batch_indices = np.random.randint(low=0, high=len(experience), size=batch_size)
    batch = [experience[i] for i in batch_indices]
    X = np.zeros((batch_size, 1, 100, 100))
    Y = np.zeros((batch_size, num_actions))
    A = np.zeros((batch_size, 1), dtype=int)

    for i in range(len(batch)):
        s_t, a_t, r_t, s_tp1, game_over = batch[i]
        X[i] = s_t
        A[i] = a_t
        Y[i] = model(torch.tensor(s_t).to(device).float()).cpu().detach()[0].numpy()
        Q_sa = np.max(model(torch.tensor(s_tp1).to(device).float()).cpu().detach()[0].numpy())
        if game_over:
            Y[i, a_t] = r_t
        else:
            Y[i, a_t] = r_t + gamma * Q_sa
    return X, Y, A

def main():
    model = DQNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_criterion = torch.nn.MSELoss()
    game = CatchGameEnv()
    experience = collections.deque(maxlen=MEMORY_SIZE)

    num_wins = 0
    epsilon = INITIAL_EPSILON
    loss_list = []
    num_wins_list = []
    Q_values = []

    for e in range(NUM_EPOCHS):
        game.reset()
        loss = 0.0
        Qmax = 0

        a_0 = 1  # Initial action: Stay
        game_over, x_t, r_t = game.mainGame(a_0)
        s_t = convert_frame_to2darray(x_t)

        while not game_over:
            model.eval()
            s_tm1 = s_t

            if e <= NUM_EPOCHS_OBSERVE or np.random.rand() <= epsilon:
                a_t = np.random.randint(low=0, high=NUM_ACTIONS)
            else:
                q_values = model(torch.tensor(s_t).to(device).float()).cpu().detach()[0].numpy()
                Qmax = np.max(q_values)
                a_t = np.argmax(q_values)

            game_over, x_t, r_t = game.mainGame(a_t)
            s_t = convert_frame_to2darray(x_t)

            if r_t == 1:
                num_wins += 1

            experience.append((s_tm1, a_t, r_t, s_t, game_over))

            if e > NUM_EPOCHS_OBSERVE:
                model.train()
                X, Y, A = get_next_batch(experience, model, NUM_ACTIONS, GAMMA, BATCH_SIZE)
                optimizer.zero_grad()
                outputs = model(torch.tensor(X).to(device).float())
                outs = torch.zeros((X.shape[0], Y.shape[1])).to(device)

                for i in range(X.shape[0]):
                    outs[i, A[i, 0]] = outputs[i, A[i, 0]]

                loss_net = loss_criterion(outs, torch.tensor(Y).to(device).float())
                loss_net.backward()
                optimizer.step()
                loss += loss_net.item()

            if epsilon > FINAL_EPSILON:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / NUM_EPOCHS

        print("Epoch : %d | Loss: %f | Win Count : %d | Qmax : %f" % (e, loss, num_wins, Qmax))
        loss_list.append(loss)
        num_wins_list.append(num_wins)
        Q_values.append(Qmax)

        if e % 1000 == 0:
            save_model(model, e, optimizer, "modelcatch.pt")

    # Plotting results
    output_dir = "d:/reinforcementlearning/RLCatchDQN/output"
    os.makedirs(output_dir, exist_ok=True)

    plt.plot(loss_list, label='loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(output_dir, "loss.png"))
    plt.clf()

    plt.plot(Q_values, label='Q values')
    plt.title('Q Values Over Episodes')
    plt.xlabel('Epoch')
    plt.ylabel('Q Value')
    plt.savefig(os.path.join(output_dir, "qvalue.png"))
    plt.clf()

    plt.plot(num_wins_list, label='Wins')
    plt.title('Number of Wins')
    plt.xlabel('Epoch')
    plt.ylabel('Wins')
    plt.savefig(os.path.join(output_dir, "wins.png"))

if __name__ == "__main__":
    sys.exit(int(main() or 0))

