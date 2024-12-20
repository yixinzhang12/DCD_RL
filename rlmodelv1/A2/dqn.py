import gymnasium as gym
import math
import random
from itertools import count
import torch
from eval_policy import eval_policy, device
from model import MyModel
from replay_buffer import ReplayBuffer
import torch.nn.functional as F


BATCH_SIZE = 256
GAMMA = 0.99
EPS_EXPLORATION = 0.2
TARGET_UPDATE = 10
NUM_EPISODES = 4000
TEST_INTERVAL = 25
LEARNING_RATE = 10e-4
RENDER_INTERVAL = 20
ENV_NAME = 'CartPole-v1'
PRINT_INTERVAL = 10

env = gym.make(ENV_NAME)
state_shape = len(env.reset()[0])
n_actions = env.action_space.n

model = MyModel(state_shape, n_actions).to(device)
target = MyModel(state_shape, n_actions).to(device)
target.load_state_dict(model.state_dict())
target.eval()

optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
memory = ReplayBuffer()

def choose_action(state, test_mode=False):
    # TODO implement an epsilon-greedy strategy
    # raise NotImplementedError()
    epsilon = EPS_EXPLORATION if not test_mode else 0
    state_tensor = torch.tensor([state], dtype=torch.float32).to(device)
    if random.random() < epsilon:
        action = torch.tensor([[random.randrange(n_actions)]], dtype=torch.long).to(device)
    else:
        with torch.no_grad():
            q_values = model(state_tensor)
            action = torch.argmax(q_values, dim=1).unsqueeze(0)
    return action

def optimize_model(state, action, next_state, reward, done):
    # TODO given a tuple (s_t, a_t, s_{t+1}, r_t, done_t) update your model weights
    # tensor conversion
    state_tensor = torch.tensor([state], dtype=torch.float32).to(device)
    action_tensor = torch.tensor([[action]], dtype=torch.long).to(device)
    reward_tensor = torch.tensor([reward], dtype=torch.float32).to(device)
    next_state_tensor = torch.tensor([next_state], dtype=torch.float32).to(device)
    done_tensor = torch.tensor([done], dtype=torch.float32).to(device)

    # get q
    q_values = model(state_tensor)
    state_action_value = q_values.gather(1, action_tensor) #current q

    with torch.no_grad(): #target q
        next_q_values = target(next_state_tensor)
        next_state_value = next_q_values.max(1)[0]
        next_state_value = next_state_value * (1 - done_tensor)
        expected_state_action_value = reward_tensor + GAMMA * next_state_value

    loss = F.mse_loss(state_action_value, expected_state_action_value.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train_reinforcement_learning(render=False):
    steps_done = 0
    best_score = -float("inf")
    
    for i_episode in range(1, NUM_EPISODES+1):
        episode_total_reward = 0
        state, _ = env.reset()
        for t in count():
            action = choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy()[0][0])
            steps_done += 1
            episode_total_reward += reward

            memory.push(state, action.cpu().numpy()[0][0], next_state, reward, terminated)

            optimize_model(state, action, next_state, reward, terminated)

            state = next_state

            if render:
                env.render()

            if (terminated or truncated):
                if i_episode % PRINT_INTERVAL == 0:
                    print('[Episode {:4d}/{}] [Steps {:4d}] [reward {:.1f}]'
                        .format(i_episode, NUM_EPISODES, t, episode_total_reward))
                break

        if i_episode % TARGET_UPDATE == 0:
            target.load_state_dict(model.state_dict())

        if i_episode % TEST_INTERVAL == 0:
            print('-'*10)
            score = eval_policy(policy=model, env=ENV_NAME, render=render)
            if score > best_score:
                best_score = score
                torch.save(model.state_dict(), "best_model_{}.pt".format(ENV_NAME))
                print('saving model.')
            print("[TEST Episode {}] [Average Reward {}]".format(i_episode, score))
            print('-'*10)


if __name__ == "__main__":
    train_reinforcement_learning()
