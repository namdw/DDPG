import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import matplotlib.pyplot as plt

from DDPG import DDPG

def toTensor(nplist):
	return torch.Tensor(nplist).unsqueeze(0)

seed = 123
np.random.seed(seed)
torch.manual_seed(seed)

env = gym.make('Pendulum-v0')
# env = gym.make('MountainCarContinuous-v0')
num_state = len(env.observation_space.high)
num_action = len(env.action_space.high)

print(num_state, num_action)

ddpg = DDPG(num_state, num_action)
score_list = []

num_episode = 1000
num_step = 200
for episode in range(num_episode):
	state = env.reset()
	score = 0
	for step in range(num_step):
		env.render()
		prev_state = state
		action = ddpg.selectAction(toTensor(state))[0].detach().numpy()
		state, reward, done, _ = env.step(action)
		query = [prev_state, action, reward, state]
		ddpg.buffer(query)
		ddpg.train()

		score += reward
		if done:
			print('Episode {} finished with score of {}'.format(episode, score))
			break

plt.plot(score_list)
plt.show()