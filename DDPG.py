
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Actor(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(Actor, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size	

		self.linear1 = nn.Linear(input_size, hidden_size)
		self.linear2 = nn.Linear(hidden_size, hidden_size)
		self.linear3 = nn.Linear(hidden_size, output_size)

	def forward(self, x):
		output = F.relu(self.linear1(x))
		output = F.relu(self.linear2(output))
		output = self.linear3(output)

		return output

class Critic(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(Critic, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size

		self.linear1 = nn.Linear(input_size, hidden_size)
		self.linear2 = nn.Linear(hidden_size, hidden_size)
		self.linear3 = nn.Linear(hidden_size, output_size)

	def forward(self, state, action):
		action = action.view(1,-1)
		x = torch.cat((state, action), dim=1)
		output = F.relu(self.linear1(x))
		output = F.relu(self.linear2(output))
		output = self.linear3(output)

		return output

class DDPG(object):
	def __init__(self, num_state, num_action, lr=1e-3, gamma=0.99, actor_hidden_size=128, critic_hidden_size=128, tau=0.01):
		self.num_state = num_state
		self.num_action = num_action
		self.actor_hidden_size = actor_hidden_size
		self.critic_hidden_size = critic_hidden_size
		self.gamma = gamma
		self.tau = tau

		self.actor = Actor(self.num_state, self.actor_hidden_size, self.num_action)
		self.critic = Critic(self.num_state+self.num_action, self.critic_hidden_size, 1)

		self.actor_p = Actor(self.num_state, self.actor_hidden_size, self.num_action)
		self.critic_p = Critic(self.num_state+self.num_action, self.critic_hidden_size, 1)

		self.copy_parameters(self.actor_p, self.actor)
		self.copy_parameters(self.critic_p, self.critic)

		self.criterion = nn.MSELoss()
		self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
		self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)

		self.R = []
		self.Memory_size = 1000

	def buffer(self, query):
		self.R.append(query)
		if len(self.R) > self.Memory_size:
			self.R.pop(0)

	def copy_parameters(self, target, source):
		for target_param, param in zip(target.parameters(), source.parameters()):
			target_param.data.copy_(param.data)

	def update_paramters_tau(self, target, source, tau):
		for target_param, param in zip(target.parameters(), source.parameters()):
			target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)

	def selectAction(self, x):
		return self.actor(x)

	def _toTensor(self, x):
		return torch.Tensor(x).unsqueeze(0)

	def train(self, N=10):
		if len(self.R)==0:
			return
		loss_critic = 0
		batch = []
		for _ in range(N):
			i = np.random.randint(0, len(self.R))
			sample_traj = self.R[i]
			batch.append(sample_traj)
		for sample in batch:
			state = self._toTensor(sample[0])
			action = self._toTensor(sample[1])
			reward = sample[2]
			next_state = self._toTensor(sample[3])
			temp_action = self.actor_p(next_state).detach()
			y = self.gamma * self.critic_p.forward(next_state, temp_action) + reward
			# y = reward + self.gamma * self.critic_p.forward(next_state, temp_action)
			loss_critic = loss_critic + (y - self.critic(state, action))**2
		loss_critic = loss_critic / N
		self.critic_optim.zero_grad()
		loss_critic.backward()
		self.critic_optim.step()

		loss_actor = 0
		for sample in batch:
			state = self._toTensor(sample[0])
			action = self._toTensor(sample[1])
			loss_actor = loss_actor - self.critic(state, self.actor(state))
		loss_actor = loss_actor / N
		self.actor_optim.zero_grad()
		loss_actor.backward()
		self.actor_optim.step()

		self.update_paramters_tau(self.actor_p, self.actor, self.tau)
		self.update_paramters_tau(self.critic_p, self.critic, self.tau)





