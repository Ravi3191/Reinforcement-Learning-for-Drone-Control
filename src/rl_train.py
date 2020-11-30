import numpy as np
import torch
import os
import torchvision.models as models
import torch.optim as optim
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import torchvision
import yaml
import copy

from torch.utils.tensorboard import SummaryWriter

with open('config.yaml') as file:
  arg_params = yaml.load(file)


import memory_buffer as mem_buf
import Flightmare_bridge as FM
import rl_models

from datetime import datetime


class Learner():

	def __init__():

		self.env = FM.FlightMare(arg_params)

		self.device = "cpu"

		if (arg_params['GPU']):
			if(torch.cuda.is_available()):
				self.device = "cuda:0"

		date = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
		path = arg_params['logging_path'] + date[:2] + date[3:5] + date[5:7] + date[6:10] + date[11:13] + date[14:16] + date[17:19] + '/'

		try:
			os.mkdir(path)
		except:
			print('unable to create directory')
			exit()


		self.logging = SummaryWriter(path)

		self.steps = 0

		self.value_net        = rl_models.ValueNetwork(arg_params['n_channels'],arg_params['state_dims']).to(self.device)

		self.target_value_net = rl_models.ValueNetwork(arg_params['n_channels'],arg_params['state_dims']).to(self.device)

		self.soft_q_net1 = rl_models.SoftQNetwork(arg_params['n_channels'],arg_params['velocity_dim'] + arg_params['heading_dim'],\
		                                arg_params[state_dims]).to(self.device)
		self.soft_q_net2 = rl_models.SoftQNetwork(arg_params['n_channels'],arg_params['velocity_dim'] + arg_params['heading_dim'],\
		                                arg_params['state_dims']).to(self.device)
		self.policy_net = rl_models.PolicyNetwork(arg_params['n_channels'],arg_params['velocity_dim'] + arg_params['heading_dim'],\
		                                arg_params['state_dims'], arg_params['latent_dim']).to(self.device)

		for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
		    target_param.data.copy_(param.data)
		    

		self.value_criterion  = nn.MSELoss()
		self.soft_q_criterion1 = nn.MSELoss()
		self.soft_q_criterion2 = nn.MSELoss()

		self.value_optimizer  = optim.Adam(self.value_net.parameters(), lr = arg_params['value_lr'])
		self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr = arg_params['q_lr'])
		self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr = arg_params['q_lr'])
		self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr = arg_params['policy_lr'])

		self.replay_buffer = mem_buf.ReplayBuffer_dir(arg_params)
		self.batch_size = arg_params['batch_size']

	def update(self,gamma=0.99,soft_tau=1e-2):
	    
	    im, state, goal, action, reward, next_im, next_state, done = self.replay_buffer.sample()

	    im, state, goal, action, reward, next_im, next_state, done = im.to(self.device), state.to(self.device), goal.to(self.device),\
	     action.to(self.device), reward.to(self.device), next_im.to(self.device), next_state.to(self.device), done.to(self.device)

	    predicted_q_value1 = self.soft_q_net1(im, curr_pos, goal,action)
	    predicted_q_value2 = self.soft_q_net2(im, curr_pos ,goal,action)
	    predicted_value    = self.value_net(im, curr_pos, goal)
	    new_action, log_prob, epsilon, mean, log_std = self.policy_net.evaluate(im, curr_pos, goal)

	        
	# Training Q Function
	    target_value = self.target_value_net(next_im,next_state,goal)
	    target_q_value = reward + (1 - done) * gamma * target_value
	    q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())
	    q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())

	    self.logging.add_scalar('Loss/softq1',q_value_loss1,self.steps)
	    self.logging.add_scalar('Loss/softq2',q_value_loss2,self.steps)	    

	    self.soft_q_optimizer1.zero_grad()
	    self.q_value_loss1.backward()
	    self.soft_q_optimizer1.step()
	    self.soft_q_optimizer2.zero_grad()
	    self.q_value_loss2.backward()
	    self.soft_q_optimizer2.step()    


	# Training Value Function
	    predicted_new_q_value = torch.min(self.soft_q_net1(im, curr_pos, goal, new_action),self.soft_q_net2(im, curr_pos, goal, new_action))
	    target_value_func = predicted_new_q_value - log_prob
	    value_loss = self.value_criterion(predicted_value, target_value_func.detach())
	    self.logging.add_scalar('Loss/value',value_loss,self.steps)
	    
	    self.value_optimizer.zero_grad()
	    self.value_loss.backward()
	    self.value_optimizer.step()

	# Training Policy Function
	    policy_loss = (log_prob - predicted_new_q_value).mean()
	    self.logging.add_scalar('Loss/policy_loss',policy_loss,self.steps)
	    self.policy_optimizer.zero_grad()
	    self.policy_loss.backward()
	    self.policy_optimizer.step()
	    
	    
	    for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
	        target_param.data.copy_(
	            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
	        )

	    self.steps += 1

	def train(self):

		episodes = 0
		rewards = np.zeros(arg_params['max_episodes'])

		while episodes < arg_params['max_episodes']:

		    complete_state = self.env.reset()
		    im = ToTensor()(complete_state[0])
		    curr_state = torch.from_numpy(complete_state[1])
		    goal = torch.from_numpy(complete_state[2])
		    episode_reward = 0

		    iters = 0

		    #im, curr_pos, goal, action, reward, next_im, next_state, done
		    
		    for step in range(arg_params['max_episode_len']):

		        iters += 1

		        action = self.policy_net.get_action(im,curr_state,goal_state).detach()
		        next_complete_state, reward, done = self.env.step(action.numpy())

		        next_obs = ToTensor()(next_complete_state[0])
		        next_state = torch.from_numpy(next_complete_state[1])
		                
		        self.replay_buffer.push(im, curr_state, goal, action, reward, next_obs, next_state, done)
		        
		        im = next_obs
		        curr_state = next_state
		        episode_reward += reward

		        if len(self.replay_buffer) > self.batch_size:
		            self.update()
		        
		        if done:
		            break

		    rewards[episodes] = episode_reward
		   	self.logging.add_scalar('Reward/episode_reward',episode_reward,episodes)
		    episodes += 1
		    print("episode :",episodes," episode reward:", episode_reward," total_steps:", iters)
		    

		    #save models

		    torch.save(self.value_net, arg_params['value_network_path'])
		    torch.save(self.policy_net, arg_params['policy_network_path'])
		    torch.save(self.soft_q_net1, arg_params['softq1_path'])
		    torch.save(self.soft_q_net2, arg_params['softq2_path'])
		    np.save(arg_params['reward_array_path'],rewards)


if __name__ == '__main__':

	SAC = Learner()
	SAC.train()
