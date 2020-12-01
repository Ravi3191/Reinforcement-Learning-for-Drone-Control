import torch
import numpy as np
import os

class ReplayBuffer_dir:
    def __init__(self, arg_params):

        self.observation_path = arg_params['root_dir'] + arg_params['observation_path']
        self.next_observation_path = arg_params['root_dir'] + arg_params['next_observation_path']

        os.makedirs(self.observation_path,exist_ok = True)
        os.makedirs(self.next_observation_path,exist_ok = True)

        self.capacity = arg_params['replay_size']
        self.batch_size = arg_params['batch_size']

        self.curr_pos_buffer = torch.zeros((self.capacity,arg_params['state_dims']))
        self.goal_buffer = torch.zeros((self.capacity,arg_params['state_dims']))
        self.action_buffer = torch.zeros((self.capacity,arg_params['action_dims']))
        self.reward_buffer = torch.zeros((self.capacity,1))
        self.next_state_buffer = torch.zeros((self.capacity,arg_params['state_dims']))
        self.done_buffer = torch.zeros((self.capacity,1))

        self.position = 0
        self.state = 0

        self.arg_params = arg_params
    
    def push(self, im, curr_pos, goal, action, reward, next_im, next_state, done):
        
        curr_obs_path = self.observation_path + str(self.position) + '.pt' 
        next_obs_path = self.next_observation_path + str(self.position) + '.pt'
        torch.save(im[0,:,:,:],curr_obs_path)
        torch.save(next_im[0,:,:,:],next_obs_path)

        self.curr_pos_buffer[self.position] = curr_pos[0,:]
        self.goal_buffer[self.position] = goal[0,:]
        self.action_buffer[self.position] = action
        self.reward_buffer[self.position] = reward
        self.next_state_buffer[self.position] = next_state[0,:]
        self.done_buffer[self.position] = done

        if(self.state == 0 and self.position == self.capacity - 1):
          state = 1

        self.position = (self.position + 1) % self.capacity
    
    def sample(self):
        sample_len = self.batch_size

        if (self.state == 0 and self.position + 1 < self.batch_size):
          sample_len = self.positon + 1
          sample_indices = torch.arange(sample_len)

        elif (self.state == 0 and self.position + 1 > self.batch_size):
          sample_indices = torch.randint(0,self.position,(sample_len,))

        else:
          sample_indices = torch.randint(0,self.capacity-1,(sample_len,))

        obs_batch = torch.zeros((sample_len,self.arg_params['n_channels'],self.arg_params['img_height'],self.arg_params['img_width']))
        next_obs_batch = torch.zeros((sample_len,self.arg_params['n_channels'],self.arg_params['img_height'],self.arg_params['img_width']))

        for i in range(sample_len):
          obs_batch[i,:,:,:] = torch.load(self.observation_path + str(int(sample_indices[i])) + '.pt')
          next_obs_batch[i,:,:,:] = torch.load(self.next_observation_path + str(int(sample_indices[i])) + '.pt')

        return obs_batch, self.curr_pos_buffer[sample_indices,:], self.goal_buffer[sample_indices,:],\
         self.action_buffer[sample_indices,:] , self.reward_buffer[sample_indices,:], \
         next_obs_batch, self.next_state_buffer[sample_indices,:], self.done_buffer[sample_indices]
    
    def __len__(self):
        if (self.state == 0):
          return self.position + 1
        else:
          return self.capacity