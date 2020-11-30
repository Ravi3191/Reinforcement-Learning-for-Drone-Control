import torch
import numpy as np

class ReplayBuffer_dir:
    def __init__(self, arg_params):

        self.observation_path = arg_params['observation_path']
        self.next_observation_path = arg_params['next_observation_path']

        self.capacity = arg_params['replay_size']
        self.batch_size = arg_params['batch_size']

        self.curr_pos_buffer = torch.zeros((capacity,arg_params['state_dims']))
        self.goal_buffer = torch.zeros((capacity,arg_params['state_dims']))
        self.action_buffer = torch.zeros((capacity,arg_params['action_dims']))
        self.reward_buffer = torch.zeros((capacity,1))
        self.next_state_buffer = torch.zeros((capacity,arg_params['state_dims']))
        self.done_buffer = torch.zeros((capacity,1))

        self.position = 0
        self.state = 0
    
    def push(self, im, curr_pos, goal, action, reward, next_im, next_state, done):
        
        curr_obs_path = self.obesrvation_path + str(self.position) + '.pt' 
        next_obs_path = self.next_observation_path + str(self.position) + '.pt'
        torch.save(im,curr_obs_path)
        torch.save(next_im,curr_obs_path)

        self.curr_pos_buffer[self.position] = curr_pos
        self.goal_buffer[self.position] = goal
        self.action_buffer[self.position] = action
        self.reward_buffer[self.position] = reward
        self.next_state_buffer[self.position] = next_state
        self.done_buffer = done

        if(self.state == 0 and self.position == self.capacity - 1):
          state = 1

        self.position = (self.position + 1) % self.capacity
    
    def sample(self):
        sample_len = self.batch_size

        if (self.state == 0 and self.position + 1 < self.batch_size):
          sample_len = self.positon + 1
          sample_indices = torch.arange(sample_len)

        elif (self.state == 0 and self.position + 1 > self.batch_size):
          sample_indices = torch.randint(0,self.position,sample_len)

        else:
          sample_indices = torch.randint(0,self.capacity-1,sample_len)

        obs_batch = torch.zeros((sample_len,arg_params['n_channels'],arg_params['img_height'],arg_params['img_width']))
        next_obs_batch = torch.zeros((sample_len,arg_params['n_channels'],arg_params['img_height'],arg_params['img_width']))

        for i in range(sample_len):
          obs_batch[i,:,:,:] = torch.laod(self.observation_path + str(sample_indices[i]) + '.pt')
          next_obs_batch[i,:,:,:] = torch.laod(self.next_obesrvation_path + str(sample_indices[i]) + '.pt')
        
        return obs_batch, self.curr_pos_buffer[sample_indices,:], self.goal_buffer[sample_indices,:],\
         self.action_buffer[sample_indices,:] , self.reward_buffer[sample_indices,:], \
         next_obs_batch, self.next_state_buffer[sample_indices,:], self.done_buffer[sample_indices,:]
    
    def __len__(self):
        if (self.state == 0):
          return self.position + 1
        else:
          return self.capacity