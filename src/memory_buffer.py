import torch


class ReplayBuffer:
    def __init__(self, arg_params):
      
        self.capacity = arg_params[replay_size]
        self.batch_size = arg_params[batch_size]

        self.image_buffer = torch.zeros((capacity,arg_params[n_channels],arg_params[img_height],arg_params[img_width]))
        self.curr_pos_buffer = torch.zeros((capacity,arg_params[state_dims]))
        self.goal_buffer = torch.zeros((capacity,arg_params[state_dims]))
        self.action_buffer = torch.zeros((capacity,arg_params[action_dims]))
        self.reward_buffer = torch.zeros((capacity,1))
        self.next_state_buffer = torch.zeros((capacity,arg_params[state_dims]))
        self.done_buffer = torch.zeros((capacity,1))

        self.position = 0
        self.state = 0
    
    def push(self, im, curr_pos, goal, action, reward, next_state, done):
        
        self.image_buffer[self.position] = im
        self.curr_pos_buffer[self.position] = curr_pos
        self.goal_buffer[self.position] = goal
        self.action_buffer[self.position] = action
        self.reward_buffer[self.position] = reward
        self.next_state_buffer[self.position] = next_state
        self.done_buffer = done

        if(self.state == 0 && self.position == self.capacity - 1):
          state = 1

        self.position = (self.position + 1) % self.capacity
    
    def sample(self):
        sample_len = self.batch_size

        if (self.state == 0 && self.position + 1 < self.batch_size):
          sample_len = self.positon + 1
          sample_indices = torch.arange(sample_len)

        else if (self.state == 0 && self.position + 1 > self.batch_size):
          sample_len = self.batch_size
          sample_indices = torch.randint(0,self.position,sample_len)

        else:
          sample_indices = torch.randint(0,self.capacity-1,sample_len)

        return self.image_buffer[sample_indices,:,:,:], self.curr_pos_buffer[sample_indices,:], self.goal_buffer[sample_indices,:], self.action_buffer[sample_indices,:] , self.reward_buffer[sample_indices,:], self.next_state_buffer[sample_indices,:], self.done_buffer[sample_indices,:]
    
    def __len__(self):
        if (self.state == 0):
          return self.position + 1
        else:
          return self.capacity