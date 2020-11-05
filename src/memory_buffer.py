import torch


class ReplayBuffer:

  def __init__(self, capacity):

  	"""
		stores the data from previous rollouts
		capacity: size of the buffer
  	"""

    self.pos = 
    self.images = 
    self.capacity = capacity
    self.cur_size = 
    self.state = 'new'

  def push(self, replay):
  	"""
  		adds images and position to the buffer
  	"""

  def sample(self, idx):
  	"""
  		return the images and the pos at a particular idx
  	"""

  	if self.state == 'full':
    	return images[idx],pos[idx]
    else:
    	index = int(torch.rand(1)*self.cur_size)
    	return images[index],pos[index]

  def __len__(self):

    return self.actions.shape[0]
  
  def print_here(self):
    print(self.state.shape)

    pass