import torch


def reward_from_traj(pos,policy,map):

	"""
		Calculates the trajectory based reward by 
		rolling out the policy for a few time frames and 
		aggregating the reward accumilated from the uncertainity

		pos: Current position of the robot
		policy: The current DRL model which gives velocity and heading
		map: SLAM map that has uncertainity 
	"""



	return rewards