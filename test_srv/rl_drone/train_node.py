"""
Integrates everything with ros and adds data to the buffer
when  it receives anything from the environment
"""

import rospy
from config import *

import train.py as train
