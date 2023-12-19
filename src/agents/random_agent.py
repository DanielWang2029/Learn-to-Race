"""This is OpenAI' Spinning Up PyTorch implementation of Soft-Actor-Critic with
minor adjustments.
For the official documentation, see below:
https://spinningup.openai.com/en/latest/algorithms/sac.html#documentation-pytorch-version
Source:
https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/sac.py
"""
import itertools
from copy import deepcopy

import torch
import numpy as np
from gym.spaces import Box
from torch.optim import Adam
import scipy

from src.agents.SACAgent import SACAgent
from src.config.yamlize import yamlize, create_configurable, NameToSourcePath
from src.utils.utils import ActionSample

from src.constants import DEVICE
import math
from src.agents.SafetyFilter import getSafetyFilter

@yamlize
class RandomAgent(SACAgent):
    """Adopted from https://github.com/learn-to-race/l2r/blob/main/l2r/baselines/rl/sac.py"""

    def __init__(
        self,
        steps_to_sample_randomly: int,
        gamma: float,
        alpha: float,
        polyak: float,
        lr: float,
        actor_critic_cfg_path: str,
        safety_data_path: str,
        load_checkpoint_from: str = "",
        segment_length: int = 24,
        safety_margin: float = 4.2,
        verbose_controller: bool = True,

    ):
        """Initialize Soft Actor-Critic Agent

        Args:
            steps_to_sample_randomly (int): Number of steps to sample randomly
            gamma (float): Gamma parameter
            alpha (float): Alpha parameter
            polyak (float): Polyak parameter coef.
            lr (float): Learning rate parameter.
            actor_critic_cfg_path (str): Actor Critic Config Path
            safety_data_path (str): Safety data folder
            load_checkpoint_from (str, optional): Load checkpoint from path. If '', then doesn't load anything. Defaults to ''.
        """


        # SACAgent init. Might not work.
        super(RandomAgent, self).__init__(steps_to_sample_randomly, gamma, alpha, polyak, lr, actor_critic_cfg_path, load_checkpoint_from)


        self.safety_data_path = safety_data_path
        self.safety_margin = safety_margin
        self.safety_controller = SafeController(verbose=verbose_controller) #TODO: Modularize
        self.segment_length = segment_length
        
        self.nearest_idx = None

        # For now, as much as it pains me, respect the original interface of SACAgent (don't pass in t)
        self.t = 0

        
    def _load_track_info(self, centerline_array):
        # Add some secondary init / perhaps pass this in at update?

        # load centerline and x, y pos.
        self.raceline = centerline_array                                    
        race_x = self.raceline[:, 0]                                             
        race_y = self.raceline[:, 1]                                             
                                                                         
        X_diff = np.concatenate([race_x[1:] - race_x[:-1],             
                                 [race_x[0] - race_x[-1]]])            
        Y_diff = np.concatenate([race_y[1:] - race_y[:-1],             
                                 [race_y[0] - race_y[-1]]])   

        # compute yaw through simple means (pure angle)         
        race_yaw = np.arctan(Y_diff / X_diff)  # (L-1, n)                   
        
        #normalize yaw to desired range (probably 0 - 2pi?)
        race_yaw[X_diff < 0] += np.pi   

        #TODO: Fix this. please. 
        def smooth_yaw(yaw):
            for i in range(len(yaw) - 1):
                dyaw = yaw[i + 1] - yaw[i]

                while dyaw >= math.pi / 2.0:
                    yaw[i + 1] -= math.pi * 2.0
                    dyaw = yaw[i + 1] - yaw[i]

                while dyaw <= -math.pi / 2.0:
                    yaw[i + 1] += math.pi * 2.0
                    dyaw = yaw[i + 1] - yaw[i]
            return yaw

        # 'Smooth' yaw values so that the difference between values is always between -pi/2, pi/2                
        self.race_yaw = smooth_yaw(race_yaw)
        
        # compute max and min yaw per track
        self.max_yaw = np.max(self.race_yaw)
        self.min_yaw = np.min(self.race_yaw)

    def _get_safety_value(self, feat, nearest_idx):

        # Get the closest index on the track.
        track_index = nearest_idx
        nearest_idx = abs(track_index//self.segment_length * self.segment_length) # This index is used to find the correponnding safety set.

        ## Only reload if move onto the next segment ( optimization )
        if nearest_idx == self.nearest_idx:
            pass 
        else: 
            # Get safety values from system
            self.safety = np.load(f"{self.safety_data_path}/{nearest_idx}.npz", allow_pickle=True)
            self.nearest_idx = nearest_idx
            self.grid = (self.safety['x'], self.safety['y'], self.safety['v'], self.safety['yaw'])

        # Unpack IMU values ( probably create a safety wrapper on env. )
        v = feat.flatten()[-4].item()
        x = feat.flatten()[-3].item()
        y = feat.flatten()[-2].item()
        yaw = feat.flatten()[-1].item()

        ## make sure yaw in consistent with the range of race_yaw
        if yaw >= self.max_yaw:
            yaw -= 2 * np.pi
        if yaw < self.min_yaw:
            yaw += 2 * np.pi

        # Use the racetrack geometry at the nearest_idx instead of the idx for coordinate transform
        origin, yaw0 = self.raceline[nearest_idx], self.race_yaw[nearest_idx]

        # Transform raceline ( global ) data to local coordinate system.
        def toLocal(x, y, v, yaw, origin, local_race_yaw):
            ## TODO: add v and yaw
            '''
            # v' = v
            # yaw' = yaw - yaw0
            '''
            yaw = yaw-local_race_yaw
            # (x, y) -> (x', y')
            XY = np.array([x,y])
            transform = np.array([[np.cos(local_race_yaw), np.sin(local_race_yaw)], 
                        [-np.sin(local_race_yaw), np.cos(local_race_yaw)]])
            coords = (XY-origin).dot(transform.T)
            return np.array(np.concatenate([coords, [v, yaw]]))
    
        local_state = toLocal(x, y, v, yaw, origin, yaw0) #np.array([5, 2, 10, 1.5])

        # make sure yaw is in the correct range
        if (local_state[3]>max(self.safety['yaw'])):
            local_state[3] -= 2*np.pi
        if (local_state[3]<min(self.safety['yaw'])):
            local_state[3] += 2*np.pi
        
        min_state = np.array([min(self.safety['x']), min(self.safety['y']), min(self.safety['v']), min(self.safety['yaw'])])
        max_state = np.array([max(self.safety['x']), max(self.safety['y']), max(self.safety['v']), max(self.safety['yaw'])])
        local_state = np.clip(local_state, a_min=min_state, a_max=max_state)
        
        try:    
            ## Calculate the safety value, by interpolating amongst known values
            V_est = scipy.interpolate.interpn(self.grid, self.safety['V'], local_state)
        except ValueError as err:
            print("Safety value not found", err, local_state)
            ## Assume unsafe
            V_est = -1
            #print(local_state[3], self.safety['yaw'])
        print(f"V_est={V_est}")
        return V_est, local_state

    def select_action(self, feat, nearest_idx = -1, deterministic=False):
        self.record['transition_actor'] = 'random'
        self.t += 1
        action = ActionSample()
        action.action = np.random.uniform([-1, -0.125], [1, 1])
        return action

# safety_data_path + trackname?
# safety_margin
# start_steps



class SafeController():
    def __init__(self, 
            uMin=(-1.0, -1.0), # [\delta, a]
            uMax=(1.0, 1.0),
            verbose = False):
        self.uMax = uMax 
        self.uMin = uMin
        self.verbose = verbose

    def select_action(self, local_state, safety):
        """
        Input: 
            local_state:
            safety_set: 
        Output: 
            action: [steering, acc]
        """
        ## esimate dV/dx with finite difference 

        def find_nearest(array,value):
            return (np.abs(array - value)).argmin()

        x, y, v, yaw = local_state.tolist()
        x_idx = find_nearest(safety['x'], x)
        y_idx = find_nearest(safety['y'], y)
        v_idx = find_nearest(safety['v'], v)
        yaw_idx = find_nearest(safety['yaw'], yaw)

        J_v = safety['J3'][x_idx, y_idx, v_idx, yaw_idx]
        J_yaw = safety['J4'][x_idx, y_idx, max(v_idx, 1), yaw_idx]


        # Update actions with parameters given pre-computed safety estimates.
        opt_w = self.uMin[0] if J_yaw < 0 else (0 if J_yaw == 0 else self.uMax[0])
        w_msg = 'RIGHT' if J_yaw < 0 else ('-' if J_yaw == 0 else 'LEFT')
        
        opt_a = self.uMin[1] if J_v < 0 else (0 if J_v == 0 else self.uMax[1])
        a_msg = 'BRAKE' if J_v < 0 else ('-' if J_v == 0 else 'ACCELERATE')

        '''
        ## Prevent the braking from being too conservative
        if ((v<3) & (V_current>3)) | ((v<1) & (V_current>1.8)):
            opt_a = 0
        '''
        # I feel like most of this logic can be condensed to a few selective numpy calls...

        ## Prevent the vehicle from coming to a complete stop
        if v<0.3:
            opt_a = 0.1
        ## Do not brake if already slow
        elif v<1:
            opt_a = 0
        
        ## Prevent the steering angle from being too large
        # v in m/s; 1/s->2.24mph
        if v > 30:
            opt_w = np.clip(opt_w, a_min=-1/12, a_max=1/12)
        elif v > 20:
            opt_w = np.clip(opt_w, a_min=-1/6, a_max=1/6)
        elif v > 10:
            opt_w = np.clip(opt_w, a_min=-1/3, a_max=1/3)
        
        if self.verbose:
            #print(f"Safe Controller: {'LEFT' if opt_w>0 else 'RIGHT'}")
            print(f"Safe Controller: {w_msg}, {a_msg}")
        return np.array([opt_w, opt_a])
