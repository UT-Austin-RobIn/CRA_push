# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import argparse
import os
import random
import time
from distutils.util import strtobool
import copy
import shutil
import threading
# import multiprocessing
import torch.multiprocessing as multiprocessing

# import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

# from torch.distributions import Independent, Normal

import robosuite as suite
from robosuite.wrappers.gym_wrapper import GymWrapper
from robosuite import load_controller_config
from robosuite.controllers.joint_pos import JointPositionController
from policy import Agent

def collect_data(start, num_data, data_dict, config, keys, device, agent, anneal_step_num, args, state_dim, env_factor_dim, action_dim):
    random.seed(start)
    np.random.seed(start)
    torch.manual_seed(start)
    env = suite.make(**config)
    env = GymWrapper(env, keys=keys)
    env.step_change = 21
    next_obs, next_env_factor = env.reset()
    next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float32)
    next_env_factor = torch.as_tensor(next_env_factor, device=device, dtype=torch.float32)
    # next_img_1 = torch.as_tensor(next_img_1, device=device, dtype=torch.float32)
    next_done = torch.zeros(1).to(device)

    num_episode = 0
    avg_complete_rate = []
    avg_ori_complete_rate = []
    avg_pos_complete_rate = []
    avg_episode_reward = []
    avg_episode_dense_position_reward = []
    avg_episode_dense_orientation_reward = []
    avg_episode_dense_com_reward = []
    avg_episode_contact_reward = []
    avg_episode_len = []
    avg_eff_force_reward = []
    avg_episode_force = []
    avg_eff_vel_reward = []
    avg_eff_episode_vel = []
    avg_curr_obj_vel_error_reward = []
    avg_obj_vel_change_reward = []
    avg_episode_obj_vel = []
    termination_stat = np.zeros(8)

    episode_reward = 0
    episode_dense_position_reward = 0
    episode_dense_orientation_reward = 0
    episode_dense_com_reward = 0
    episode_contact_reward = 0
    episode_len = 0
    episode_eff_force_reward = 0
    episode_force = []
    episode_eff_vel_reward = 0
    episode_eff_vel = []
    episode_curr_obj_vel_error_reward = 0
    episode_obj_vel_change_reward = 0
    episode_obj_vel = []

    in_obs = torch.zeros((num_data-start,) + (state_dim,)).to(device)
    in_env_factor = torch.zeros((num_data-start,) + (env_factor_dim,)).to(device)
    in_actions = torch.zeros((num_data-start,) + (action_dim,)).to(device)
    in_logprobs = torch.zeros((num_data-start,)).to(device)
    in_rewards = torch.zeros((num_data-start)).to(device)
    in_dones = torch.zeros((num_data-start,)).to(device)
    in_values = torch.zeros((num_data-start,)).to(device)
    in_sigmas = torch.zeros((num_data-start,)).to(device)

    for step in range(0, num_data-start):
        global_step = start + step

        in_obs[step] = next_obs
        in_env_factor[step] = next_env_factor
        # img_1s[step] = next_img_1
        in_dones[step] = next_done

        sigma = 1
        if anneal_step_num - global_step <= 100_000:
            sigma = sigma * (100_000/anneal_step_num)
        else:
            sigma = sigma * ((anneal_step_num - global_step)/anneal_step_num)
        sigma = torch.as_tensor([sigma], device=device, dtype=torch.float32)

        # ALGO LOGIC: action logic
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(obs=next_obs, 
                                                                    env_factor=next_env_factor, 
                                                                #    img_1=next_img_1,
                                                                    sigma=sigma)
        in_sigmas[step] = sigma
        in_values[step] = value
        in_actions[step] = action
        in_logprobs[step] = logprob

        
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, done, complete, next_env_factor, infos = env.step(action.cpu().numpy())
        # env.render()
        reward["total_reward"] = reward["total_reward"]
        reward["dense_position_reward"] = reward["dense_position_reward"]
        reward["dense_orientation_reward"] = reward["dense_orientation_reward"]
        reward["dense_com_reward"] = reward["dense_com_reward"]
        reward["contact_reward"] = reward["contact_reward"]
        reward["eff_force_reward"] = reward["eff_force_reward"]

        episode_reward += reward["total_reward"]
        episode_dense_position_reward += reward["dense_position_reward"]
        episode_dense_orientation_reward += reward["dense_orientation_reward"]
        episode_dense_com_reward += reward["dense_com_reward"]
        episode_contact_reward += reward["contact_reward"]
        episode_eff_force_reward += reward["eff_force_reward"]
        episode_eff_vel_reward += reward["eff_vel_reward"]
        episode_force.append(np.array(env.robots[0].recent_ee_forcetorques.current[:3])[2])
        episode_eff_vel.append(np.linalg.norm(env.robots[0].recent_ee_vel.current[:3]))
        episode_curr_obj_vel_error_reward += reward["curr_obj_vel_error_reward"]
        episode_obj_vel_change_reward += reward["obj_vel_change_reward"]
        episode_obj_vel.append(np.linalg.norm(env.sim.data.get_body_xvelp("composite_cube_root")))

        episode_len += 1
        next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float32)
        in_rewards[step] = torch.as_tensor(reward["total_reward"], device=device, dtype=torch.float32)
        # dones[step] = torch.as_tensor(done, device=device, dtype=torch.float32)
        next_done = torch.as_tensor(done, device=device, dtype=torch.float32)
        next_env_factor = torch.as_tensor(next_env_factor, device=device, dtype=torch.float32)
        
        # done with one eposide
        if np.any(done) and step != args.num_steps - 1:
            rate = complete["total_complete"]
            print(f"global_step={global_step}, episodic_return={episode_reward}, complete_rate={rate}")
            
            num_episode += 1
            avg_complete_rate.append(complete["total_complete"])
            avg_ori_complete_rate.append(complete["orientation_complete"])
            avg_pos_complete_rate.append(complete["position_complete"])
            avg_episode_reward.append(episode_reward)
            avg_episode_dense_position_reward.append(episode_dense_position_reward)
            avg_episode_dense_orientation_reward.append(episode_dense_orientation_reward)
            avg_episode_dense_com_reward.append(episode_dense_com_reward)
            avg_episode_contact_reward.append(episode_contact_reward)
            avg_episode_len.append(episode_len)
            avg_eff_force_reward.append(episode_eff_force_reward)
            avg_episode_force.append(np.mean(episode_force))
            avg_eff_vel_reward.append(episode_eff_vel_reward)
            avg_eff_episode_vel.append(np.mean(episode_eff_vel))
            avg_curr_obj_vel_error_reward.append(episode_curr_obj_vel_error_reward)
            avg_obj_vel_change_reward.append(episode_obj_vel_change_reward)
            avg_episode_obj_vel.append(np.mean(episode_obj_vel))
            termination_stat[infos["termination_stat"]] += 1
        
            episode_reward = 0
            episode_dense_position_reward = 0
            episode_dense_orientation_reward = 0
            episode_dense_com_reward = 0
            episode_contact_reward = 0
            episode_len = 0
            episode_eff_force_reward = 0
            episode_force = []
            episode_eff_vel_reward = 0
            episode_eff_vel = []
            episode_curr_obj_vel_error_reward = 0
            episode_obj_vel_change_reward = 0
            episode_obj_vel = []
            next_obs, next_env_factor = env.reset()
            env.global_step = global_step
            
            next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float32)
            next_env_factor = torch.as_tensor(next_env_factor, device=device, dtype=torch.float32)


    data_dict["avg_complete_rate"] = np.array(avg_complete_rate)
    data_dict["avg_ori_complete_rate"] = np.array(avg_ori_complete_rate)
    data_dict["avg_pos_complete_rate"] = np.array(avg_pos_complete_rate)
    data_dict["avg_episode_reward"] = np.array(avg_episode_reward)
    data_dict["avg_episode_dense_position_reward"] = np.array(avg_episode_dense_position_reward)
    data_dict["avg_episode_dense_orientation_reward"] = np.array(avg_episode_dense_orientation_reward)
    data_dict["avg_episode_dense_com_reward"] = np.array(avg_episode_dense_com_reward)
    data_dict["avg_episode_contact_reward"] = np.array(avg_episode_contact_reward)
    data_dict["avg_episode_len"] = np.array(avg_episode_len)
    data_dict["avg_eff_force_reward"] = np.array(avg_eff_force_reward)
    data_dict["avg_episode_force"] = np.array(avg_episode_force)
    data_dict["avg_eff_vel_reward"] = np.array(avg_eff_vel_reward)
    data_dict["avg_eff_episode_vel"] = np.array(avg_eff_episode_vel)
    data_dict["avg_curr_obj_vel_error_reward"] = np.array(avg_curr_obj_vel_error_reward)
    data_dict["avg_obj_vel_change_reward"] = np.array(avg_obj_vel_change_reward)
    data_dict["avg_episode_obj_vel"] = np.array(avg_episode_obj_vel)
    data_dict["termination_stat"] = np.array(termination_stat)
    
    data_dict["obs"] = in_obs.to("cpu")
    data_dict["env_factor"] = in_env_factor.to("cpu")
    data_dict["actions"] = in_actions.to("cpu")
    data_dict["logprobs"] = in_logprobs.to("cpu")
    data_dict["rewards"] = in_rewards.to("cpu")
    data_dict["dones"] = in_dones.to("cpu")
    data_dict["values"] = in_values.to("cpu")
    data_dict["sigmas"] = in_sigmas.to("cpu")
    