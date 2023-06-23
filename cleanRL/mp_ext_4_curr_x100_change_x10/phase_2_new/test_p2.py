import argparse
import os
import random
import time
from distutils.util import strtobool

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.distributions import Independent, Normal

import robosuite as suite
from robosuite.wrappers.gym_wrapper import GymWrapper
from robosuite import load_controller_config
from robosuite.controllers.joint_pos import JointPositionController
from robosuite.utils.binding_utils import MjData
import imageio
import pickle
import sys
sys.path.append("..")
os.path.abspath('../policy.py')
from policy import Agent, Env_Encoder, ProprioAdaptTConv



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=44,
        help="seed of the experiment")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    args = parser.parse_args()
    
    return args


def save_plot_data(env, recv_force_xyz, obj_mass, obj_vel, total_reward, eff_force_reward, epi_num):
    os.mkdir("plot_"+str(epi_num))
    for key in env.behavior_plot.keys():
        np.save("plot_"+str(epi_num)+"/"+key+".npy", np.array(env.behavior_plot[key]))
    applied_force = np.array(env.robots[0].controller.applied_force)
    np.save("plot_"+str(epi_num)+"/applied_force.npy", applied_force)
    np.save("plot_"+str(epi_num)+"/recv_force_xyz.npy", recv_force_xyz)
    np.save("plot_"+str(epi_num)+"/obj_mass.npy", obj_mass)
    np.save("plot_"+str(epi_num)+"/obj_vel.npy", obj_vel)
    np.save("plot_"+str(epi_num)+"/total_reward.npy", total_reward)
    np.save("plot_"+str(epi_num)+"/eff_force_reward.npy", eff_force_reward)
    

if __name__ == "__main__":
    args = parse_args()

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    controller_config = load_controller_config(default_controller="OSC_POSE")
    controller_config["impedance_mode"] = "variable_kp"

    config = {
        "env_name": "Lift",
        "robots": "Panda",
        "controller_configs": controller_config,
        # "reward_scale":None,
        "reward_shaping":True,
        "horizon":4000,
        "has_renderer":True,
        "use_camera_obs":True,
        "has_offscreen_renderer":True,
        # "render_camera":"frontview",
        # "camera_names":["frontview",],
        "render_camera":"birdview",
        # "render_camera":"sideview",
        "camera_names":["birdview",],
        "control_freq": 10
    }
    
    # print(env.observation_spec().keys())
    keys = ['robot0_joint_pos_cos', 
            'robot0_joint_pos_sin', 
            'robot0_joint_vel', 
            'robot0_eef_pos', 
            'robot0_eef_quat', 
            # 'robot0_gripper_qpos', 
            # 'robot0_gripper_qvel', 
            'cube_pos', 
            # 'cube_quat'
            ]

    env = suite.make(**config)
    env = GymWrapper(env, keys=keys)
    env.test_mode = True
    env.step_change = 21
    

    state_dim = 26
    env_factor_dim = 15
    extrinsics_dim = 6
    action_dim = 6
    policy_in_dim = state_dim + extrinsics_dim
    state_action_dim = state_dim + action_dim
    env.num_past = 5
    env.action_dm = action_dim
    env.state_dm = state_dim
    env.p3_training = True
    agent = Agent(policy_in_dim=policy_in_dim, 
                  policy_out_dim=action_dim, 
                  encoder_in=state_action_dim, 
                  encoder_out=extrinsics_dim,
                  phase=3,
                  device=device).to(device)
    agent.load_state_dict(torch.load("agent_0.9787234042553191.pt", map_location=device))
    # adaptTConv_actor = ProprioAdaptTConv(state_action_dim,extrinsics_dim).to(device)
    # adaptTConv_actor.load_state_dict(torch.load("actor_0.8901098901098901.pth", map_location=device))
    # adaptTConv_critic = ProprioAdaptTConv(state_action_dim,extrinsics_dim).to(device)
    # adaptTConv_critic.load_state_dict(torch.load("critic_0.8901098901098901.pth", map_location=device))
    # agent.load_state_dict(torch.load("0.9326_0.2507_1638400.pt", map_location=device))
    # agent.actor_env_encoder = adaptTConv_actor
    # agent.critic_env_encoder = adaptTConv_critic
    agent.eval()
    
    # ALGO Logic: Storage setup
    epi_num = 1
    next_obs, next_env_factor = env.reset()

    next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float32)
    next_env_factor = torch.as_tensor(next_env_factor, device=device, dtype=torch.float32)
    
    epi_reward = 0
    epi_steps = 0

    writer = SummaryWriter(f"test/{epi_num}")
    os.makedirs("test_gif",exist_ok=True)
    env.writer = imageio.get_writer("test_gif/test_"+str(epi_num)+".gif", duration=0.1)

    recv_force_xyz = []
    obj_mass = []
    obj_vel = []
    total_reward = []
    eff_force_reward = []

    while epi_num <= 10:

    
        sigma = 0.01
        sigma = torch.as_tensor([sigma], device=device, dtype=torch.float32)
        # ALGO LOGIC: action logic
        # action = agent.get_action(obs=next_obs, 
        #                             env_factor=next_env_factor, 
        #                             )
        
        action, _,_,_ = agent.get_action_and_value(obs=next_obs,
                                                                                env_factor=next_env_factor, 
                                                                                sigma=sigma
                                                                                )
        

        action = action.detach()
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, done, complete, next_env_factor, infos = env.step(action.cpu().numpy())
        env.render()
        # time.sleep(10)
        # print(env.sim.model.body_mass[env.blue_body_id])
        # print(env.sim.model.body_mass[env.red_body_id])
        # print()

        # writer.add_scalar("test_15_3/force", np.linalg.norm(np.array(env.robots[0].recent_ee_forcetorques.current[:3])), epi_steps)
        # writer.add_scalar("test_15_3/total_mass", env.total_mass, epi_steps)
        # writer.add_scalar("test_15_3/avg_friction", np.mean(env.friction_list), epi_steps)

        next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float32)
        next_env_factor = torch.as_tensor(next_env_factor, device=device, dtype=torch.float32)

        epi_reward += reward["dense_position_reward"] / 10
        epi_steps += 1
        
        # print(env.robots[0].recent_ee_forcetorques.current[:3][2] * 10)
        # recv_force_xyz.append(np.array(env.robots[0].recent_ee_forcetorques.current[:3])[2])
        recv_force_xyz.append(np.array(env.robots[0].recent_ee_forcetorques.current[:3]))
        obj_mass.append(env.total_mass)
        obj_vel.append(env.sim.data.get_body_xvelp("composite_cube_box1_main"))
        total_reward.append(reward["total_reward"])
        eff_force_reward.append(reward["eff_force_reward"])
        # print("eef force Z:: ",np.array(env.robots[0].recent_ee_forcetorques.current[:3])[2])

        # done with one eposide
        if np.any(done):
            print()
            print("complete rate: ",complete)
            print("epi_reward: ",epi_reward)
            print("epi_steps: ",epi_steps)
            epi_steps = 0
            epi_reward = 0
            env.writer = imageio.get_writer("test_gif/test_"+str(epi_num)+".gif", duration=0.1)
            if complete["total_complete"] > 0.99:
                save_plot_data(env, recv_force_xyz, obj_mass, obj_vel, total_reward, eff_force_reward, epi_num)
            recv_force_xyz = []
            obj_mass = []
            obj_vel = []
            total_reward = []
            eff_force_reward = []
            env.robots[0].controller.applied_force = []
            

            next_obs, next_env_factor = env.reset()

            next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float32)
            next_env_factor = torch.as_tensor(next_env_factor, device=device, dtype=torch.float32)
            epi_num += 1
            writer = SummaryWriter(f"test/{epi_num}")
            


            

