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

class ProprioAdaptTConv(nn.Module):
    def __init__(self,
        act_obs_sum_dim: int,
        out_dim: int,
    ):
        super(ProprioAdaptTConv, self).__init__()

        self.channel_transform = nn.Sequential(
            nn.Linear(act_obs_sum_dim, 32),
            nn.ReLU6(),
        )

        self.temporal_aggregation = nn.Sequential(
            nn.Conv1d(32, 128, (3,), stride=(1,)),
            nn.ReLU6(),
            nn.Conv1d(128, 128, (2,), stride=(1,)),
            nn.ReLU6(),
            nn.Conv1d(128, 128, (2,), stride=(1,)),
            nn.ReLU6(),
        )

        self.low_dim_proj = nn.Sequential(
            nn.Linear(128,out_dim),
            nn.ReLU6(),
        )


    def forward(self, x):
        x = self.channel_transform(x)

        if len(x.shape) == 3:
            x = x.permute((0, 2, 1))
        else:
            x = x.permute((1, 0))
        
        y = self.temporal_aggregation(x)
        if len(x.shape) == 3:
            y = torch.flatten(y, 1)
        else:
            y = torch.flatten(y, 0)
        y = self.low_dim_proj(y)

        return y

class Env_Encoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Env_Encoder, self).__init__()
        self.env_encoder = nn.Sequential(
            nn.Linear(in_dim, 2048),
            nn.ReLU6(),
            nn.Linear(2048, 2048),
            nn.ReLU6(),
            nn.Linear(2048, out_dim),
            nn.ReLU6(),
        )
    
    def forward(self, x):
        return self.env_encoder(x)

class Policy(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Policy, self).__init__()
        self.policy = nn.Sequential(
            nn.Linear(in_dim, 2048),
            nn.ReLU6(),
            nn.Linear(2048, 2048),
            nn.ReLU6(),
            nn.Linear(2048, 2048),
            nn.ReLU6(),
            nn.Linear(2048, out_dim),
        )
    
    def forward(self, x):
        return self.policy(x)

class Agent(nn.Module):
    def __init__(self, policy_in_dim, policy_out_dim, device):
        super().__init__()

        self.actor_env_encoder = ProprioAdaptTConv(act_obs_sum_dim=33, out_dim=2)
        self.critic_env_encoder = ProprioAdaptTConv(act_obs_sum_dim=33, out_dim=2)

        self.actor_policy = Policy(policy_in_dim, policy_out_dim)
        self.critic_policy = Policy(policy_in_dim, 1)

        self.policy_out_dim = policy_out_dim

        self.device = device

    def get_value(self, obs, 
                  env_factor, 
                #   img_1
                  ):
        env_extrinsics = self.critic_env_encoder(env_factor)
        if len(env_extrinsics.shape) == 1:
            pi_input = torch.cat((obs, env_extrinsics))
        else:
            pi_input = torch.cat((obs, env_extrinsics), dim=1)

        # pi_input = obs
        return self.critic_policy(pi_input)
    
    def get_action(self, obs, 
                   env_factor, 
                #    img_1
                   ):
        
        # env_factor[2] = 0
        # env_factor[3] = 0

        env_extrinsics = self.actor_env_encoder(env_factor)
        # print(env_extrinsics)
        # env_extrinsics = torch.rand(env_extrinsics.shape).to("cuda:0")
        if len(obs.shape) == 1:
            pi_input = torch.cat((obs, env_extrinsics))
        else:
            pi_input = torch.cat((obs, env_extrinsics), dim=1)

        # pi_input = obs
        return self.actor_policy(pi_input)

    def get_action_and_value(self, obs, 
                             env_factor, 
                            #  img_1, 
                             action=None, 
                             sigma=0.5):
        
        action_mean = self.get_action(obs, 
                                      env_factor, 
                                    #   img_1
                                      )
        value = self.get_value(obs, 
                               env_factor, 
                            #    img_1
                               )
        
        sigma = torch.full(size=action_mean.shape, fill_value=sigma, device=self.device)
        # sigma = torch.exp(sigma) 
        probs = Normal(*(action_mean,sigma))
        if action is None:
            action = probs.sample()

        if len(obs.shape) == 2:
            return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value
        else:
            return action, probs.log_prob(action).sum(0), probs.entropy().sum(0), value

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
        "horizon":400,
        "has_renderer":True,
        "use_camera_obs":True,
        "has_offscreen_renderer":True,
        # "render_camera":"frontview",
        # "camera_names":["frontview",],
        "render_camera":"birdview",
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
    env.num_past = 5
    env.action_dm = 6
    env.state_dm = 27
    env.p3_training = True

    state_dim = 27
    env_factor_dim = 9
    extrinsics_dim = 2
    action_dim = 6
    num_past = 5
    policy_in_dim = state_dim + extrinsics_dim

    agent = Agent(policy_in_dim, action_dim, device).to(device)
    agent.load_state_dict(torch.load("0.8642_0.3426_20480.pt", map_location=device))
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
    env.writer = imageio.get_writer("test_gif/test_"+str(epi_num)+".gif", duration=100)

    recv_force_xyz = []
    obj_mass = []
    obj_vel = []
    total_reward = []
    eff_force_reward = []

    while epi_num <= 10:

    
        sigma = 0.01

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
            env.writer = imageio.get_writer("test_gif/test_"+str(epi_num)+".gif", duration=100)
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
            


            

