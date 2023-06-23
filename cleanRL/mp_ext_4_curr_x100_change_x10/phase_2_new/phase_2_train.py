import argparse
import os
import random
import time
from distutils.util import strtobool
import copy
import shutil

# import gymnasium as gym
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
import sys
sys.path.append("..")
os.path.abspath('../policy.py')
from policy import Agent, Env_Encoder, ProprioAdaptTConv




class RCA_phase_2:

    def __init__(self,):
        self.writer = SummaryWriter(f"runs/RCA_phase_2")
        self.seed=11
        self.device = torch.device("cuda:0")

        self.state_dim = 26
        self.env_factor_dim = 15
        self.extrinsics_dim = 6
        self.action_dim = 6
        self.policy_in_dim =  self.state_dim + self.extrinsics_dim
        self.act_obs_sum_dim = self.state_dim + self.action_dim
        self.num_past = 5
        self.checkpoint = "1.0_0.0_675840.pt"

        self.agent = Agent(policy_in_dim=self.policy_in_dim, 
                        policy_out_dim=self.action_dim, 
                        encoder_in=self.env_factor_dim, 
                        encoder_out=self.extrinsics_dim,
                        phase=1,
                        device=self.device).to(self.device)

        self.agent.load_state_dict(torch.load(self.checkpoint))
        self.agent.eval()
        
        # total num of steps to train
        self.total_step = 1_100_000
        self.sigma = 0.01
        # num of steps in one epoch
        self.epoch_size = 10240
        self.batch_size = 2048
        self.lr = 0.001

        self.setup_seed()
        self.setup_adapt_tconv()
        self.setup_env()
        
        self.best_perform_adapt_path = os.path.join("best_perform_adapt")
        os.makedirs(self.best_perform_adapt_path, exist_ok=True)
        self.max_compete_rate = -10

        adapt_tconv_path = os.path.join("min_loss_adapt")
        os.makedirs(adapt_tconv_path, exist_ok=True)
        self.adapt_tconv_actor_path = os.path.join(adapt_tconv_path, "adapt_tconv_actor.pt")
        self.adapt_tconv_actor_txt_path = os.path.join(adapt_tconv_path, "adapt_tconv_actor.txt")
        self.adapt_tconv_critic_path = os.path.join(adapt_tconv_path, "adapt_tconv_critic.pt")
        self.adapt_tconv_critic_txt_path = os.path.join(adapt_tconv_path, "adapt_tconv_critic.txt")

    def setup_adapt_tconv(self,):
        self.lowest_val_loss_actor = 0.9
        self.lowest_val_loss_critic = 0.9
        self.adapt_tconv_actor = ProprioAdaptTConv(in_dim=self.act_obs_sum_dim, out_dim=self.extrinsics_dim, device=self.device)
        self.adapt_tconv_actor.to(self.device)
        self.adapt_tconv_critic = ProprioAdaptTConv(in_dim=self.act_obs_sum_dim, out_dim=self.extrinsics_dim, device=self.device)
        self.adapt_tconv_critic.to(self.device)
        self.optimizer_actor = torch.optim.Adam(self.adapt_tconv_actor.parameters(), lr=self.lr)
        self.criterion_actor = nn.MSELoss()
        self.optimizer_critic = torch.optim.Adam(self.adapt_tconv_critic.parameters(), lr=self.lr)
        self.criterion_critic = nn.MSELoss()
    
    def setup_seed(self,):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
    
    def setup_env(self,):
        controller_config = load_controller_config(default_controller="OSC_POSE")
        controller_config["impedance_mode"] = "variable_kp"
        config = {
            "env_name": "Lift",
            "robots": "Panda",
            "controller_configs": controller_config,
            # "has_renderer":True,
            # "has_offscreen_renderer":True,
            "use_camera_obs":False,
            "reward_shaping":True,
            "horizon":400,
            # "render_camera":"birdview",
            # "camera_names":["frontview",],
            "control_freq": 10,
        }
        keys = ['robot0_joint_pos_cos', 
                'robot0_joint_pos_sin', 
                'robot0_joint_vel', 
                'robot0_eef_pos', 
                'robot0_eef_quat', 
                'cube_pos', 
                # 'cube_quat'
                ]
        self.env = suite.make(**config)
        self.env = GymWrapper(self.env, keys=keys)
        self.env.step_change = 21


    def train_adapt_tconv(self, total_dataset):
        train_size = int(len(total_dataset) * 0.8)
        val_size = len(total_dataset) - train_size

        # Split the dataset into training and validation sets
        train_dataset, val_dataset = torch.utils.data.random_split(total_dataset, [train_size, val_size])
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        for epoch in range(100):
            train_loss_actor = 0
            train_loss_critic = 0
            for i, (inputs, targets) in enumerate(train_dataloader):
                self.optimizer_actor.zero_grad()
                self.optimizer_critic.zero_grad()
                
                inputs.to(self.device)
                targets.to(self.device)
                
                inputs_2 = inputs.clone().detach()
                targets_2 = targets.clone().detach()

                outputs_actor = self.adapt_tconv_actor(inputs)
                loss_actor = self.criterion_actor(outputs_actor, targets[:,:self.extrinsics_dim])
                loss_actor.backward()
                self.optimizer_actor.step()

                outputs_critic = self.adapt_tconv_critic(inputs_2)
                loss_critic = self.criterion_critic(outputs_critic, targets_2[:,self.extrinsics_dim:])
                loss_critic.backward()
                self.optimizer_critic.step()

                train_loss_actor+=loss_actor.item()
                train_loss_critic+=loss_critic.item()

            train_loss_actor /= len(train_dataloader)
            train_loss_critic /= len(train_dataloader)

            # print(train_loss_actor)
            # exit()

            # Validate the MLP after each epoch
            with torch.no_grad():
                val_loss_actor = 0
                val_loss_critic = 0
                for inputs, targets in val_dataloader:
                    inputs.to(self.device)
                    targets.to(self.device)

                    outputs_actor = self.adapt_tconv_actor(inputs)
                    outputs_critic = self.adapt_tconv_critic(inputs)
                    val_loss_actor += self.criterion_actor(outputs_actor, targets[:,:self.extrinsics_dim]).item()
                    val_loss_critic += self.criterion_critic(outputs_critic, targets[:,self.extrinsics_dim:]).item()

                val_loss_actor /= len(val_dataloader)
                val_loss_critic /= len(val_dataloader)

                # writer.add_scalar("Phase_2/env_encoder_validation_Loss", val_loss, epoch)
                # writer.add_scalar("Phase_2/env_encoder_train_Loss", train_loss, epoch)
                print(f"Epoch {epoch + 1}/100, Actor Train Loss: {train_loss_actor:.8f}, Validation Loss: {val_loss_actor:.8f}")
                print(f"Epoch {epoch + 1}/100, Critic Train Loss: {train_loss_critic:.8f}, Validation Loss: {val_loss_critic:.8f}")
                print()
                if self.lowest_val_loss_actor > val_loss_actor:
                    self.lowest_val_loss_actor = val_loss_actor
                    if os.path.isfile(self.adapt_tconv_actor_path) and os.path.isfile(self.adapt_tconv_actor_txt_path):
                        os.remove(self.adapt_tconv_actor_path)
                        os.remove(self.adapt_tconv_actor_txt_path)
                    torch.save(self.adapt_tconv_actor.state_dict(), self.adapt_tconv_actor_path)
                    f = open(self.adapt_tconv_actor_txt_path, "w")
                    f.write(str(self.lowest_val_loss_actor) + " at epoch: " + str(epoch + 1))
                    print("best actor val_loss: ", self.lowest_val_loss_actor)

                if self.lowest_val_loss_critic > val_loss_critic:
                    self.lowest_val_loss_critic = val_loss_critic
                    if os.path.isfile(self.adapt_tconv_critic_path) and os.path.isfile(self.adapt_tconv_critic_txt_path):
                        os.remove(self.adapt_tconv_critic_path)
                        os.remove(self.adapt_tconv_critic_txt_path)
                    torch.save(self.adapt_tconv_critic.state_dict(), self.adapt_tconv_critic_path)
                    f = open(self.adapt_tconv_critic_txt_path, "w")
                    f.write(str(self.lowest_val_loss_critic) + " at epoch: " + str(epoch + 1))
                    print("best critic val_loss: ", self.lowest_val_loss_critic)

    def collect_rollouts(self,):
        X = None
        Y_gt = None
        
        next_obs, next_env_factor = self.env.reset()
        next_obs = torch.as_tensor(next_obs, device=self.device, dtype=torch.float32)
        next_env_factor = torch.as_tensor(next_env_factor, device=self.device, dtype=torch.float32)
        done = False
        complete = 0
        epi_step = 0

        prev_actions = torch.as_tensor(np.zeros((1,self.num_past,self.action_dim)),device=self.device, dtype=torch.float32)
        prev_states = torch.as_tensor(np.zeros((1,self.num_past,self.state_dim)),device=self.device, dtype=torch.float32)
        
        with torch.no_grad():
            while not done:
            
                extrinsics_gt = self.agent.actor_env_encoder(next_env_factor)
                extrinsics_gt_critic = self.agent.critic_env_encoder(next_env_factor)

                obs_act = torch.cat((prev_actions, prev_states), dim=2)
                if Y_gt is None:
                    X = obs_act
                    Y_gt = torch.unsqueeze(torch.cat((extrinsics_gt,extrinsics_gt_critic),dim=0),0)
                else:
                    X = torch.cat((X, obs_act), dim=0)
                    Y_gt = torch.cat((Y_gt, torch.unsqueeze(torch.cat((extrinsics_gt,extrinsics_gt_critic),dim=0),0)), dim=0)

                student_extrinsics = self.adapt_tconv_actor(torch.squeeze(obs_act,0))
                action = self.agent.get_action_extrinsics(next_obs, 
                                                        # extrinsics_gt, 
                                                        student_extrinsics,
                                                        torch.as_tensor([self.sigma],device=self.device, dtype=torch.float32)
                                                            )
                
                # print(prev_actions[:,1:,].shape)
                # print(torch.unsqueeze(action, 0).shape)

                prev_actions = torch.cat((prev_actions[:,1:,], torch.unsqueeze(torch.unsqueeze(action, 0),0)), dim=1)
                prev_states = torch.cat((prev_states[:,1:,], torch.unsqueeze(torch.unsqueeze(next_obs, 0),0)), dim=1)

                next_obs, reward, done, complete, next_env_factor, infos = self.env.step(action.cpu().numpy())
                next_obs = torch.as_tensor(next_obs, device=self.device, dtype=torch.float32)
                next_env_factor = torch.as_tensor(next_env_factor, device=self.device, dtype=torch.float32)
                epi_step += 1

        epi_complete = complete["total_complete"]

        print("complete rate in current episode: ", str(epi_complete))
        print("current episode steps: ", epi_step)

        dataset = torch.utils.data.TensorDataset(X, Y_gt)

        return dataset, epi_complete, epi_step
        

    def collect_rollouts_train_adapt_tconv(self,):
        curr_step = 0
        temp_step = 0
        total_complete = []
        total_dataset = None

        while curr_step < self.total_step:
            dataset, epi_complete, epi_step = self.collect_rollouts()
            total_complete.append(epi_complete)
            curr_step += epi_step
            if total_dataset is None:
                total_dataset = torch.utils.data.ConcatDataset([dataset])
            else:
                total_dataset = torch.utils.data.ConcatDataset([total_dataset,dataset])

            # print("total number of episode: ",total_epi)
            # print("avg complete rate in all episode: " + str(total_complete/total_epi))
            print("steps so far: ",curr_step)
            print()
            print()
            if curr_step - temp_step >= self.epoch_size:
                    # or curr_step >= self.total_step:

                completion_rate = (np.array(total_complete) > 0.99).sum()/len(total_complete)
                print("lowest val loss actor: ", self.lowest_val_loss_actor)
                print("lowest val loss critic: ", self.lowest_val_loss_critic)
                print("avg complete rate in all episodes before new update: " 
                      + str(completion_rate))
                print()
                print()
                self.writer.add_scalar("phase_2/evaluation/actor_loss", 
                            self.lowest_val_loss_actor, curr_step)
                self.writer.add_scalar("phase_2/evaluation/critic_loss", 
                            self.lowest_val_loss_critic, curr_step)
                self.writer.add_scalar("phase_2/evaluation/completion_rate", 
                            completion_rate, curr_step)
                total_complete = []

                if completion_rate > self.max_compete_rate:
                    self.max_compete_rate = completion_rate
                    best_perform_folder = os.listdir(self.best_perform_adapt_path)
                    for item in best_perform_folder:
                        if item.endswith(".pt"):
                            os.remove(os.path.join(self.best_perform_adapt_path, item))
                    # torch.save(self.adapt_tconv_actor.state_dict(), str(self.best_perform_adapt_path)+"/actor_"+str(completion_rate)+".pt")
                    # torch.save(self.adapt_tconv_critic.state_dict(), str(self.best_perform_adapt_path)+"/critic_"+str(completion_rate)+".pt")
                    new_agent = Agent(policy_in_dim=self.policy_in_dim, 
                        policy_out_dim=self.action_dim, 
                        encoder_in=self.env_factor_dim, 
                        encoder_out=self.extrinsics_dim,
                        phase=1,
                        device=self.device).to(self.device)
                    new_agent.load_state_dict(torch.load(self.checkpoint))
                    new_agent.actor_env_encoder = self.adapt_tconv_actor
                    new_agent.critic_env_encoder = self.adapt_tconv_critic
                    torch.save(new_agent.state_dict(), str(self.best_perform_adapt_path)+"/agent_"+str(completion_rate)+".pt")

                self.train_adapt_tconv(total_dataset)
                print()
                print()
                temp_step = curr_step
                # os.makedirs(base_path, exist_ok=True)
                # ckpt_path = os.path.join(base_path, "phase_2_dataset_"+ str(total_step) +".pth")
                # torch.save(total_dataset, ckpt_path)
                total_dataset = None

if __name__ == "__main__":

    RCA_phase_2_trainer = RCA_phase_2()
    RCA_phase_2_trainer.collect_rollouts_train_adapt_tconv()
    