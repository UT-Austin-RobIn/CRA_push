import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import torch.nn.functional as F


class Env_Encoder(nn.Module):
    def __init__(self, in_dim, out_dim, device):
        super(Env_Encoder, self).__init__()
        self.device = device
        self.env_encoder = nn.Sequential(
            nn.Linear(in_dim, 2048),
            nn.ReLU6(),
            nn.Linear(2048, 2048),
            nn.ReLU6(),
            nn.Linear(2048, out_dim),
            nn.ReLU6(),
        )
    
    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        return self.env_encoder(x)
    
class ProprioAdaptTConv(nn.Module):
    def __init__(self,
        in_dim: int,
        out_dim: int,
        device,
    ):
        super(ProprioAdaptTConv, self).__init__()
        self.device = device

        self.channel_transform = nn.Sequential(
            nn.Linear(in_dim, 32),
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
        torch.as_tensor(x, dtype=torch.float32, device=self.device)
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

class Policy(nn.Module):
    def __init__(self, in_dim, out_dim, device):
        super(Policy, self).__init__()
        self.device = device
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
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        return self.policy(x)

class Agent(nn.Module):
    def __init__(self, policy_in_dim, policy_out_dim, encoder_in, encoder_out, phase, device):
        super().__init__()
        
        if phase == 1:
            self.actor_env_encoder = Env_Encoder(in_dim=encoder_in, out_dim=encoder_out, device=device)
            self.critic_env_encoder = Env_Encoder(in_dim=encoder_in, out_dim=encoder_out, device=device)
        elif phase == 3:
            self.actor_env_encoder = ProprioAdaptTConv(in_dim=encoder_in, out_dim=encoder_out, device=device)
            self.critic_env_encoder = ProprioAdaptTConv(in_dim=encoder_in, out_dim=encoder_out, device=device)

        self.actor_policy = Policy(policy_in_dim, policy_out_dim, device=device)
        self.critic_policy = Policy(policy_in_dim, 1, device=device)
        
        self.policy_out_dim = policy_out_dim
        self.device = device

    def get_value(self, obs, 
                  env_factor,
                  ):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        env_factor = torch.as_tensor(env_factor, dtype=torch.float32, device=self.device)

        env_extrinsics = self.critic_env_encoder(env_factor)
        if len(env_extrinsics.shape) == 1:
            pi_input = torch.cat((obs, env_extrinsics))
        else:
            pi_input = torch.cat((obs, env_extrinsics), dim=1)

        return self.critic_policy(pi_input)
    
    def get_action(self, obs, 
                   env_factor,
                   ):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        env_factor = torch.as_tensor(env_factor, dtype=torch.float32, device=self.device)


        # env_factor = torch.ones(env_factor.shape).to("cuda:0")
        env_extrinsics = self.actor_env_encoder(env_factor)
        # print(env_extrinsics)
        # env_extrinsics = torch.rand(env_extrinsics.shape).to("cuda:0")
        if len(obs.shape) == 1:
            pi_input = torch.cat((obs, env_extrinsics))
        else:
            pi_input = torch.cat((obs, env_extrinsics), dim=1)

        return self.actor_policy(pi_input)

    def get_action_and_value(self, obs, 
                             env_factor,  
                             action=None, 
                             sigma=0.5):
        
        action_mean = self.get_action(obs, 
                                      env_factor, 
                                      )
        value = self.get_value(obs, 
                               env_factor,
                               )
        
        sigma = sigma.reshape(sigma.shape[0],1)
        sigma = torch.broadcast_to(sigma, (sigma.shape[0], self.policy_out_dim))
        sigma = sigma.reshape(action_mean.shape)
        probs = Normal(action_mean,sigma)
        if action is None:
            action = probs.sample()

        if len(obs.shape) == 2:
            return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value
        else:
            return action, probs.log_prob(action).sum(0), probs.entropy().sum(0), value
        
    def get_action_extrinsics(self, obs,
                              env_extrinsics,
                              sigma=0.5
                            ):
        
        if len(obs.shape) == 1:
            pi_input = torch.cat((obs, env_extrinsics))
        else:
            pi_input = torch.cat((obs, env_extrinsics), dim=1)

        action_mean = self.actor_policy(pi_input)
        sigma = sigma.reshape(sigma.shape[0],1)
        sigma = torch.broadcast_to(sigma, (sigma.shape[0], self.policy_out_dim))
        sigma = sigma.reshape(action_mean.shape)
        probs = Normal(action_mean,sigma)
        action = probs.sample()

        return action