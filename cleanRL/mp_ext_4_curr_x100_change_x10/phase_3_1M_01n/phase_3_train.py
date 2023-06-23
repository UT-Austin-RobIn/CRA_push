# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
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
# from torch.distributions import Independent, Normal

import robosuite as suite
from robosuite.wrappers.gym_wrapper import GymWrapper
from robosuite import load_controller_config
from robosuite.controllers.joint_pos import JointPositionController


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    # parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
    #     help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=42,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--device", type=str, default="0",
        help="which gpu to use")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    # parser.add_argument("--en 
    parser.add_argument("--total-timesteps", type=int, default=1000_000_000,
    # parser.add_argument("--total-timesteps", type=int, default=512_000,
    # parser.add_argument("--total-timesteps", type=int, default=200,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=8e-5,
    # parser.add_argument("--learning-rate", type=float, default=8e-6,
    # parser.add_argument("--learning-rate", type=float, default=4e-5,
        help="the learning rate of the optimizer")
    # !!!!!!!!!!!!!!!!!!!!
    # large batch size
    parser.add_argument("--num-steps", type=int, default=10240,
    # parser.add_argument("--num-steps", type=int, default=200,   
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    # number of mini batch
    parser.add_argument("--num-minibatches", type=int, default=5,
    # parser.add_argument("--num-minibatches", type=int, default=100,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=7,
        help="the K epochs to update the policy")
    
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.25,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    
    # large batch size
    args.batch_size = int(args.num_steps)
    # mini batch size
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
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

        self.actor_env_encoder = Env_Encoder(in_dim=9,out_dim=2)
        self.critic_env_encoder = Env_Encoder(in_dim=9,out_dim=2)

        self.actor_policy = Policy(policy_in_dim, policy_out_dim)
        self.critic_policy = Policy(policy_in_dim, 1)

        self.policy_out_dim = policy_out_dim

        self.device = device

    def get_value(self, obs, 
                  env_factor, 
                #   img_1
                  ):
        env_extrinsics = self.critic_env_encoder(env_factor)
        if len(obs.shape) == 1:
            pi_input = torch.cat((obs, env_extrinsics))
        else:
            pi_input = torch.cat((obs, env_extrinsics), dim=1)

        # pi_input = obs
        return self.critic_policy(pi_input)
    
    def get_action(self, obs, 
                   env_factor, 
                #    img_1
                   ):
        env_extrinsics = self.actor_env_encoder(env_factor)
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

def log_termination_stat(writer, termination_stat, global_step):
    epi_num = sum(termination_stat)
    writer.add_scalar("termination stat/success", termination_stat[0]/epi_num, global_step)
    writer.add_scalar("termination stat/joint_collision", termination_stat[1]/epi_num, global_step)
    writer.add_scalar("termination stat/hand_collision", termination_stat[2]/epi_num, global_step)
    writer.add_scalar("termination stat/joint_limit", termination_stat[3]/epi_num, global_step)
    writer.add_scalar("termination stat/lifting_obj", termination_stat[4]/epi_num, global_step)
    writer.add_scalar("termination stat/over_100_N", termination_stat[5]/epi_num, global_step)
    writer.add_scalar("termination stat/timeout", termination_stat[6]/epi_num, global_step)

if __name__ == "__main__":

    if not os.path.exists("top_policies"):
        os.makedirs("top_policies")
    if not os.path.exists("check_points"):
            os.makedirs("check_points")

    args = parse_args()
    # run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    run_name = f"{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            # monitor_gym=True, no longer works for gymnasium
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda:"+args.device if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    controller_config = load_controller_config(default_controller="OSC_POSE")
    controller_config["impedance_mode"] = "variable_kp"

    config = {
        "env_name": "Lift",
        "robots": "Panda",
        "controller_configs": controller_config,
        # "has_renderer":True,
        # "has_offscreen_renderer":True,
        "use_camera_obs":False,
        # "reward_scale":None,
        "reward_shaping":True,
        "horizon":400,
        # "render_camera":"birdview",
        # "camera_names":["frontview",],
        # "camera_segmentations":["element",],
        # "phase":1,
        "control_freq": 10,
    }
    
    # print(env.observation_spec().keys())
    keys = ['robot0_joint_pos_cos', 
            'robot0_joint_pos_sin', 
            'robot0_joint_vel', 
            'robot0_eef_pos', 
            'robot0_eef_quat', 
            'cube_pos', 
            # 'cube_quat'
            ]

    env = suite.make(**config)
    env = GymWrapper(env, keys=keys)

    env.step_change = 21
    env.num_past = 5
    env.action_dm = 6
    env.state_dm = 27
    env.p3_training = True
    
    state_dim = 27
    env_factor_dim = 33
    extrinsics_dim = 2
    action_dim = 6
    policy_in_dim = state_dim + extrinsics_dim
    act_obs_sum_dim = state_dim + action_dim

    agent = Agent(policy_in_dim, action_dim, device).to(device)
    agent.load_state_dict(torch.load("0.9326_0.2507_1638400.pt", map_location=device))

    agent_adapt_actor = ProprioAdaptTConv(act_obs_sum_dim, extrinsics_dim).to(device)
    agent_adapt_actor.load_state_dict(torch.load("actor_0.8901098901098901.pth", map_location=device))

    agent_adapt_critic = ProprioAdaptTConv(act_obs_sum_dim, extrinsics_dim).to(device)
    agent_adapt_critic.load_state_dict(torch.load("critic_0.8901098901098901.pth", map_location=device))

    agent.actor_env_encoder = agent_adapt_actor
    agent.critic_env_encoder = agent_adapt_critic


    optimizer = optim.Adam(list(agent.actor_policy.parameters()) + list(agent.critic_policy.parameters()), lr=args.learning_rate)
    
    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps,) + (state_dim,)).to(device)
    env_factor = torch.zeros((args.num_steps,) + (env.num_past, env_factor_dim,)).to(device)
    # img_1s = torch.zeros((args.num_steps,) + (num_past,84,84)).to(device)
    actions = torch.zeros((args.num_steps,) + (action_dim,)).to(device)
    logprobs = torch.zeros((args.num_steps,)).to(device)
    rewards = torch.zeros((args.num_steps,)).to(device)
    dones = torch.zeros((args.num_steps,)).to(device)
    values = torch.zeros((args.num_steps,)).to(device)
    sigmas = torch.zeros((args.num_steps,)).to(device)


    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    num_updates = args.total_timesteps // args.batch_size

    
    anneal_step_num = 1_000_000
    best_complete_rate = 0
    top_10_policy_list = np.array([(0,0,0,copy.deepcopy(agent))]*10)

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            if lrnow <= 5.85e-5:
                lrnow = 5.85e-5
            optimizer.param_groups[0]["lr"] = lrnow
        
        next_obs, next_env_factor = env.reset()
        next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float32)
        next_env_factor = torch.as_tensor(next_env_factor, device=device, dtype=torch.float32)
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
        termination_stat = np.zeros(7)

        episode_reward = 0
        episode_dense_position_reward = 0
        episode_dense_orientation_reward = 0
        episode_dense_com_reward = 0
        episode_contact_reward = 0
        episode_len = 0
        episode_eff_force_reward = 0
        episode_force = []

        for step in range(0, args.num_steps):
            global_step += 1

            obs[step] = next_obs
            env_factor[step] = next_env_factor
            dones[step] = next_done

            sigma = 0.1
            if anneal_step_num - global_step <= 100_000:
                sigma = sigma * (100_000/anneal_step_num)
            else:
                sigma = sigma * ((anneal_step_num - global_step)/anneal_step_num)
            sigma = torch.as_tensor([sigma], device=device, dtype=torch.float32)

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs=next_obs, 
                                                                       env_factor=next_env_factor, 
                                                                       sigma=sigma)
            sigmas[step] = sigma
            values[step] = value
            actions[step] = action
            logprobs[step] = logprob

            
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
            episode_force.append(np.array(env.robots[0].recent_ee_forcetorques.current[:3])[2])

            episode_len += 1
            next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float32)
            rewards[step] = torch.as_tensor(reward["total_reward"], device=device, dtype=torch.float32)
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
                termination_stat[infos["termination_stat"]] += 1
            
                episode_reward = 0
                episode_dense_position_reward = 0
                episode_dense_orientation_reward = 0
                episode_dense_com_reward = 0
                episode_contact_reward = 0
                episode_len = 0
                episode_eff_force_reward = 0
                episode_force = []
                next_obs, next_env_factor = env.reset()
                
                next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float32)
                next_env_factor = torch.as_tensor(next_env_factor, device=device, dtype=torch.float32)
                # next_done = torch.zeros(1).to(device)
            
            if global_step % 25_000 == 0 and len(avg_complete_rate) != 0:
                f = open(
                        "check_points/"
                        +str(global_step)
                        +"_"+str(round(np.average(avg_complete_rate), 4))
                        +"_"+str(round(np.std(avg_complete_rate), 4))
                        +".pt"
                        ,"wb", buffering=0)
                torch.save(agent.state_dict(),f)
                f.flush()
                os.fsync(f.fileno())
                f.close()

        # Finish collect data in one mini-batch
        pos = np.searchsorted(top_10_policy_list[:,0], (np.array(avg_complete_rate) > 0.99).sum()/len(avg_complete_rate), side="right")
        if pos != 0:
            new_top_policy = [(np.array(avg_complete_rate) > 0.99).sum()/len(avg_complete_rate), np.std(np.array(avg_complete_rate) > 0.99), global_step, copy.deepcopy(agent)]
            top_10_policy_list = np.insert(top_10_policy_list, pos, new_top_policy, axis=0)[1:]
            shutil.rmtree("top_policies")
            os.makedirs("top_policies")
            for policy_tuple in top_10_policy_list:
                f = open(
                    "top_policies/"
                    +str(round(policy_tuple[0], 4))
                    +"_"+str(round(policy_tuple[1], 4))
                    +"_"+str(policy_tuple[2])
                    +".pt"
                    ,"wb", buffering=0)
                torch.save(policy_tuple[3].state_dict(),f)
                f.flush()
                os.fsync(f.fileno())
                f.close()

        writer.add_scalar("evaluation/episodic_completion_rate", (np.array(avg_complete_rate) > 0.99).sum()/len(avg_complete_rate), global_step)
        writer.add_scalar("evaluation/episodic_train_completion_rate", np.average(avg_complete_rate), global_step)
        writer.add_scalar("evaluation/episodic_train_position_completion_rate", np.average(avg_pos_complete_rate), global_step)
        writer.add_scalar("evaluation/episodic_train_orientation_completion_rate", np.average(avg_ori_complete_rate), global_step)
        writer.add_scalar("return/episodic_return", np.average(avg_episode_reward), global_step)
        writer.add_scalar("return/episodic_position_return", np.average(avg_episode_dense_position_reward), global_step)
        writer.add_scalar("return/episodic_orientation_return", np.average(avg_episode_dense_orientation_reward), global_step)
        # writer.add_scalar("return/episodic_com_return", np.average(avg_episode_dense_com_reward), global_step)
        writer.add_scalar("return/episodic_contact_return", np.average(avg_episode_contact_reward), global_step)
        writer.add_scalar("evaluation/episodic_length", np.average(avg_episode_len), global_step)
        writer.add_scalar("evaluation/std_episodic_completion_rate", np.std(avg_complete_rate), global_step)
        writer.add_scalar("hyper-param/noise", sigma, global_step)
        writer.add_scalar("return/episodic_eff_force_return", np.average(avg_eff_force_reward), global_step)
        writer.add_scalar("evaluation/episodic_eff_force", np.average(avg_episode_force), global_step)
        log_termination_stat(writer, termination_stat, global_step)
        
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
            
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs, 
                                         next_env_factor, 
                                         ).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                nextnonterminal = 1.0
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t] # this is extra
                # delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs
        b_env_factor = env_factor
        # b_img_1s = img_1s

        b_logprobs = logprobs
        b_actions = actions
        b_advantages = advantages
        b_returns = returns
        b_values = values
        b_sigmas = sigmas


        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []

        # Backpropagte update_epochs times on this batch of data
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            # Divide and backpropagte over this minibatch_size of data
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(obs=b_obs[mb_inds],
                                                                              env_factor=b_env_factor[mb_inds],
                                                                            #   img_1=b_img_1s[mb_inds],
                                                                              action=b_actions[mb_inds],
                                                                              sigma=b_sigmas[mb_inds],
                                                                              )
                
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("hyper-param/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("losses/SPS", int(global_step / (time.time() - start_time)), global_step)

    writer.close()
