# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import argparse
import os
import random
import time
from distutils.util import strtobool
import copy
import shutil
import wandb
wandb.login(key="190caefcc554590440e42593bfd6931f88f46f16")
os.environ["WANDB_SILENT"] = "true"

# import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.multiprocessing as multiprocessing

import robosuite as suite
from robosuite.wrappers.gym_wrapper import GymWrapper
from robosuite import load_controller_config
from robosuite.controllers.joint_pos import JointPositionController
from policy import Agent

from data_worker import collect_data

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--total-timesteps", type=int, default=1000_000_000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=8e-5,
        help="the learning rate of the optimizer")
    # large batch size
    parser.add_argument("--num-steps", type=int, default=10242, 
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    # number of mini batch
    parser.add_argument("--num-minibatches", type=int, default=5,
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


def log_termination_stat(writer, termination_stat, global_step):
    epi_num = sum(termination_stat)
    wandb.log({"termination stat/success": termination_stat[0]/epi_num}, step=global_step)
    wandb.log({"termination stat/joint_collision": termination_stat[1]/epi_num}, step=global_step)
    wandb.log({"termination stat/hand_collision": termination_stat[2]/epi_num}, step=global_step)
    wandb.log({"termination stat/joint_limit": termination_stat[3]/epi_num}, step=global_step)
    wandb.log({"termination stat/lifting_obj": termination_stat[4]/epi_num}, step=global_step)
    wandb.log({"termination stat/over_100_N": termination_stat[5]/epi_num}, step=global_step)
    wandb.log({"termination stat/timeout": termination_stat[6]/epi_num}, step=global_step)
    wandb.log({"termination stat/speed_limit": termination_stat[7]/epi_num}, step=global_step)

if __name__ == "__main__":
    # for multiprocessing
    torch.multiprocessing.set_start_method('spawn')
    
    # run_name = f"{args.seed}__{int(time.time())}"
    run_name = "w_obj_vel_r"

    wandb.init(
      # Set the project where this run will be logged
      project="CRA_push", 
      # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
      name="cube_ori_"+run_name,
      # Track hyperparameters and run metadata
      config={
      "w_obj_vel_r": 0.2,
      "anneal_step_num" : 500_000,
      }
    )
    
    if not os.path.exists("top_policies"):
        os.makedirs("top_policies")
    if not os.path.exists("check_points"):
            os.makedirs("check_points")

    args = parse_args()
    

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
        "has_offscreen_renderer":False,
        "use_camera_obs":False,
        "reward_shaping":True,
        "horizon":400,
        # "render_camera":"birdview",
        # "camera_names":["birdview",],
        # "camera_segmentations":["element",],
        "control_freq": 10,
    }
    
    keys = ['robot0_joint_pos_cos', 
            'robot0_joint_pos_sin', 
            'robot0_joint_vel', 
            'robot0_eef_pos', 
            'robot0_eef_quat', 
            'cube_pos', 
            'cube_quat',
            ]

    env = suite.make(**config)
    env = GymWrapper(env, keys=keys)
    env.step_change = 21


    state_dim = 27
    env_factor_dim = 16
    extrinsics_dim = 4
    action_dim = 6
    num_past = 5
    policy_in_dim = state_dim + extrinsics_dim

    agent = Agent(policy_in_dim=policy_in_dim, 
                  policy_out_dim=action_dim, 
                  encoder_in=env_factor_dim, 
                  encoder_out=extrinsics_dim,
                  phase=1, 
                  device=device).to(device)

    for m in agent.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate)
    
    global_step = 0
    start_time = time.time()
    # number of large size of batch (10240) to collect
    num_updates = args.total_timesteps // args.batch_size
    
    anneal_step_num = 500_000
    best_complete_rate = 0
    top_10_policy_list = np.array([(0,0,0,copy.deepcopy(agent))]*10)

    for update in range(1, num_updates + 1):

        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            if lrnow <= 5.85e-5:
                lrnow = 5.85e-5
            optimizer.param_groups[0]["lr"] = lrnow
        
        obs = torch.empty(0).to(device)
        env_factor = torch.empty(0).to(device)
        actions = torch.empty(0).to(device)
        logprobs = torch.empty(0).to(device)
        rewards = torch.empty(0).to(device)
        dones = torch.empty(0).to(device)
        values = torch.empty(0).to(device)
        sigmas = torch.empty(0).to(device)

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
        

        
        def create_replay_buffer():
            manager = multiprocessing.Manager() 
            params = (config, keys, device, agent, anneal_step_num, args, state_dim, env_factor_dim, action_dim)

            num_data_per_process = int(args.num_steps/6)
            data_dict_1 = manager.dict()
            process_1 = multiprocessing.Process(target=collect_data, 
                                                args=(global_step, 
                                                      global_step + num_data_per_process,
                                                      data_dict_1, *params))

            data_dict_2 = manager.dict()
            process_2 = multiprocessing.Process(target=collect_data, 
                                                args=(global_step + num_data_per_process, 
                                                      global_step + num_data_per_process*2,
                                                      data_dict_2, *params))

            data_dict_3 = manager.dict()
            process_3 = multiprocessing.Process(target=collect_data, 
                                                args=(global_step + num_data_per_process*2, 
                                                      global_step + num_data_per_process*3,
                                                      data_dict_3, *params))

            data_dict_4 = manager.dict()
            process_4 = multiprocessing.Process(target=collect_data, 
                                                args=(global_step + num_data_per_process*3, 
                                                      global_step + num_data_per_process*4,
                                                      data_dict_4, *params))

            data_dict_5 = manager.dict()
            process_5 = multiprocessing.Process(target=collect_data, 
                                                args=(global_step + num_data_per_process*4, 
                                                      global_step + num_data_per_process*5,
                                                      data_dict_5, *params))

            data_dict_6 = manager.dict()
            process_6 = multiprocessing.Process(target=collect_data, 
                                                args=(global_step + num_data_per_process*5, 
                                                      global_step + num_data_per_process*6,
                                                      data_dict_6, *params))

            process_1.start()
            process_2.start()
            process_3.start()
            process_4.start()
            process_5.start()
            process_6.start()

            process_1.join()
            process_2.join()
            process_3.join()
            process_4.join()
            process_5.join()
            process_6.join()


            log_keys = ["avg_complete_rate",
                        "avg_ori_complete_rate",
                        "avg_pos_complete_rate",
                        "avg_episode_reward",
                        "avg_episode_dense_position_reward",
                        "avg_episode_dense_orientation_reward",
                        "avg_episode_dense_com_reward",
                        "avg_episode_contact_reward",
                        "avg_episode_len",
                        "avg_eff_force_reward",
                        "avg_episode_force",
                        "avg_eff_vel_reward",
                        "avg_eff_episode_vel",
                        "avg_curr_obj_vel_error_reward",
                        "avg_obj_vel_change_reward",
                        "avg_episode_obj_vel",]
            ppo_keys = ["obs", "env_factor", "actions", "logprobs", "rewards", "dones", "values", "sigmas"]
            data_dict_keys = ["data_dict_1", "data_dict_2", "data_dict_3", "data_dict_4", "data_dict_5", "data_dict_6"]


            for key in log_keys:
                for data_dict_key in data_dict_keys:
                    temp = locals()[data_dict_key][key]
                    if len(temp) > 0:
                        globals()[key].append(temp.flatten())
                globals()[key] = np.concatenate(globals()[key], dtype=object)
            
            for key in ppo_keys:
                for data_dict_key in data_dict_keys:
                    globals()[key] = torch.cat((globals()[key], locals()[data_dict_key][key].to(device))).to(device)
            
            for data_dict_key in data_dict_keys:
                globals()["termination_stat"] = globals()["termination_stat"] + locals()[data_dict_key]["termination_stat"]

            

        # start replaybuffer collection
        create_replay_buffer()
        global_step = global_step + obs.shape[0]



        sigma = 1
        if anneal_step_num - global_step <= 100_000:
            sigma = sigma * (100_000/anneal_step_num)
        else:
            sigma = sigma * ((anneal_step_num - global_step)/anneal_step_num)

        # save check points
        if global_step % 30726 == 0 and len(avg_complete_rate) != 0:
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
        
        wandb.log({"evaluation/episodic_completion_rate": (np.array(avg_complete_rate) > 0.99).sum()/len(avg_complete_rate)}, step=global_step)
        wandb.log({"evaluation/episodic_train_completion_rate": np.average(avg_complete_rate)}, step=global_step)
        wandb.log({"evaluation/episodic_train_position_completion_rate": np.average(avg_pos_complete_rate)}, step=global_step)
        wandb.log({"evaluation/episodic_train_orientation_completion_rate": np.average(avg_ori_complete_rate)}, step=global_step)
        wandb.log({"return/episodic_return": np.average(avg_episode_reward)}, step=global_step)
        wandb.log({"return/episodic_position_return": np.average(avg_episode_dense_position_reward)}, step=global_step)
        wandb.log({"return/episodic_orientation_return": np.average(avg_episode_dense_orientation_reward)}, step=global_step)
        # wandb.log({"return/episodic_com_return", np.average(avg_episode_dense_com_reward), global_step)
        wandb.log({"return/episodic_contact_return": np.average(avg_episode_contact_reward)}, step=global_step)
        wandb.log({"evaluation/episodic_length": np.average(avg_episode_len)}, global_step)
        wandb.log({"evaluation/std_episodic_completion_rate": np.std(avg_complete_rate)}, step=global_step)
        wandb.log({"hyper-param/noise": sigma},  step=global_step)
        wandb.log({"return/episodic_eff_force_return": np.average(avg_eff_force_reward)}, step=global_step)
        wandb.log({"evaluation/episodic_eff_force": np.average(avg_episode_force)}, step=global_step)
        wandb.log({"return/episodic_eff_vel_return": np.average(avg_eff_vel_reward)}, step=global_step)
        wandb.log({"evaluation/episodic_eff_vel": np.average(avg_eff_episode_vel)}, step=global_step)
        wandb.log({"return/episodic_curr_obj_vel_error_return": np.average(avg_curr_obj_vel_error_reward)}, step=global_step)
        wandb.log({"return/episodic_obj_vel_change_return": np.average(avg_obj_vel_change_reward)}, step=global_step)
        wandb.log({"evaluation/episodic_obj_vel": np.average(avg_episode_obj_vel)}, step=global_step)
        log_termination_stat(writer, termination_stat, global_step)

        
        next_obs = obs[-1]
        next_env_factor = env_factor[-1]
        next_done = dones[-1]
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
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs
        b_env_factor = env_factor
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
        wandb.log({"hyper-param/learning_rate": optimizer.param_groups[0]["lr"]}, step=global_step)
        wandb.log({"losses/value_loss": v_loss.item()}, step=global_step)
        wandb.log({"losses/policy_loss": pg_loss.item()}, step=global_step)
        wandb.log({"losses/entropy": entropy_loss.item()}, step=global_step)
        wandb.log({"losses/old_approx_kl": old_approx_kl.item()}, step=global_step)
        wandb.log({"losses/approx_kl": approx_kl.item()}, step=global_step)
        wandb.log({"losses/clipfrac": np.mean(clipfracs)}, step=global_step)
        wandb.log({"losses/explained_variance": explained_var}, step=global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        wandb.log({"losses/SPS": int(global_step / (time.time() - start_time))}, step=global_step)

    writer.close()
