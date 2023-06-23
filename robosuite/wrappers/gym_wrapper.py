"""
This file implements a wrapper for facilitating compatibility with OpenAI gym.
This is useful when using these environments with code that assumes a gym-like 
interface.
"""

import numpy as np
from gym import spaces
from gym.core import Env

from robosuite.wrappers import Wrapper
from robosuite.utils.mjmod import DynamicsModder
import imageio
import xml.etree.ElementTree as ET
from copy import deepcopy
from robosuite.utils.mjcf_utils import new_body, new_geom, new_site
import math
import time
from PIL import Image, ImageDraw, ImageFont
import robosuite.utils.transform_utils as trans


class GymWrapper(Wrapper, Env):
    """
    Initializes the Gym wrapper. Mimics many of the required functionalities of the Wrapper class
    found in the gym.core module

    Args:
        env (MujocoEnv): The environment to wrap.
        keys (None or list of str): If provided, each observation will
            consist of concatenated keys from the wrapped environment's
            observation dictionary. Defaults to proprio-state and object-state.

    Raises:
        AssertionError: [Object observations must be enabled if no keys]
    """

    def __init__(self, env, keys=None):
        # Run super method
        super().__init__(env=env)
        # Create name for gym
        robots = "".join([type(robot.robot_model).__name__ for robot in self.env.robots])
        self.name = robots + "_" + type(self.env).__name__

        # Get reward range
        self.reward_range = (0, self.env.reward_scale)

        if keys is None:
            keys = []
            # Add object obs if requested
            if self.env.use_object_obs:
                keys += ["object-state"]
            # Add image obs if requested
            if self.env.use_camera_obs:
                keys += [f"{cam_name}_image" for cam_name in self.env.camera_names]
            # Iterate over all robots to add to state
            for idx in range(len(self.env.robots)):
                keys += ["robot{}_proprio-state".format(idx)]
        self.keys = keys

        # Gym specific attributes
        self.env.spec = None
        self.metadata = None

        # set up observation and action spaces
        obs = self.env.reset()
        self.modality_dims = {key: obs[key].shape for key in self.keys}
        flat_ob = self._flatten_obs(obs)
        self.obs_dim = flat_ob.size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low=low, high=high)
        low, high = self.env.action_spec
        self.action_space = spaces.Box(low=low, high=high)

        self.step_change = 0
        self.writer = imageio.get_writer("test.mp4", fps=10)
        self.test_mode = False

        tabel_id_v0 = env.sim.model.geom_name2id("table_visual_0")
        tabel_id_v1 = env.sim.model.geom_name2id("table_visual_1")
        tabel_id_v2 = env.sim.model.geom_name2id("table_visual_2")
        tabel_id_v3 = env.sim.model.geom_name2id("table_visual_3")
        tabel_id_v4 = env.sim.model.geom_name2id("table_visual_4")
        tabel_id_v5 = env.sim.model.geom_name2id("table_visual_5")
        tabel_id_v6 = env.sim.model.geom_name2id("table_visual_6")
        tabel_id_v7 = env.sim.model.geom_name2id("table_visual_7")
        # env.sim.model.geom_rgba[geom_id, :4] = np.array([64, 11, 7, 255])/255.0

        self.table_colors = np.array([np.array([255, 255, 255, 255])/255.0, 
                                 np.array([222, 195, 193, 255])/255.0, 
                                 np.array([201, 156, 153, 255])/255.0, 
                                 np.array([176, 117, 113, 255])/255.0, 
                                 np.array([150, 81, 77, 255])/255.0, 
                                 np.array([125, 52, 47, 255])/255.0, 
                                 np.array([99, 30, 25, 255])/255.0, 
                                 np.array([64, 11, 7, 255])/255.0])
        self.num_past = 5
        self.action_dm = 10
        self.state_dm = 35
        self.p3_training = False
        self.max_com_dist_x = -100
        self.max_com_dist_y = -100
        self.max_com_dist_z = -100

    def _flatten_obs(self, obs_dict, verbose=False):
        """
        Filters keys of interest out and concatenate the information.

        Args:
            obs_dict (OrderedDict): ordered dictionary of observations
            verbose (bool): Whether to print out to console as observation keys are processed

        Returns:
            np.array: observations flattened into a 1d array
        """
        ob_lst = []
        for key in self.keys:
            if key in obs_dict:
                if verbose:
                    print("adding key: {}".format(key))
                ob_lst.append(np.array(obs_dict[key]).flatten())
        return np.concatenate(ob_lst)
    
    def create_log(self):
        self.behavior_plot = {}
        self.behavior_plot["trans_xyz"] = []
        self.behavior_plot["rot_xyz"] = []
        self.behavior_plot["trans_xyz_stiff"] = []
        self.behavior_plot["rot_xyz_stiff"] = []
        self.behavior_plot["frictions"] = []

    def reset(self):
        # loggger for behavior
        self.create_log()

        self.past_actions = np.zeros((self.num_past, self.action_dm))
        self.past_states = np.zeros((self.num_past, self.state_dm))

        # random inital box position
        self.env.start_pos = np.array([np.random.uniform(-0.13,-0.08), np.random.uniform(-0.04,-0.03), 0.84999541])
        self.env.start_ori = self.degs_to_quat(0,0,np.random.uniform(-5,5))

        # assign dest_pos and dest_ori
        self.env.dest_pos = np.array([-0.105, -0.589, 0.84999541])
        self.env.dest_ori = self.degs_to_quat(0,0,0)
        
        indicator_config = {
            "name": "indicator0",
            "type": "box",
            "size": [0.14, 0.08, 0.1],
            "rgba": [1, 0, 0, 0.5],
            "quat": self.env.dest_ori,
            "pos":self.env.dest_pos,
        }

        COM_indicator_config = {
            "name": "indicator1",
            "type": "sphere",
            "size": [0.01],
            "rgba": [1, 1, 1, 1],
            "pos":self.env.dest_pos,
        }
        self.indicator_configs = []
        self.indicator_configs.append(indicator_config)
        self.indicator_configs.append(COM_indicator_config)
        self.env.set_xml_processor(processor=self._add_indicators_to_model)


        ob_dict = self.env.reset()
        self.env.prev_dist = np.linalg.norm(self.env.sim.data.body_xpos[self.env.red_body_id] - self.env.dest_pos)
        self.env.prev_ori_dist = self.env.quats_angle_dist(self.env.sim.data.body_xquat[self.env.red_body_id], self.env.dest_ori)
        self.step_count = 0

        self.modder = DynamicsModder(sim=self.env.sim)
        for i  in range(7):
            self.modder.mod_frictionloss("robot0_joint"+str(i+1), 8)
            self.modder.mod_damping("robot0_joint"+str(i+1), 30)
            if i == 5 or i == 6:
                self.modder.mod_frictionloss("robot0_joint"+str(i+1), 2)
                self.modder.mod_damping("robot0_joint"+str(i+1), 2)

        self.modder.mod_friction("composite_cube_box1_g0", [0.01, 0.005, 0.0001])
        self.contact_friction = 0.01
        self.modder.mod_friction("gripper0_finger1_collision", [self.contact_friction, 0.005, 0.0001])
        self.modder.mod_friction("gripper0_finger2_collision", [self.contact_friction, 0.005, 0.0001])
        self.modder.mod_mass("composite_cube_box1_main", 0.1)
        table_frictions = np.random.uniform(0.1, 0.9, 8)
        sorted_indices = sorted(range(len(table_frictions)), key=lambda i: table_frictions[i])
        self.table_frictions = np.sort(table_frictions)
        for rank, index in enumerate(sorted_indices):
            tabel_geom_id = self.env.sim.model.geom_name2id("table_visual_"+str(index))
            self.env.sim.model.geom_rgba[tabel_geom_id, :4] = self.table_colors[rank]
            self.modder.mod_friction("table_collision_"+str(index), [self.table_frictions[rank], 0.005, 0.0001])

        self.total_mass = np.random.uniform(0.1, 1.5)
        self.side_mass_list = np.random.dirichlet(np.ones(2),size=1)[0]
        self.side_one_mass = self.side_mass_list[0] * self.total_mass
        self.side_two_mass = self.side_mass_list[1] * self.total_mass
        self.side_one_mass_list = np.random.dirichlet(np.ones(4),size=1)[0]
        self.side_two_mass_list = np.random.dirichlet(np.ones(4),size=1)[0]
        # self.side_one_mass_list = np.array([1,1,1,1])/4
        # self.side_two_mass_list = np.array([1,1,1,1])/4
        for i in range(4):
            self.side_one_mass_list[i] = np.clip(self.side_one_mass_list[i] * self.side_one_mass, 0.01, np.inf)
            self.side_two_mass_list[i] = np.clip(self.side_two_mass_list[i] * self.side_two_mass, 0.01, np.inf)
            self.modder.mod_mass("composite_cube_ball"+str(i)+"_main", self.side_one_mass_list[i])
            self.modder.mod_mass("composite_cube_ball"+str(i+4)+"_main", self.side_two_mass_list[i])
        self.ball_mass_list = np.concatenate(([self.side_one_mass], [self.side_two_mass]))
        self.modder.update()
        
        
        # find the friction of the table that the box is on
        corner_pos_list = self.down_corner_pos(ob_dict["cube_pos"], ob_dict["cube_quat"])
        self.friction_list = self.get_corner_friction_list(corner_pos_list)
        self.behavior_plot["frictions"].append(np.mean(self.friction_list))

        com = self.get_COM()

        # indicate COM with a sphere
        self.modder.mod_position("indicator1_body", com+[0,0,0.12])
        self.modder.update()

        gripper_pos = self.env.sim.data.site_xpos[self.env.robots[0].eef_site_id]
        com_to_gripper_pos = com - gripper_pos
        com_to_gripper_pos[0] = com_to_gripper_pos[0]/ 0.2
        com_to_gripper_pos[1] = com_to_gripper_pos[1]/ 0.05
        # com_to_gripper_pos[2] = com_to_gripper_pos[2]/ 1.0
        next_env_factor = com_to_gripper_pos[:2]

        next_env_factor = np.concatenate((next_env_factor,
                                    self.ball_mass_list/1.5,
                                    (self.friction_list-0.1)/0.8,)
                                    # self.joint_frictions-0.1/1.9,)
                                    )


        #  normalize the observation
        ob_dict["cube_quat"] = self.quat_to_euler(ob_dict["cube_quat"])[2] /15
        ob_dict["cube_pos"] = ob_dict["cube_pos"][:2]


        angle = trans.mat2euler(self.env.robots[0].controller.ee_ori_mat)[2]/np.pi * 180
        if angle < 0:
            angle = angle - (-180)
        else:
            angle = 180 - angle
            angle = -angle
        ob_dict["robot0_eef_quat"] = angle/15
        ob_dict["robot0_eef_pos"] = ob_dict["robot0_eef_pos"][:2]

        if self.test_mode:
            # self.writer.append_data(np.flip(ob_dict.pop("birdview_image"), axis=0))
            self.writer.append_data(self.add_num_to_img(np.flip(ob_dict.pop("birdview_image"), axis=0), self.step_count))

        if self.p3_training:
            self.prev_state = self._flatten_obs(ob_dict)
            next_env_factor = np.concatenate((self.past_actions, self.past_states),axis=1)
            return self._flatten_obs(ob_dict), next_env_factor
        else:
            return self._flatten_obs(ob_dict), next_env_factor

    def step(self, action):
        """
        Extends vanilla step() function call to return flattened observation instead of normal OrderedDict.

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:

                - (np.array) flattened observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """

        prev_action = np.copy(action)
        action = np.clip(action, -1.0, 1.0)
        stiffness = action[:3]
        stiffness = (stiffness + 1) * 150
        stiffness = np.insert(stiffness, 2, [150,150,150])
        eff_action = action[3:]
        eff_action = np.insert(eff_action, 2, [0,0,0])
        action = np.concatenate((stiffness, eff_action))
        action = np.append(action, -1)


        self.behavior_plot["trans_xyz_stiff"].append(action[:3])
        self.behavior_plot["rot_xyz_stiff"].append(action[3:6])
        self.behavior_plot["trans_xyz"].append(action[6:9])
        self.behavior_plot["rot_xyz"].append(action[9:12])

        self.step_count += 1
        if self.step_count % self.step_change == 0:

            # self.joint_frictions = np.random.uniform(0.1, 2, 7)
            # for i  in range(7):
            #     self.modder.mod_frictionloss("robot0_joint"+str(i+1), self.joint_frictions[i])
            
            # self.contact_friction = np.random.uniform(0.1, 0.9)
            self.contact_friction = 0.01
            self.modder.mod_friction("gripper0_finger1_collision", [self.contact_friction, 0.005, 0.0001])
            self.modder.mod_friction("gripper0_finger2_collision", [self.contact_friction, 0.005, 0.0001])

            self.total_mass = np.random.uniform(0.1, 1.5)
            self.side_mass_list = np.random.dirichlet(np.ones(2),size=1)[0]
            self.side_one_mass = self.side_mass_list[0] * self.total_mass
            self.side_two_mass = self.side_mass_list[1] * self.total_mass
            # self.side_one_mass_list = np.random.dirichlet(np.ones(4),size=1)[0]
            # self.side_two_mass_list = np.random.dirichlet(np.ones(4),size=1)[0]
            self.side_one_mass_list = np.array([1,1,1,1])/4
            self.side_two_mass_list = np.array([1,1,1,1])/4
            for i in range(4):
                self.side_one_mass_list[i] = np.clip(self.side_one_mass_list[i] * self.side_one_mass, 0.01, np.inf)
                self.side_two_mass_list[i] = np.clip(self.side_two_mass_list[i] * self.side_two_mass, 0.01, np.inf)
                self.modder.mod_mass("composite_cube_ball"+str(i)+"_main", self.side_one_mass_list[i])
                self.modder.mod_mass("composite_cube_ball"+str(i+4)+"_main", self.side_two_mass_list[i])
            self.ball_mass_list = np.concatenate(([self.side_one_mass], [self.side_two_mass]))
            self.modder.update()
        
        ob_dict, reward, done, info = self.env.step(action)

        
        # find the friction of the table that the box is on
        corner_pos_list = self.down_corner_pos(ob_dict["cube_pos"], ob_dict["cube_quat"])
        self.friction_list = self.get_corner_friction_list(corner_pos_list)
        self.behavior_plot["frictions"].append(np.mean(self.friction_list))

        # calculate the center of mass of the two boxes
        com = self.get_COM()
        
        # indicate COM with a sphere
        self.modder.mod_position("indicator1_body", com+[0,0,0.12])
        self.modder.update()

        gripper_pos = self.env.sim.data.site_xpos[self.env.robots[0].eef_site_id]
        com_to_gripper_pos = com - gripper_pos
        com_to_gripper_pos[0] = com_to_gripper_pos[0]/ 0.2
        com_to_gripper_pos[1] = com_to_gripper_pos[1]/ 0.05
        # com_to_gripper_pos[2] = com_to_gripper_pos[2]/ 1.0
        next_env_factor = com_to_gripper_pos[:2]
        

        next_env_factor = np.concatenate((next_env_factor,
                                    self.ball_mass_list/1.5,
                                    (self.friction_list-0.1)/0.8,)
                                    # self.joint_frictions-0.1/1.9,)
                                    )

        
        # check if the cube is in the goal position and ori and check success
        cube_pos = ob_dict["cube_pos"]
        dist_complete = 1 - np.linalg.norm(cube_pos - self.env.dest_pos) / np.linalg.norm(self.env.start_pos - self.env.dest_pos)
        ori_complete = 1 - (self.env.quats_angle_dist(ob_dict["cube_quat"], self.env.dest_ori)/90)
        complete_dict = {}
        complete_dict["position_complete"] = dist_complete
        complete_dict["orientation_complete"] = ori_complete
        complete = (0.5 * dist_complete) + (0.5 * ori_complete)
        if self.env._check_success():
            complete = 1.0
        complete_dict["total_complete"] = complete

        #  normalize the observation
        ob_dict["cube_pos"] = ob_dict["cube_pos"][:2]
        ob_dict["cube_quat"] = self.quat_to_euler(ob_dict["cube_quat"])[2] /15

        angle = trans.mat2euler(self.env.robots[0].controller.ee_ori_mat)[2]/np.pi * 180
        if angle < 0:
            angle = angle - (-180)
        else:
            angle = 180 - angle
            angle = -angle
        ob_dict["robot0_eef_quat"] = angle/15
        ob_dict["robot0_eef_pos"] = ob_dict["robot0_eef_pos"][:2]

        
        if self.test_mode and done:
            # self.writer.append_data(np.flip(ob_dict.pop("birdview_image"), axis=0))
            self.writer.append_data(self.add_num_to_img(np.flip(ob_dict.pop("birdview_image"), axis=0), self.step_count))
            self.writer.close()
            time.sleep(0.5)
        elif self.test_mode:
            # self.writer.append_data(np.flip(ob_dict.pop("birdview_image"), axis=0))
            self.writer.append_data(self.add_num_to_img(np.flip(ob_dict.pop("birdview_image"), axis=0), self.step_count))


        if self.p3_training:
            self.past_actions = np.concatenate((self.past_actions[1:], np.expand_dims(prev_action, axis=0)))
            self.past_states = np.concatenate((self.past_states[1:], np.expand_dims(self.prev_state, axis=0)))
            next_env_factor = np.concatenate((self.past_actions, self.past_states),axis=1)
            self.prev_state = self._flatten_obs(ob_dict)
            return self._flatten_obs(ob_dict), reward, done, complete_dict, next_env_factor, info
        else:
            return self._flatten_obs(ob_dict), reward, done, complete_dict, next_env_factor, info

    def _add_indicators_to_model(self, xml):
        """
        Adds indicators to the mujoco simulation model

        Args:
            xml (string): MJCF model in xml format, for the current simulation to be loaded
        """
        if self.indicator_configs is not None:
            root = ET.fromstring(xml)
            worldbody = root.find("worldbody")

            for indicator_config in self.indicator_configs:
                config = deepcopy(indicator_config)
                indicator_body = new_body(name=config["name"] + "_body", pos=config.pop("pos", (0, 0, 0)))
                indicator_body.append(new_site(**config))
                worldbody.append(indicator_body)

            xml = ET.tostring(root, encoding="utf8").decode("utf8")

        return xml

    def set_indicator_pos(self, indicator, pos):
        """
        Sets the specified @indicator to the desired position @pos

        Args:
            indicator (str): Name of the indicator to set
            pos (3-array): (x, y, z) Cartesian world coordinates to set the specified indicator to
        """
        # Make sure indicator is valid
        indicator_names = set(self.get_indicator_names())
        assert indicator in indicator_names, "Invalid indicator name specified. Valid options are {}, got {}".format(
            indicator_names, indicator
        )
        # Set the specified indicator
        self.env.sim.model.body_pos[self.env.sim.model.body_name2id(indicator + "_body")] = np.array(pos)
    
    def get_indicator_names(self):
        """
        Gets all indicator object names for this environment.

        Returns:
            list: Indicator names for this environment.
        """
        return (
            [ind_config["name"] for ind_config in self.indicator_configs] if self.indicator_configs is not None else []
        )

    def degs_to_quat(self, roll_deg, pitch_deg, yaw_deg):
        # Convert angles to radians.
        roll_rad = roll_deg * np.pi / 180
        pitch_rad = pitch_deg * np.pi / 180
        yaw_rad = yaw_deg * np.pi / 180

        # Calculate the quaternion components.
        cy = np.cos(yaw_rad / 2)
        sy = np.sin(yaw_rad / 2)
        cp = np.cos(pitch_rad / 2)
        sp = np.sin(pitch_rad / 2)
        cr = np.cos(roll_rad / 2)
        sr = np.sin(roll_rad / 2)

        # Construct the quaternion.
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        # Return the quaternion.
        return np.array([w, x, y, z])

    
    def quaternion_to_rotation_matrix(self, q):
        w, x, y, z = q
        R = np.array([
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)]
        ])
        return R

    def normalize_quaternion(self, q):
        return q / np.linalg.norm(q)

    def down_corner_pos(self, center_position, quat):
        
        half_width = 0.05
        half_height = 0.05
        half_length = 0.25

        local_corners = np.array([
            [-half_length, -half_width, -half_height],
            [half_length, -half_width, -half_height],
            [-half_length, half_width, -half_height],
            [half_length, half_width, -half_height]
        ])

        # Calculate the rotation matrix from the quaternion
        quat = self.normalize_quaternion(quat)
        R = self.quaternion_to_rotation_matrix(quat)

        # Apply the rotation matrix and translate to global positions
        global_corners = R @ local_corners.T
        global_corners = global_corners.T + center_position

        return global_corners
    
    # table surface segment
    def is_point_inside_rectangle(self, point, rectangle_center):
        half_length = 0.4
        half_width = 0.05
        
        relative_position = point - rectangle_center
        
        if -half_length <= relative_position[0] <= half_length and -half_width <= relative_position[1] <= half_width:
            return True
        return False
    
    def get_corner_friction_list(self, corner_pos_list):
        corner_friction_list = np.array([])
        for corner_pos in corner_pos_list:
            for i in range(8):
                seg_id = self.env.sim.model.geom_name2id("table_collision_"+str(i))
                seg_pos = self.env.sim.data.geom_xpos[seg_id]
                if self.is_point_inside_rectangle(corner_pos, seg_pos):
                    corner_friction_list = np.append(corner_friction_list, self.env.sim.model.geom_friction[seg_id][0])
                    break
                elif i == 7:
                    corner_friction_list = np.append(corner_friction_list, 0.5)
        return corner_friction_list
    
    def get_COM(self,):
        ball_pos_0 = self.env.sim.data.body_xpos[self.env.ball_id_0]
        ball_pos_1 = self.env.sim.data.body_xpos[self.env.ball_id_1]
        ball_pos_2 = self.env.sim.data.body_xpos[self.env.ball_id_2]
        ball_pos_3 = self.env.sim.data.body_xpos[self.env.ball_id_3]
        ball_pos_4 = self.env.sim.data.body_xpos[self.env.ball_id_4]
        ball_pos_5 = self.env.sim.data.body_xpos[self.env.ball_id_5]
        ball_pos_6 = self.env.sim.data.body_xpos[self.env.ball_id_6]
        ball_pos_7 = self.env.sim.data.body_xpos[self.env.ball_id_7]

        ball_mass_0 = self.env.sim.model.body_mass[self.env.ball_id_0]
        ball_mass_1 = self.env.sim.model.body_mass[self.env.ball_id_1]
        ball_mass_2 = self.env.sim.model.body_mass[self.env.ball_id_2]
        ball_mass_3 = self.env.sim.model.body_mass[self.env.ball_id_3]
        ball_mass_4 = self.env.sim.model.body_mass[self.env.ball_id_4]
        ball_mass_5 = self.env.sim.model.body_mass[self.env.ball_id_5]
        ball_mass_6 = self.env.sim.model.body_mass[self.env.ball_id_6]
        ball_mass_7 = self.env.sim.model.body_mass[self.env.ball_id_7]

        com = ((ball_pos_0 * ball_mass_0 + ball_pos_1 * ball_mass_1 + ball_pos_2 * ball_mass_2 + ball_pos_3 * ball_mass_3 + ball_pos_4 * ball_mass_4 + ball_pos_5 * ball_mass_5 + ball_pos_6 * ball_mass_6 + ball_pos_7 * ball_mass_7) 
                / (ball_mass_0 + ball_mass_1 + ball_mass_2 + ball_mass_3 + ball_mass_4 + ball_mass_5 + ball_mass_6 + ball_mass_7))
        return com
    
    def add_num_to_img(self, img_array, num):
        # Assuming 'img_array' is your numpy array image
        # img_array = img_array * 255  # replace this with your array
        # print("img_array.shape: ", img_array.shape)
        # print("img_array.d: ", img_array.dtype)

        # Convert numpy array to PIL Image
        img = Image.fromarray(img_array.astype('uint8')).convert('RGB')
        

        draw = ImageDraw.Draw(img)

        # Use a truetype font
        font = ImageFont.load_default()

        text = str(num)  # The number you want to add
        textwidth = draw.textlength(text, font)

        # Calculate x and y coordinates if you want the number in the top right corner
        margin = 10
        x = img.width - textwidth - margin
        y = margin

        # Draw the text
        draw.text((x, y), text, font=font, fill="black")

        # print("np.array(img): ", np.array(img).dtype)

        # Convert back to numpy array if needed
        img_array_with_text = np.array(img)
        return img_array_with_text

    def quat_to_euler(self, q):
        # Ensure the quaternion array is of the form [w, x, y, z]
        q = np.array(q)

        # Normalize the quaternion
        q = q / np.sqrt(np.dot(q, q))

        w, x, y, z = q[0], q[1], q[2], q[3]

        # Compute roll (x-axis rotation)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x**2 + y**2)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Compute pitch (y-axis rotation)
        sinp = 2.0 * (w * y - z * x)
        if np.abs(sinp) >= 1:
            # This is a singularity case where pitch is +/- 90 degrees and roll/yaw are dependent
            # We arbitrarily set roll to 0 and compute yaw
            pitch = np.sign(sinp) * np.pi / 2  # use 90 degrees if sinp > 0, -90 if sinp < 0
        else:
            pitch = np.arcsin(sinp)

        # Compute yaw (z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y**2 + z**2)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        # Convert the rotation to degrees for readability
        roll_deg = roll * (180.0 / np.pi)
        pitch_deg = pitch * (180.0 / np.pi)
        yaw_deg = yaw * (180.0 / np.pi)

        return roll_deg, pitch_deg, yaw_deg

    def seed(self, seed=None):
        """
        Utility function to set numpy seed

        Args:
            seed (None or int): If specified, numpy seed to set

        Raises:
            TypeError: [Seed must be integer]
        """
        # Seed the generator
        if seed is not None:
            try:
                np.random.seed(seed)
            except:
                TypeError("Seed must be an integer type!")

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Dummy function to be compatible with gym interface that simply returns environment reward

        Args:
            achieved_goal: [NOT USED]
            desired_goal: [NOT USED]
            info: [NOT USED]

        Returns:
            float: environment reward
        """
        # Dummy args used to mimic Wrapper interface
        return self.env.reward()
