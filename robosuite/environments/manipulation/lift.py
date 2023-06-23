from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat
from robosuite.models.base import MujocoModel
from robosuite.models.objects.composite_body.composite_box import CompositeBoxObject
from scipy.stats import norm


class Lift(SingleArmEnv):
    """
    This class corresponds to the lifting task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 1.6, 0.05),
        table_friction=(0, 0, 0),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        self.arm_limit_collision_penalty = -10.0
        self.dest_pos = np.array([-0.05, -0.6, 0.84999541])
        self.dest_ori = np.array([1, 0, 0, 0])
        self.start_pos = np.array([-0.105, -0.03, 0.84999541])
        self.start_ori = np.array([1, 0, 0, 0])
        
        self.begin_count = 0
        self.prev_dist = np.linalg.norm(self.start_pos - self.dest_pos)
        self.prev_obj_vel = 0
        self.prev_ori_dist = 1 - np.dot(self.start_ori, self.dest_ori)
        self.max_dist_move = -100
        self.max_COM_reward = -100

        self.gaussian_distribution = norm(loc=4, scale=1)
        self.global_step = 0
        
        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def reward(self, action=None):
        """
        Reward function for the task.
        """
        reward = 0.0
        self.begin_count += 1

        cube_pos = self.sim.data.body_xpos[self.red_body_id]
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]


        # ###################################
        # dist of gripper to center of mass reward
        # ###################################
        com = self.get_COM()
        com_distance = np.linalg.norm(gripper_site_pos - com)
        com_distance_tanh = (1 - np.tanh(com_distance))
        dense_com_reward = com_distance_tanh
        dense_com_reward = 0
        # print("dense_com_reward: ", dense_com_reward)


        # ###################################
        # dist of cube pos to dest pos reward
        # ###################################
        curr_dist = np.linalg.norm(cube_pos - self.dest_pos)
        dense_position_reward = (self.prev_dist - curr_dist)
        self.prev_dist = curr_dist
        dense_position_reward = 0
        # print("dense_position_reward: ", dense_position_reward)


        # ###################################
        # velocity of cube reward
        # ###################################
        curr_obj_vel = np.linalg.norm(self.sim.data.get_body_xvelp("composite_cube_root"))
        desired_obj_vel = 0.035
        desired_obj_vel_threshold = 0.01
        obj_vel_err = abs(desired_obj_vel - curr_obj_vel)
        curr_obj_vel_error_reward = 0
        min_obj_vel_threshold = 0.001
        # This small reward incentivizes making the object to move
        if curr_obj_vel > min_obj_vel_threshold and curr_obj_vel < desired_obj_vel-desired_obj_vel_threshold:
            curr_obj_vel_error_reward = 0.01
        # This larger reward incentivizes making the object to move at the right velocity
        if obj_vel_err < desired_obj_vel_threshold:
            curr_obj_vel_error_reward = 1

        
        # ###################################
        # cube constant velocity reward
        # ###################################
        obj_vel_change_reward = 0
        # obj_vel_change = self.prev_obj_vel - curr_obj_vel
        # obj_vel_change_reward = - np.square(obj_vel_change)
        # obj_vel_change_reward = obj_vel_change_reward * 10
        self.prev_obj_vel = curr_obj_vel


        # ###################################
        # cube orientation reward
        # ###################################
        dense_orientation_reward = 0
        curr_ori_dist = self.quats_angle_dist(self.sim.data.body_xquat[self.red_body_id], self.dest_ori)
        ori_potential_reward = (self.prev_ori_dist - curr_ori_dist)
        if curr_ori_dist < 2.5:
            if ori_potential_reward > 0:
                dense_orientation_reward = ori_potential_reward
        else:
            dense_orientation_reward = ori_potential_reward
        self.prev_ori_dist = curr_ori_dist
        # print(dense_orientation_reward)

        # ###################################
        # cube COM velocity reward
        # ###################################
        eff_vel_reward = 0
        eff_vel = np.linalg.norm(self.robots[0].recent_ee_vel.current[:3])
        if eff_vel >= 0.035:
            eff_vel_reward = (1 - (eff_vel - 0.035)/(0.12 - 0.035))
        eff_vel_reward = 0
    

        # Sparse reward components
        sparse_position_reward = 0
        sparse_orientation_reward = 0

        if self._check_success():
            sparse_orientation_reward = 1
            sparse_position_reward = 1


        # ###################################
        # gripper cube contact reward
        # gripper cube contact force reward
        # ###################################
        contact_reward = 0
        eff_force_reward = 0
        # Contact penalty
        if self.check_contact(geoms_1=["gripper0_finger1_collision",
                                       "gripper0_finger2_collision",
                                       "gripper0_finger1_pad_collision",
                                       "gripper0_finger2_pad_collision"], 
                                        geoms_2=self.CompositeBoxObject):

            contact_reward = 0.005
            # eff_force = np.linalg.norm(np.array(self.robots[0].recent_ee_forcetorques.current[:3]))
            eff_force_z = np.array(self.robots[0].recent_ee_forcetorques.current[:3])[2]
            # eff_force_reward = self.gaussian_distribution.pdf(eff_force_z)/self.gaussian_distribution.pdf(4)
            # eff_force_reward = eff_force_reward / 100 * 2 * 45
        # eff_force_reward = 0
        contact_reward = 0
        # eff_force_reward = 0
        

        w_obj_vel_r =  0.2
        w_dense_position_reward = 100
        w_dense_orientation_reward = 1/45 * 100 * 2

        total_reward = (
            dense_position_reward * w_dense_position_reward+ 
            dense_orientation_reward * w_dense_orientation_reward + 
            dense_com_reward + 
            contact_reward + 
            eff_force_reward +
            eff_vel_reward +
            curr_obj_vel_error_reward * w_obj_vel_r +
            obj_vel_change_reward +
            5 * (sparse_position_reward + sparse_orientation_reward)
        )

        reward_dict = {}
        reward_dict["total_reward"] = total_reward
        reward_dict["dense_position_reward"] = dense_position_reward
        reward_dict["dense_orientation_reward"] = dense_orientation_reward
        reward_dict["dense_com_reward"] = dense_com_reward
        reward_dict["contact_reward"] = contact_reward
        reward_dict["eff_force_reward"] = eff_force_reward
        reward_dict["eff_vel_reward"] = eff_vel_reward
        reward_dict["curr_obj_vel_error_reward"] = curr_obj_vel_error_reward
        reward_dict["obj_vel_change_reward"] = obj_vel_change_reward
        
        return reward_dict

    def _post_action(self, action):
        """
        In addition to super method, add additional info if requested

        Args:
            action (np.array): Action to execute within the environment

        Returns:
            3-tuple:

                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) info about current env step
        """
        reward, done, info = super()._post_action(action)
        
        done, reward, termination_stat = self._check_terminated(done, reward)

        info["termination_stat"] = termination_stat
        return reward, done, info
    
    def _check_terminated(self, done, reward):
        """
        Check if the task has completed one way or another. The following conditions lead to termination:

            - Collision
            - Task completion (pushing succeeded)
            - Joint Limit reached

        Returns:
            bool: True if episode is terminated
        """

        termination_stat = 0

        # Prematurely terminate if contacting the table with the arm
        if self.check_contact(self.robots[0].robot_model):
            reward["total_reward"] = self.arm_limit_collision_penalty
            print("joint collision happens")
            termination_stat = 1
            done = True

        if self.check_contact("gripper0_hand_collision"):
            reward["total_reward"] = self.arm_limit_collision_penalty
            print("gripper hand collision happens")
            termination_stat = 2
            done = True
        
        if self.robots[0].check_q_limits():
            reward["total_reward"] = self.arm_limit_collision_penalty
            print("reach joint limits")
            termination_stat = 3
            done = True
        
        if self.begin_count >= 3 and self.sim.data.body_xpos[self.red_body_id][2] >= 0.91:
            print(self.sim.data.body_xpos[self.red_body_id][2])
            reward["total_reward"] = self.arm_limit_collision_penalty
            print("lifting object happens")
            termination_stat = 4
            done = True

        if (measured_force:=np.linalg.norm(np.array(self.robots[0].recent_ee_forcetorques.current[:3]))) >= 100:
            print("too much force: ", measured_force)
            reward["total_reward"] = self.arm_limit_collision_penalty
            termination_stat = 5
            done = True

        if self.timestep >= self.horizon:
            termination_stat = 6
            print("timeout")
        
        # if self.global_step >= 300_000:
            # if (eff_vel:=np.linalg.norm(self.robots[0].recent_ee_vel.current[:3])) >= 0.12:
            #     print("eff move too fast: ", eff_vel)
            #     reward["total_reward"] = self.arm_limit_collision_penalty
            #     termination_stat = 7
            #     done = True

        # Prematurely terminate if task is success
        if self._check_success():
            termination_stat = 0
            done = True

        return done, reward, termination_stat

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        new_xpos = (xpos[0]+0.06, xpos[1]-0.06, xpos[2] -0.162)
        self.robots[0].robot_model.set_base_xpos(new_xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        self.CompositeBoxObject = CompositeBoxObject(
            name="composite_cube",
        )

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.CompositeBoxObject)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.CompositeBoxObject,
                x_range=[-0.03, 0.03],
                y_range=[-0.03, 0.03],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.CompositeBoxObject,
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.red_body_id = self.sim.model.body_name2id("composite_cube_box1_main")
        
        self.ball_id_0 = self.sim.model.body_name2id("composite_cube_ball0_main")
        self.ball_id_1 = self.sim.model.body_name2id("composite_cube_ball1_main")
        self.ball_id_2 = self.sim.model.body_name2id("composite_cube_ball2_main")
        self.ball_id_3 = self.sim.model.body_name2id("composite_cube_ball3_main")
        self.ball_id_4 = self.sim.model.body_name2id("composite_cube_ball4_main")
        self.ball_id_5 = self.sim.model.body_name2id("composite_cube_ball5_main")
        self.ball_id_6 = self.sim.model.body_name2id("composite_cube_ball6_main")
        self.ball_id_7 = self.sim.model.body_name2id("composite_cube_ball7_main")

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            # cube-related observables
            @sensor(modality=modality)
            def cube_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.red_body_id])

            @sensor(modality=modality)
            def cube_quat(obs_cache):
                return np.array(self.sim.data.body_xquat[self.red_body_id])

            @sensor(modality=modality)
            def gripper_to_cube_pos(obs_cache):
                return (
                    obs_cache[f"{pf}eef_pos"] - obs_cache["cube_pos"]
                    if f"{pf}eef_pos" in obs_cache and "cube_pos" in obs_cache
                    else np.zeros(3)
                )

            sensors = [cube_pos, cube_quat, gripper_to_cube_pos]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        self.begin_count = 0

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array([self.start_pos[0],self.start_pos[1],obj_pos[2]]), self.start_ori]))

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the cube.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the cube
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.CompositeBoxObject)
    
    def quats_angle_dist(self, quat1, quat2):
        """
        Calculate the angle between two quaternions
        """
        dot_product = np.dot(quat1, quat2)
        # Calculate the difference in orientation (angle between quaternions)
        orientation_error = 2 * np.arccos(abs(dot_product))
        orientation_error = orientation_error/np.pi * 180
        return orientation_error

    def _check_success(self):
        """
        Check if cube has been lifted.

        Returns:
            bool: True if cube has been lifted
        """
        position_error = np.linalg.norm(self.sim.data.body_xpos[self.red_body_id] - self.dest_pos)
        orientation_error = self.quats_angle_dist(self.sim.data.body_xquat[self.red_body_id], self.dest_ori)

        return position_error < 0.1 and orientation_error < 5
    
    def get_COM(self,):
        ball_pos_0 = self.sim.data.body_xpos[self.ball_id_0]
        ball_pos_1 = self.sim.data.body_xpos[self.ball_id_1]
        ball_pos_2 = self.sim.data.body_xpos[self.ball_id_2]
        ball_pos_3 = self.sim.data.body_xpos[self.ball_id_3]
        ball_pos_4 = self.sim.data.body_xpos[self.ball_id_4]
        ball_pos_5 = self.sim.data.body_xpos[self.ball_id_5]
        ball_pos_6 = self.sim.data.body_xpos[self.ball_id_6]
        ball_pos_7 = self.sim.data.body_xpos[self.ball_id_7]

        ball_mass_0 = self.sim.model.body_mass[self.ball_id_0]
        ball_mass_1 = self.sim.model.body_mass[self.ball_id_1]
        ball_mass_2 = self.sim.model.body_mass[self.ball_id_2]
        ball_mass_3 = self.sim.model.body_mass[self.ball_id_3]
        ball_mass_4 = self.sim.model.body_mass[self.ball_id_4]
        ball_mass_5 = self.sim.model.body_mass[self.ball_id_5]
        ball_mass_6 = self.sim.model.body_mass[self.ball_id_6]
        ball_mass_7 = self.sim.model.body_mass[self.ball_id_7]

        com = ((ball_pos_0 * ball_mass_0 + ball_pos_1 * ball_mass_1 + ball_pos_2 * ball_mass_2 + ball_pos_3 * ball_mass_3 + ball_pos_4 * ball_mass_4 + ball_pos_5 * ball_mass_5 + ball_pos_6 * ball_mass_6 + ball_pos_7 * ball_mass_7) 
                / (ball_mass_0 + ball_mass_1 + ball_mass_2 + ball_mass_3 + ball_mass_4 + ball_mass_5 + ball_mass_6 + ball_mass_7))
        return com
