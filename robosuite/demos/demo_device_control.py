"""Teleoperate robot with keyboard or SpaceMouse.

***Choose user input option with the --device argument***

Keyboard:
    We use the keyboard to control the end-effector of the robot.
    The keyboard provides 6-DoF control commands through various keys.
    The commands are mapped to joint velocities through an inverse kinematics
    solver from Bullet physics.

    Note:
        To run this script with macOS, you must run it with root access.

SpaceMouse:

    We use the SpaceMouse 3D mouse to control the end-effector of the robot.
    The mouse provides 6-DoF control commands. The commands are mapped to joint
    velocities through an inverse kinematics solver from Bullet physics.

    The two side buttons of SpaceMouse are used for controlling the grippers.

    SpaceMouse Wireless from 3Dconnexion: https://www.3dconnexion.com/spacemouse_wireless/en/
    We used the SpaceMouse Wireless in our experiments. The paper below used the same device
    to collect human demonstrations for imitation learning.

    Reinforcement and Imitation Learning for Diverse Visuomotor Skills
    Yuke Zhu, Ziyu Wang, Josh Merel, Andrei Rusu, Tom Erez, Serkan Cabi, Saran Tunyasuvunakool,
    János Kramár, Raia Hadsell, Nando de Freitas, Nicolas Heess
    RSS 2018

    Note:
        This current implementation only supports macOS (Linux support can be added).
        Download and install the driver before running the script:
            https://www.3dconnexion.com/service/drivers.html

Additionally, --pos_sensitivity and --rot_sensitivity provide relative gains for increasing / decreasing the user input
device sensitivity


***Choose controller with the --controller argument***

Choice of using either inverse kinematics controller (ik) or operational space controller (osc):
Main difference is that user inputs with ik's rotations are always taken relative to eef coordinate frame, whereas
    user inputs with osc's rotations are taken relative to global frame (i.e.: static / camera frame of reference).

    Notes:
        OSC also tends to be more computationally efficient since IK relies on the backend pybullet IK solver.


***Choose environment specifics with the following arguments***

    --environment: Task to perform, e.g.: "Lift", "TwoArmPegInHole", "NutAssembly", etc.

    --robots: Robot(s) with which to perform the task. Can be any in
        {"Panda", "Sawyer", "IIWA", "Jaco", "Kinova3", "UR5e", "Baxter"}. Note that the environments include sanity
        checks, such that a "TwoArm..." environment will only accept either a 2-tuple of robot names or a single
        bimanual robot name, according to the specified configuration (see below), and all other environments will
        only accept a single single-armed robot name

    --config: Exclusively applicable and only should be specified for "TwoArm..." environments. Specifies the robot
        configuration desired for the task. Options are {"bimanual", "single-arm-parallel", and "single-arm-opposed"}

            -"bimanual": Sets up the environment for a single bimanual robot. Expects a single bimanual robot name to
                be specified in the --robots argument

            -"single-arm-parallel": Sets up the environment such that two single-armed robots are stationed next to
                each other facing the same direction. Expects a 2-tuple of single-armed robot names to be specified
                in the --robots argument.

            -"single-arm-opposed": Sets up the environment such that two single-armed robots are stationed opposed from
                each other, facing each other from opposite directions. Expects a 2-tuple of single-armed robot names
                to be specified in the --robots argument.

    --arm: Exclusively applicable and only should be specified for "TwoArm..." environments. Specifies which of the
        multiple arm eef's to control. The other (passive) arm will remain stationary. Options are {"right", "left"}
        (from the point of view of the robot(s) facing against the viewer direction)

    --switch-on-grasp: Exclusively applicable and only should be specified for "TwoArm..." environments. If enabled,
        will switch the current arm being controlled every time the gripper input is pressed

    --toggle-camera-on-grasp: If enabled, gripper input presses will cycle through the available camera angles

Examples:

    For normal single-arm environment:
        $ python demo_device_control.py --environment PickPlaceCan --robots Sawyer --controller osc

    For two-arm bimanual environment:
        $ python demo_device_control.py --environment TwoArmLift --robots Baxter --config bimanual --arm left --controller osc

    For two-arm multi single-arm robot environment:
        $ python demo_device_control.py --environment TwoArmLift --robots Sawyer Sawyer --config single-arm-parallel --controller osc


"""

import argparse

import numpy as np

import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import VisualizationWrapper

from robosuite.utils.mjmod import DynamicsModder
import math
import time
import robosuite.utils.transform_utils as trans
from torch.utils.tensorboard import SummaryWriter

def quaternion_to_rotation_matrix(q):
        w, x, y, z = q
        R = np.array([
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)]
        ])
        return R

def normalize_quaternion(q):
    return q / np.linalg.norm(q)

def down_corner_pos(center_position, quat):
    
    half_width = 0.05
    half_height = 0.05
    half_length = 0.16

    local_corners = np.array([
        [-half_length, -half_width, -half_height],
        [half_length, -half_width, -half_height],
        [-half_length, half_width, -half_height],
        [half_length, half_width, -half_height]
    ])

    # Calculate the rotation matrix from the quaternion
    quat = normalize_quaternion(quat)
    R = quaternion_to_rotation_matrix(quat)

    # Apply the rotation matrix and translate to global positions
    global_corners = R @ local_corners.T
    global_corners = global_corners.T + center_position

    return global_corners

# table surface segment
def is_point_inside_rectangle(point, rectangle_center):
    half_length = 0.4
    half_width = 0.05
    
    relative_position = point - rectangle_center
    
    if -half_length <= relative_position[0] <= half_length and -half_width <= relative_position[1] <= half_width:
        return True
    return False

def get_corner_friction_list(env, corner_pos_list):
    corner_friction_list = []
    for corner_pos in corner_pos_list:
        for i in range(8):
            seg_id = env.sim.model.geom_name2id("table_collision_"+str(i))
            seg_pos = env.sim.data.geom_xpos[seg_id]
            if is_point_inside_rectangle(corner_pos, seg_pos):
                corner_friction_list.append(env.sim.model.geom_friction[seg_id][0])
                break
            elif i == 7:
                corner_friction_list.append(0.5)
    return corner_friction_list

def get_COM(env):
    ball_pos_0 = env.sim.data.body_xpos[env.ball_id_0]
    ball_pos_1 = env.sim.data.body_xpos[env.ball_id_1]
    ball_pos_2 = env.sim.data.body_xpos[env.ball_id_2]
    ball_pos_3 = env.sim.data.body_xpos[env.ball_id_3]
    ball_pos_4 = env.sim.data.body_xpos[env.ball_id_4]
    ball_pos_5 = env.sim.data.body_xpos[env.ball_id_5]
    ball_pos_6 = env.sim.data.body_xpos[env.ball_id_6]
    ball_pos_7 = env.sim.data.body_xpos[env.ball_id_7]

    ball_mass_0 = env.sim.model.body_mass[env.ball_id_0]
    ball_mass_1 = env.sim.model.body_mass[env.ball_id_1]
    ball_mass_2 = env.sim.model.body_mass[env.ball_id_2]
    ball_mass_3 = env.sim.model.body_mass[env.ball_id_3]
    ball_mass_4 = env.sim.model.body_mass[env.ball_id_4]
    ball_mass_5 = env.sim.model.body_mass[env.ball_id_5]
    ball_mass_6 = env.sim.model.body_mass[env.ball_id_6]
    ball_mass_7 = env.sim.model.body_mass[env.ball_id_7]

    com = ((ball_pos_0 * ball_mass_0 + ball_pos_1 * ball_mass_1 + ball_pos_2 * ball_mass_2 + ball_pos_3 * ball_mass_3 + ball_pos_4 * ball_mass_4 + ball_pos_5 * ball_mass_5 + ball_pos_6 * ball_mass_6 + ball_pos_7 * ball_mass_7) 
            / (ball_mass_0 + ball_mass_1 + ball_mass_2 + ball_mass_3 + ball_mass_4 + ball_mass_5 + ball_mass_6 + ball_mass_7))
    return com

def quat_to_euler(q):
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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    parser.add_argument(
        "--config", type=str, default="single-arm-opposed", help="Specified environment configuration if necessary"
    )
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--switch-on-grasp", action="store_true", help="Switch gripper control on gripper action")
    parser.add_argument("--toggle-camera-on-grasp", action="store_true", help="Switch camera angle on gripper action")
    parser.add_argument("--controller", type=str, default="osc", help="Choice of controller. Can be 'ik' or 'osc'")
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="How much to scale rotation user inputs")
    args = parser.parse_args()

    # Import controller config for EE IK or OSC (pos/ori)
    if args.controller == "ik":
        controller_name = "IK_POSE"
    elif args.controller == "osc":
        controller_name = "OSC_POSE"
        # controller_name = "OSC_POSITION"
    else:
        print("Error: Unsupported controller specified. Must be either 'ik' or 'osc'!")
        raise ValueError

    # Get controller config
    controller_config = load_controller_config(default_controller=controller_name)

    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config
    else:
        args.config = None

    # Create environment
    env = suite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="frontview",
        # render_camera="agentview",
        # render_camera="birdview",
        # render_camera="sideview",
        ignore_done=True,
        # ignore_done=False,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=10,
        hard_reset=False,
        initialization_noise=None,
    )

    # Wrap this environment in a visualization wrapper
    env = VisualizationWrapper(env, indicator_configs=None)

    # Setup printing options for numbers
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
        env.viewer.add_keypress_callback(device.on_press)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    else:
        raise Exception("Invalid device choice: choose either 'keyboard' or 'spacemouse'.")

    
    writer = SummaryWriter(f"runs/")
    steps = 0

    while True:
        # Reset the environment
        obs = env.reset()
        print("!!!!!")
        # limit_euler = trans.mat2euler(env.env.robots[0].controller.ee_ori_mat)
        # limit_euler = np.array([[limit_euler[0]-0.001, limit_euler[1]-0.001, -2.87],
        #                         [limit_euler[0]+0.001, limit_euler[1]+0.001, 2.87]])

        modder = DynamicsModder(sim=env.sim)
        # friction = np.random.uniform(low = 0, high = 0.5)
        # modder.mod("cube_main", "inertia", [0.02, 0.01, 0.01])
        
        # print(env.sim.model.body_inertia[env.sim.model.body_name2id("cube_main")])
        

        # print(env.sim.model._body_name2id.keys())
        # print(env.sim.model._geom_name2id.keys())
        # exit()

        # body_id = env.sim.model.body_name2id("composite_cube_box1_main")
        # print(env.sim.model.body_mass[body_id])
        # 
        # print(env.sim.model._geom_name2id.keys())
        
        # max mass = 2
        # min mass = 0.1
        # max friction = 1
        # min friction = 0.1

        # table_colors = np.flip(np.array(
        #                         [np.array([255, 255, 255, 255])/255.0, 
        #                          np.array([222, 195, 193, 255])/255.0, 
        #                          np.array([201, 156, 153, 255])/255.0, 
        #                          np.array([176, 117, 113, 255])/255.0, 
        #                          np.array([150, 81, 77, 255])/255.0, 
        #                          np.array([125, 52, 47, 255])/255.0, 
        #                          np.array([99, 30, 25, 255])/255.0, 
        #                          np.array([64, 11, 7, 255])/255.0]), axis=0)

        table_colors = np.array([np.array([255, 255, 255, 255])/255.0, 
                                 np.array([222, 195, 193, 255])/255.0, 
                                 np.array([201, 156, 153, 255])/255.0, 
                                 np.array([176, 117, 113, 255])/255.0, 
                                 np.array([150, 81, 77, 255])/255.0, 
                                 np.array([125, 52, 47, 255])/255.0, 
                                 np.array([99, 30, 25, 255])/255.0, 
                                 np.array([64, 11, 7, 255])/255.0])

        # table_frictions = np.random.uniform(0.1, 0.9, 8)
        table_frictions = [0.8] * 8
        print("table_frictions: ",table_frictions)
        sorted_indices = sorted(range(len(table_frictions)), key=lambda i: table_frictions[i])
        table_frictions = np.sort(table_frictions)
        print("table_frictions: ",table_frictions)
        # Get the rank for each element
        ranks = [0] * len(table_frictions)
        for rank, index in enumerate(sorted_indices):
            print("rank: ", rank, "index: ", index)
            # ranks[index] = rank
            tabel_geom_id = env.sim.model.geom_name2id("table_visual_"+str(index))
            env.sim.model.geom_rgba[tabel_geom_id, :4] = table_colors[rank]
            # tabel_geom_id = env.sim.model.geom_name2id("table_collision_"+str(index))
            modder.mod_friction("table_collision_"+str(index), [table_frictions[rank], 0.005, 0.0001])
            
        modder.update()
        for i in range(8):
            geom_id = env.sim.model.geom_name2id("table_collision_"+str(i))
            print("f",env.sim.model.geom_friction[geom_id])
            tabel_geom_id = env.sim.model.geom_name2id("table_visual_"+str(i))
            # print("c",env.sim.model.geom_rgba[tabel_geom_id])
        

        modder.mod_mass("composite_cube_box1_main", 0.4)
        modder.mod_friction("composite_cube_box1_g0", [0.01, 0.005, 0.0001])

        modder.mod_mass("composite_cube_ball0_main", 0.05)
        modder.mod_mass("composite_cube_ball1_main", 0.05)
        modder.mod_mass("composite_cube_ball2_main", 0.05)
        modder.mod_mass("composite_cube_ball3_main", 0.05)

        modder.mod_mass("composite_cube_ball4_main", 0.05)
        modder.mod_mass("composite_cube_ball5_main", 0.05)
        modder.mod_mass("composite_cube_ball6_main", 0.05)
        modder.mod_mass("composite_cube_ball7_main", 0.05)


        contact_friction = 0.01
        modder.mod_friction("gripper0_finger1_collision", [contact_friction, 0.005, 0.0001])
        modder.mod_friction("gripper0_finger2_collision", [contact_friction, 0.005, 0.0001])


        for i  in range(7):
            modder.mod_frictionloss("robot0_joint"+str(i+1), 8)
            modder.mod_damping("robot0_joint"+str(i+1), 30)
            if i == 5 or i == 6:
                modder.mod_frictionloss("robot0_joint"+str(i+1), 1)
                modder.mod_damping("robot0_joint"+str(i+1), 1)

        # for i  in range(7):
        #     modder.mod_frictionloss("robot0_joint"+str(i+1), 8)
        #     modder.mod_damping("robot0_joint"+str(i+1), 200)
        #     if i == 5 or i == 6:
        #         modder.mod_frictionloss("robot0_joint"+str(i+1), 1)
        #         modder.mod_damping("robot0_joint"+str(i+1), 1)

        modder.update()

        # modder.mod_mass("composite_cube_box1_main", 3)
        # modder.mod_mass("composite_cube_box2_main", 3)
        # modder.mod_friction("composite_cube_box1_g0", [0.2, 0.005, 0.0001])
        # modder.mod_friction("composite_cube_box2_g0", [0.2, 0.005, 0.0001])
        # modder.update()
        
        # self.sim.model.body_name2id

        # print()
        # print(env.sim.data.geom_xpos[env.sim.model.geom_name2id("table_collision_0")])
        # print(env.sim.data.geom_xpos[env.sim.model.geom_name2id("table_collision_1")])
        # print(env.sim.data.geom_xpos[env.sim.model.geom_name2id("table_collision_2")])
        # print(env.sim.data.geom_xpos[env.sim.model.geom_name2id("table_collision_3")])
        # print(env.sim.data.geom_xpos[env.sim.model.geom_name2id("table_collision_4")])
        # print(env.sim.data.geom_xpos[env.sim.model.geom_name2id("table_collision_5")])
        # print(env.sim.data.geom_xpos[env.sim.model.geom_name2id("table_collision_6")])
        # max_val = -100
        # Setup rendering
        cam_id = 0
        num_cam = len(env.sim.model.camera_names)
        env.render()

        # Initialize variables that should the maintained between resets
        last_grasp = 0

        # Initialize device control
        device.start_control()

        while True:
            # Set active robot
            active_robot = env.robots[0] if args.config == "bimanual" else env.robots[args.arm == "left"]

            # Get the newest action
            action, grasp = input2action(
                device=device, robot=active_robot, active_arm=args.arm, env_configuration=args.config
            )

            # If action is none, then this a reset so we should break
            if action is None:
                break

            # If the current grasp is active (1) and last grasp is not (-1) (i.e.: grasping input just pressed),
            # toggle arm control and / or camera viewing angle if requested
            if last_grasp < 0 < grasp:
                if args.switch_on_grasp:
                    args.arm = "left" if args.arm == "right" else "right"
                if args.toggle_camera_on_grasp:
                    cam_id = (cam_id + 1) % num_cam
                    env.viewer.set_camera(camera_id=cam_id)
            # Update last grasp
            last_grasp = grasp

            # Fill out the rest of the action space if necessary
            rem_action_dim = env.action_dim - action.size
            if rem_action_dim > 0:
                # Initialize remaining action space
                rem_action = np.zeros(rem_action_dim)
                # This is a multi-arm setting, choose which arm to control and fill the rest with zeros
                if args.arm == "right":
                    action = np.concatenate([action, rem_action])
                elif args.arm == "left":
                    action = np.concatenate([rem_action, action])
                else:
                    # Only right and left arms supported
                    print(
                        "Error: Unsupported arm specified -- "
                        "must be either 'right' or 'left'! Got: {}".format(args.arm)
                    )
            elif rem_action_dim < 0:
                # We're in an environment with no gripper action space, so trim the action space to be the action dim
                action = action[: env.action_dim]

            # print(action)
            # action[-1] = 0
            # Step through the simulation and render
            obs, reward, done, info = env.step(action)
            # print(action)
            # print(obs.keys())
            # print(obs["cube_pos"])
            # print(obs["robot0_joint_vel"])
            # print(env.sim.data.body_xpos[env.sim.model.body_name2id("cube_main")])
            # np.set_printoptions(suppress=True)
            # print(env.robots[0]._joint_positions)
            # print(np.linalg.norm(obs["cube_pos"][:2] - np.array([-0.174, -0.589])))
            # print(obs["robot0_eef_pos"])
            env.render()

            # red_pos = env.sim.data.body_xpos[env.red_body_id]

            # print(red_pos)
            # print(blue_pos)
            # print(com)
            # dist = np.linalg.norm(com - env.sim.data.site_xpos[env.robots[0].eef_site_id])
            # print(np.tanh(dist) - 0.08)

            # Calculate the dot product of the two quaternions
            # dot_product = np.dot(env.sim.data.body_xquat[env.cube_body_id], [1, 0, 0, 0])

            # Calculate the difference in orientation (angle between quaternions)
            # orientation_error = np.arccos(abs(dot_product))
            # print(orientation_error)

            # modder.mod_mass("composite_cube_box1_main", 1)
            # modder.mod_mass("composite_cube_box2_main", 3)
            # modder.update()

            np.set_printoptions(suppress=True)
            # print(env.sim.data.site_xpos[env.robots[0].eef_site_id])
            # print(env.sim.data.site_xpos[env.robots[0].eef_cylinder_id])
            # eef_cylinder_id

            # print("robot0_eef_pos: ",obs["robot0_eef_pos"])
            # print("cube_pos: ",obs["cube_pos"])
            # print("robot0_eef_quat: ",obs["robot0_eef_quat"])
            # print(obs["robot0_joint_pos_cos"])
            # print("cube_quat: ",np.dot(obs["cube_quat"],obs["cube_quat"]))


            # print(env.robots[0]._joint_positions)

            # print(env.robots[0].base_pos)
            # print(env.table_offset)
            
            # print(env.sim.data.body_xquat[env.blue_body_id])
            # print(env.sim.data.body_xquat[env.cube_body_id])
            # center_pos = env.sim.data.body_xpos[env.red_body_id]
            # print("center_pos: ",center_pos)

            print(np.linalg.norm(env.robots[0].recent_ee_vel.current[:3]))

            corner_pos_list = down_corner_pos(obs["cube_pos"], obs["cube_quat"])
            # print(corner_pos_list)
            # print(get_corner_friction_list(env, corner_pos_list))
            # print(np.linalg.norm(np.array(env.robots[0].recent_ee_forcetorques.current[:3])))
            # print(obs["cube_pos"])

            # print("get_COM: ", get_COM(env))
            # print(down_corner_pos(center_pos, env.sim.data.body_xquat[env.cube_body_id]))
            # print(env.sim.data.geom_xpos[env.sim.model.geom_name2id("table_collision_1")])

            # print(is_point_inside_rectangle(down_corner_pos(center_pos, env.sim.data.body_xquat[env.cube_body_id])[2], env.sim.data.geom_xpos[env.sim.model.geom_name2id("table_collision_3")]))
            # print()
            # print("!!!!")
            # print()
            # euler = trans.mat2euler(env.env.robots[0].controller.ee_ori_mat)

            # print(env.quats_angle_dist(obs["cube_quat"], env.dest_ori))

            # print(trans.quat2axisangle(obs["cube_quat"])/np.pi *180)
            # print(trans.quat2axisangle(obs["cube_quat"]/np.pi))

            # print(quat_to_euler(obs["cube_quat"])[2])
            # print(obs["cube_pos"])

            # print("base: ", env.robots[0].base_pos)
            # print("eef: ", obs["robot0_eef_pos"])
            # print("robot0_eef_quat: ", quat_to_euler(obs["robot0_eef_quat"]))
            # print("robot0_eef_quat: ", obs["robot0_eef_quat"])
            # print("cube_pos: ", obs["cube_pos"])
            # print(quat_to_euler(obs["cube_quat"]))



            # print(env.robots[0]._joint_positions)
            # print("robot0_eef_quat: ", obs["robot0_eef_pos"])
            # print(env.sim.data.body_xpos[env.red_body_id][2])

            # print(np.linalg.norm(np.array(env.robots[0].recent_ee_forcetorques.current[:3])))
            # writer.add_scalars(f'recv_force/', {
            #     'x': env.robots[0].recent_ee_forcetorques.current[:3][0],
            #     'y': env.robots[0].recent_ee_forcetorques.current[:3][1],
            #     'z': env.robots[0].recent_ee_forcetorques.current[:3][2],
            # }, steps)
            # print(np.array(env.robots[0].recent_ee_forcetorques.current[:3]))

            steps += 1

            # print(obs["robot0_eef_pos"])
            # print(obs["cube_pos"])

            # 'robot0_joint1', 'robot0_joint2', 'robot0_joint3', 'robot0_joint4', 'robot0_joint5', 'robot0_joint6', 'robot0_joint7'

            # print(env.sim.model._joint_name2id.keys())
            # modder.mod_frictionloss("robot0_joint1", 0)

            # if env.check_contact(geoms_1=["gripper0_finger1_collision",
            #                            "gripper0_finger2_collision",
            #                            "gripper0_finger1_pad_collision",
            #                            "gripper0_finger2_pad_collision"], 
            #                             geoms_2=env.CompositeBoxObject) and \
            #                             env.sim.data.site_xpos[env.robots[0].eef_site_id][2] < 0.916 and \
            #                             env.robots[0].recent_ee_forcetorques.current[:3][2] > 0.5:
            #     print(env.robots[0].recent_ee_forcetorques.current[:3][2])

            # print(action)
            # print()

            # print(env.robots[0].base_pos - obs["robot0_eef_pos"])

            # print("euler: ", euler)
            # env.env.robots[0].controller.orientation_limits = limit_euler
            # print("euler_limit: ", env.env.robots[0].controller.orientation_limits)
            # time.sleep(2)
            # obs = env.reset()


# 'robot0_gripper_qpos', 'robot0_gripper_qvel'