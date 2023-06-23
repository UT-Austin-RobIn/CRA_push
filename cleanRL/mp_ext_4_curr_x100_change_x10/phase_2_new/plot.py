from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt


folder = "plot_1/"
trans_xyz = np.load(folder+"trans_xyz.npy")
trans_xyz_stiff = np.load(folder+"trans_xyz_stiff.npy")

rot_xyz = np.load(folder+"rot_xyz.npy")
rot_xyz_stiff = np.load(folder+"rot_xyz_stiff.npy")

recv_force_xyz = np.load(folder+"recv_force_xyz.npy")
obj_mass = np.load(folder+"obj_mass.npy")
applied_force = np.load(folder+"applied_force.npy")
obj_vel = np.load(folder+"obj_vel.npy")
total_reward = np.load(folder+"total_reward.npy")
eff_force_reward = np.load(folder+"eff_force_reward.npy")


def plot_3_curve(arr_3d, t, name):
    plt.plot(t, arr_3d[:, 0], label='X '+name)
    plt.plot(t, arr_3d[:, 1], label='Y '+name)
    plt.plot(t, arr_3d[:, 2], label='Z '+name)

    # adding labels and a legend
    plt.xlabel('Time/Step')
    plt.ylabel(name)
    plt.legend()

    # displaying the plot
    # plt.show()
    plt.savefig(folder+name+".png")
    plt.close()

def plot_1_curve(arr_1d, t, name):
    plt.plot(t, arr_1d, label='X '+name)
    # adding labels and a legend
    plt.xlabel('Time/Step')
    plt.ylabel(name)
    plt.legend()

    # displaying the plot
    # plt.show()
    plt.savefig(folder+name+".png")
    plt.close()

plot_3_curve(trans_xyz, range(trans_xyz.shape[0]), "trans_xyz")
plot_3_curve(trans_xyz_stiff, range(trans_xyz_stiff.shape[0]), "trans_xyz_stiff")
plot_3_curve(np.squeeze(applied_force,axis=1), range(applied_force.shape[0]), "commanded_force")
plot_3_curve(recv_force_xyz, range(recv_force_xyz.shape[0]), "measured_force_xyz")
plot_3_curve(obj_vel, range(obj_vel.shape[0]), "obj_vel")

plot_3_curve(rot_xyz, range(rot_xyz.shape[0]), "rot_xyz")
plot_3_curve(rot_xyz_stiff, range(rot_xyz_stiff.shape[0]), "rot_xyz_stiff")

plot_1_curve(total_reward, range(len(total_reward)), "total_reward")
plot_1_curve(eff_force_reward, range(len(eff_force_reward)), "eff_force_reward")
plot_1_curve(obj_mass, range(len(obj_mass)), "obj_mass")



# plt.plot(range(len(total_reward)), total_reward, label='total_reward')
# plt.xlabel('Time/Step')
# plt.ylabel('total_reward')
# plt.legend()
# plt.savefig(folder+"total_reward.png")
# plt.close()

# plt.close()
# plt.plot(range(len(eff_force_reward)), eff_force_reward, label='eff_force_reward')
# plt.xlabel('Time/Step')
# plt.ylabel('eff_force_reward')
# plt.legend()
# plt.savefig(folder+"eff_force_reward.png")
# plt.close()

# plt.plot(range(len(obj_mass)), obj_mass, label='mass')
# plt.xlabel('Time/Step')
# plt.ylabel('Mass')
# plt.legend()
# plt.savefig(folder+"Mass.png")

