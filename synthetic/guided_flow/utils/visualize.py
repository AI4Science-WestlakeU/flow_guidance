import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch

def visualize_traj_and_vf(
        traj, 
        wrapped_model, 
        ode_num_steps, 
        x0_dist: str, 
        x1_dist: str, 
        device, 
        disp_traj_batch = 256, 
        x_lim = 2, 
        y_lim = 2, 

):
    fig, axs = plt.subplot_mosaic("AABC;AADE")
    fig.set_size_inches(12, 6)

    # 1. plot samples and trajectories
    ax = axs['A']
    # plot the start and end points with endpoint colors in the colormap
    blue = cm.get_cmap('coolwarm', 2)(0)
    red = cm.get_cmap('coolwarm', 2)(1)
    ax.scatter(traj[0, :disp_traj_batch, 0].cpu(), traj[0, :disp_traj_batch, 1].cpu(), s=3, color=blue)
    ax.scatter(traj[-1, :, 0].cpu(), traj[-1, :, 1].cpu(), s=3, color=red)
    # plot the trajectory with a gradient color
    colors = torch.linspace(0, 1, ode_num_steps).unsqueeze(1).repeat(1, disp_traj_batch).flatten().numpy()
    ax.scatter(
        traj[:, :disp_traj_batch, 0].flatten().cpu(), 
        traj[:, :disp_traj_batch, 1].flatten().cpu(), 
        c=plt.cm.coolwarm(colors), 
        alpha=0.2, s=0.1, marker='.'
    )
    ax.set_title(f'{x0_dist} to {x1_dist}')
    ax.set_xlim(-x_lim, x_lim)
    ax.set_ylim(-y_lim, y_lim)

    # 2. plot VFs
    for i, (ax_name, t) in enumerate(zip(['B', 'C', 'D', 'E'], torch.linspace(0, 1, 4, device=device))):
        ax = axs[ax_name]
        x = torch.linspace(-x_lim, x_lim, 20)
        y = torch.linspace(-y_lim, y_lim, 20)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        XY = torch.stack([X, Y], dim=-1).reshape(-1, 2).to(device)
        V = wrapped_model(t, XY).reshape(20, 20, 2).detach().cpu()
        ax.quiver(X, Y, V[:, :, 0], V[:, :, 1])
        ax.set_xlim(-x_lim, x_lim)
        ax.set_ylim(-y_lim, y_lim)
        ax.set_title(f"Vector field at t={t.item():0.3f}")

    fig.tight_layout()

    return fig, axs