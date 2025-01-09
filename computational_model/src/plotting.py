import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from .walls import state_from_onehot,state_from_loc,state_ind_from_state
# Set some reasonable plotting defaults
plt.rcParams['font.size'] = 16
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Plot progress function
def plot_progress(rews, vals, fname="figs/progress.png"):
    plt.figure(figsize=(6, 2.5))
    axs = [121, 122]  # subplot identifiers
    data = [rews, vals]
    ts = range(1, len(rews) + 1)
    labs = ["reward", "prediction"]
    for i in range(2):
        plt.subplot(axs[i])
        plt.plot(ts, data[i], "k-")
        plt.xlabel("epochs")
        plt.ylabel(labs[i])
        plt.title(labs[i])

    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    return None

# Arena lines plotting function
def arena_lines(ps, wall_loc, Larena, rew=True, col="k", rew_col="k", lw_arena=1., col_arena=np.ones(3)*0.3, lw_wall=10):
    Nstates = Larena**2
    for i in range(Larena + 1):
        plt.axvline(x=i + 0.5, color=col_arena, lw=lw_arena)
        plt.axhline(y=i + 0.5, color=col_arena, lw=lw_arena)

    if rew:
        rew_loc = state_from_onehot(Larena, ps)
        plt.scatter(rew_loc[0], rew_loc[1], c=rew_col, marker="*", s=350, zorder=50)

    for s in range(Nstates):
        for i in range(4):
            if bool(wall_loc[s, i]):
                state = state_from_loc(Larena, s)
                if i == 0:  # wall to the right
                    z1, z2 = state + [0.5, 0.5], state + [0.5, -0.5]
                elif i == 1:  # wall to the left
                    z1, z2 = state + [-0.5, 0.5], state + [-0.5, -0.5]
                elif i == 2:  # wall above
                    z1, z2 = state + [0.5, 0.5], state + [-0.5, 0.5]
                elif i == 3:  # wall below
                    z1, z2 = state + [0.5, -0.5], state + [-0.5, -0.5]
                plt.plot([z1[0], z2[0]], [z1[1], z2[1]], color=col, ls="-", lw=lw_wall)
    plt.xlim(0.49, Larena + 0.52)
    plt.ylim(0.48, Larena + 0.51)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')

# Plot arena function
def plot_arena(ps, wall_loc, Larena, ind=1):
    ps = ps[:, ind]
    wall_loc = wall_loc[:, :, ind]
    plt.figure(figsize=(6, 6))
    arena_lines(ps, wall_loc, Larena)
    plt.savefig("figs/wall/test_arena.png", bbox_inches="tight")
    plt.close()

# Plot rollout function
def plot_rollout(state, rollout, wall, Larena):
    col = [0.5, 0.8, 0.5] if bool(rollout[-1]) else [0.5, 0.5, 0.8]
    rollout = np.array(rollout, dtype=int)
    
    for a in rollout[:-1]:
        if a > 0.5:
            if wall[state_ind_from_state(Larena, state)[0], a] > 0.5:
                new_state = state
            else:
                new_state = state + [[1, 0], [-1, 0], [0, 1], [0, -1]][a]
            new_state = (new_state + Larena - 1) % Larena + 1
            x1, x2 = np.min([state[0], new_state[0]]), np.max([state[0], new_state[0]])
            y1, y2 = np.min([state[1], new_state[1]]), np.max([state[1], new_state[1]])
            
            lw = 5
            if x2 - x1 > 1.5:
                plt.plot([x1, x1-0.5], [y1, y2], ls="-", color=col, lw=lw)
                plt.plot([x2, x2+0.5], [y1, y2], ls="-", color=col, lw=lw)
            elif y2 - y1 > 1.5:
                plt.plot([x1, x2], [y1, y1-0.5], ls="-", color=col, lw=lw)
                plt.plot([x1, x2], [y2, y2+0.5], ls="-", color=col, lw=lw)
            else:
                plt.plot([x1, x2], [y1, y2], ls="-", color=col, lw=lw)
            state = new_state

# Function for plotting the agent's progress in the environment (GIF creation)
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess

def plot_weiji_gif(ps, wall_loc, states, as_, rews, Larena, RTs, fname,
                   Tplot=10, res=60, minframe=3, figsize=(4, 4), rewT=400,
                   fix_first_RT=True, first_RT=500, plot_rollouts=False,
                   rollout_time=120, rollouts=[], dpi=80, plot_rollout_frames=False):
    # Create a temp directory for storing images
    os.makedirs(f"{fname}_temp", exist_ok=True)
    Tplot = int(Tplot * 1e3 / res)  # convert to frames
    if fix_first_RT:
        RTs[:, 0] = first_RT  # fix the first RT

    for batch in range(ps.shape[1]):
        bstr = str(batch).zfill(2)
        rew_loc = state_from_onehot(Larena, ps[:, batch])

        Rtot = 0
        t = 0  # real time
        anum = 0  # number of actions
        rew_col = "lightgrey"

        while (anum < np.sum(as_[batch, :] > 0.5)) and (t < Tplot):
            anum += 1
            astr = str(anum).zfill(3)
            print(bstr, astr)

            RT = RTs[batch, anum]  # reaction time for this action
            nframe = max(int(round(RT / res)), minframe)  # number of frames to plot
            rewframes = 0

            if rews[batch, anum] > 0.5:
                rewframes = int(round(rewT / res))  # add a frame at reward

            if (anum > 1.5) and (rews[batch, anum - 1] > 0.5):
                rew_col = "k"  # show found reward

            R_increased = False
            frames = range(minframe - nframe + 1, minframe + rewframes + 1)
            frolls = {f: 0 for f in frames}

            if plot_rollouts or plot_rollout_frames:
                nroll = np.sum(rollouts[batch, anum, 0, :] > 0.5)
                print("rolls:", nroll)
                f_per_roll = int(round(rollout_time / res))
                frames = range(min(frames[0], -nroll * f_per_roll + 1), frames[-1] + 1)
                frolls = {f: 0 for f in frames}

                for roll in range(1, nroll + 1):
                    new_rolls = range(-(f_per_roll * roll - 1), -(f_per_roll * (roll - 1)))
                    frac = 0.5 if nroll == 1 else (roll - 1) / (nroll - 1)
                    new_roll_1 = int(round(frames[0] * frac - (f_per_roll - 1) * (1 - frac)))
                    new_rolls = range(new_roll_1, new_roll_1 + f_per_roll)
                    for r in new_rolls:
                        frolls[r] = nroll - roll + 1

            for f in frames:
                state = states[:, batch, anum]
                fstr = str(f - frames[0]).zfill(3)

                frac = min(max(0, (f - 1) / minframe), 1)
                plt.figure(figsize=figsize)

                arena_lines(ps[:, batch], wall_loc[:, :, batch], Larena, col="k", rew_col=rew_col)

                col = "b"
                if (rewframes > 0) and (frac >= 1):
                    col = "g"  # colour green when at reward
                    if not R_increased:
                        Rtot += 1
                        R_increased = True

                if plot_rollouts and frolls[f] > 0.5:
                    plot_rollout(state, rollouts[batch, anum, :, frolls[f]], wall_loc[:, :, batch], Larena)

                a = as_[batch, anum]
                state += frac * [int(a == 1) - int(a == 2), int(a == 3) - int(a == 4)]
                state = (state + Larena - 0.5) % Larena + 0.5
                plt.scatter([state[0]], [state[1]], marker="o", color=col, s=200, zorder=100)

                tstr = str(t).zfill(3)
                t += 1
                realt = t * res * 1e-3
                print( f, round(frac, 2), RT, round(realt, 1))
                plt.title(f"t = {round(realt, 1)} (R = {Rtot})")

                astr = str(anum).zfill(2)
                if t <= Tplot:
                    plt.savefig(
                        f"{fname}_temp/temp{bstr}_{tstr}_{fstr}_{astr}.png",
                        bbox_inches="tight",
                        dpi=dpi,
                    )
                plt.close()

    # Combine PNGs into a GIF
    subprocess.run(f"convert -delay {int(round(res / 10))} {fname}_temp/temp*.png {fname}.gif", shell=True)

    # Remove temporary files
    subprocess.run(f"rm -r {fname}_temp", shell=True)


