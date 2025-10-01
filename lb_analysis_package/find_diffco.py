import numpy as np
import mdtraj as md
import time


def msd_2d(traj: str, topo: str, frame_duration: int, start_frame=0):
    """
    Calculate the 2D mean squared displacement of a trajectory.
    Preferably use a .xtc trajectory file with .pdb topology.

    Parameters
    ----------
    traj : string
        Trajectory filename
    topo : string
        Topology filename
    frame_duration : int
        Duration of each frame in femtoseconds
    start_frame : int
        Starting frame for the MSD calculation

    Returns
    -------
    msd : np.ndarray
        Mean squared displacement of shape (n_frames, n_particles)
    num_frames : int
        Number of frames in the trajectory
    """
    dim = 2
    print('Loading trajectory...')
    tic = time.perf_counter()
    if not traj.endswith('.pdb'):    
        traj_load = md.load(traj, top=topo)
    else:
        traj_load = md.load(traj)
    box_lengths_xyz = traj_load.unitcell_lengths

    toc = time.perf_counter()
    print(f'Trajectory loaded in {toc-tic:0.4f} seconds.\nCalculating MSD...')
    tic = time.perf_counter()
    coords = traj_load.xyz[:,:,:dim] # only look at x and y coordinates
    num_frames = len(coords)
    msd = np.zeros(num_frames)
    disp = np.zeros_like(coords)


    for i in range(num_frames):
        if i > start_frame:
            step_disp = coords[i] - coords[i-1]
            # check if the particle has crossed the periodic boundary
            for j in range(len(step_disp)):
                if step_disp[j][0] > box_lengths_xyz[i][0] / 2:
                    step_disp[j][0] -= box_lengths_xyz[i][0]
                elif step_disp[j][0] < -box_lengths_xyz[i][0] / 2:
                    step_disp[j][0] += box_lengths_xyz[i][0]
                if step_disp[j][1] > box_lengths_xyz[i][1] / 2:
                    step_disp[j][1] -= box_lengths_xyz[i][1]
                elif step_disp[j][1] < -box_lengths_xyz[i][1] / 2:
                    step_disp[j][1] += box_lengths_xyz[i][1]
            disp[i] = step_disp + disp[i-1]
            msd[i] = np.mean(np.square(disp[i]))
    toc = time.perf_counter()
    print(f'MSD calculated in {toc-tic:0.4f} seconds.\n')
    frames = np.arange(num_frames)
    traj_time = frames * frame_duration

    return msd, traj_time


def diffcoeff(msds: np.ndarray, traj_time: np.ndarray, dim: int, num_sub_samples: int, frame_frac_bounds=(0.,1.)):
    """
    Fit the diffusion coefficient from the mean squared displacement.
    Parameters
    ----------
    msds : np.ndarray
        Mean squared displacement of shape (n_frames, n_particles)
    traj_time : np.ndarray
        Time at each frame in femtoseconds of shape (n_frames)
    dim : int
        Dimension of the system
    num_sub_samples : int
        Number of intervals along the trajectory to use for the average diffusion coefficient
    frame_frac_range : tuple
        Fraction of frames to use for the diffusion coefficient fit
    Returns
    -------
    diff_coeff : float
        Diffusion coefficient
    """
    # Fit the diffusion coefficient
    bounded_frames = float(len(msds) * (frame_frac_bounds[1] - frame_frac_bounds[0]))
    # if float(num_sub_samples) > bounded_frames:
    #     raise ValueError('Number of sub-samples must be less than the number of frames in the trajectory.')
    frames_per_sub_sample = int(bounded_frames / num_sub_samples)
    diff_coeff_list = []
    for i in range(num_sub_samples):
        lower_bound = int(frame_frac_bounds[0] * len(msds) + i * frames_per_sub_sample)
        upper_bound = int(frame_frac_bounds[0] * len(msds) + (i+1) * frames_per_sub_sample)
        msds_sub = msds[lower_bound:upper_bound]
        traj_time_sub = traj_time[lower_bound:upper_bound]
        diff_coeff = np.polyfit(traj_time_sub, msds_sub, 1)[0] / (2 * dim)
        diff_coeff_list.append(diff_coeff)
    diff_coeff = np.mean(diff_coeff_list)
    coeff_std_err = np.std(diff_coeff_list) / np.sqrt(num_sub_samples)

    print(f'Diffusion coefficient: {diff_coeff} nm^2/fs')
    print(f'Diffusion coefficient: {diff_coeff * 10 / 1e-5} *10^-5 cm^2/s')
    print(f'Diffusion coefficient list:\n{diff_coeff_list}\n')
    print(f'Standard error: {coeff_std_err}\n\n')

    return diff_coeff, diff_coeff_list, coeff_std_err


def calc_vaf(vels, frame_frac_bounds=(0.,1.)):
    # calculates the 2d velocity autocorrelation function (x,y)
    if isinstance(vels, str):
        vels = np.load(vels)
    print(vels.shape)
    if vels.shape[2] == 3:
        vels = vels[:,:,:2]
    lower_bound = int(frame_frac_bounds[0]*len(vels))
    upper_bound = int(frame_frac_bounds[1]*len(vels))
    vels = vels[lower_bound:upper_bound+1,...]
    vaf = np.zeros((vels.shape[0],))
    for i in range(vels.shape[0]):
        vaf[i] = np.sum(vels[0]*vels[i], axis=(0,1)) / vels.shape[1]
    return vaf