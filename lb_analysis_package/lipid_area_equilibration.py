import mdtraj.formats as mdformats
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def calc_per_lipid_area(traj: str, lipids_per_leaf: int, traj_range=[0.0,1.0], plot=True):

    '''
    Reads XTC trajectory file to calculate the average surface area per lipid molecule.
    Good test for equilibration in the NPT ensemble (surface area stable over time).

    Parameters
    ----------
    traj : str
        Path to XTC trajectory file.
    lipids_per_leaf : int
        Number of lipids per leaflet.
    range : list, optional
        Range of frames to analyze. Default is (0.0,1.0).
    plot : bool, optional
        Whether to plot the surface area per lipid over time. Default is True.
    '''
    
    with mdformats.XTCTrajectoryFile(traj) as f:
        xyz, time, step, box = f.read()
    perlip_area = np.zeros((box.shape[0],1))

    for i in range(len(perlip_area)):
        perlip_area[i] = box[i][0][0] * box[i][1][1] / lipids_per_leaf
    
    traj_range[0] = int(traj_range[0] * len(perlip_area))
    traj_range[1] = int(traj_range[1] * len(perlip_area))

    if plot == True:
        fig = plt.figure()
        plt.plot((time/1000)[traj_range[0]:traj_range[1]], perlip_area[traj_range[0]:traj_range[1]])
        plt.xlabel('Time /ns')
        plt.ylabel('Area per lipid / nm$^2$')
        plt.title('Equilibration test: stable surface area per lipid')
        plt.savefig('perlip_area.png')

    return #perlip_area, box.shape[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run LB-MD simulation.')
    parser.add_argument('-l','--lipid', type=str, help='lipid type')
    parser.add_argument('-s','--size', type=str, help='size of the system \
                        x/y dimensions in angstroms (square boxes)')
    parser.add_argument('-e','--ensemble', type=str, help='ensemble in which \
                        to run the simulation', choices=['NVT','NPT','nvt','npt'], default='NVT')
    args = parser.parse_args()
    print(args,'\n')

    lipid_amts = {'100':147, '250':916, '400':2343}
    os.chdir(f'{args.lipid}{args.size}{args.ensemble}_equilibration')
    calc_per_lipid_area('equilibration.xtc', lipid_amts[args.size], traj_range=[0,1], plot=True)