# IMPORTS
import openmm.app as app
import openmm.unit as u
import openmm as mm
import martini_openmm as martini
import mdtraj as md
import numpy as np
import os
import time
import argparse
import matplotlib.pyplot as plt
from datetime import date
import lb_fluid as lb
import sim_package as sim
import lb_analysis_package as lba


# PARSE COMMAND LINE ARGUMENTS
parser = argparse.ArgumentParser(description='Run a simulation')
parser.add_argument('-l','--lipid',type=str,help='lipid type')
parser.add_argument('-s','--size',type=str,help='system size in angstroms')
parser.add_argument('-lb', action='store_true',help='run lattice Boltzmann simulation')
parser.add_argument('-nt','--no_tracer_spring',action='store_true',help='do not include tracer forces')
args = parser.parse_args()
print(args,'\n')


# ORGANIZE FILES
lb_str, trial_dir = sim.create_dir(args.lipid, args.size, args.lb)
sys_str = f'topo_info/{args.lipid}{args.size}_system.top'
pdb_str = f'equilibrations/{args.lipid}{args.size}NVT_equilibration/eq.pdb'


# CREATE SYSTEM
platform = mm.Platform.getPlatformByName('CUDA')
conf = app.PDBFile(pdb_str)
box_vecs = conf.getTopology().getPeriodicBoxVectors().value_in_unit(u.nanometer)*u.nanometer
epsilon_r = 15
top = martini.MartiniTopFile(sys_str,
                             periodicBoxVectors=box_vecs,
                             epsilon_r=epsilon_r)
md_timestep = 10 # femtoseconds
system = top.create_system(nonbonded_cutoff=1.1*u.nanometer)
integrator = mm.LangevinMiddleIntegrator(310 * u.kelvin,
                                         10.0 / u.picosecond,
                                         md_timestep * u.femtosecond)
integrator.setRandomNumberSeed(1)
simulation = app.Simulation(top.topology, system, integrator, platform)
simulation.context.setPositions(conf.getPositions())
simulation.loadState(f'equilibrations/{args.lipid}{args.size}NVT_equilibration/eq.state')


# GET INDICES AND CREATE LB/TRACER FORCES
top_head_idx, top_neck_idx, bot_head_idx, lip_bds = sim.get_indices(pdb_str, args.lipid)
force_unit = u.kilojoules_per_mole / u.nanometer
external = sim.lb_shear(force_unit, top_head_idx, system, simulation)
tr_idcs = 'None'
if not args.no_tracer_spring:
    tr_idcs = sim.tracer_spring(force_unit, system, simulation, top_head_idx, bot_head_idx)


# SIMULATION PARAMETERS
f_dens = 33.4 # molecules/nm^3
m_dict = {'popc':0.760091, 'dopc':0.734039} # kg/mol
m = m_dict[args.lipid]
viscosity = 0.001 # nm^2/fs
vel = (0.00001,0) # nm/fs
flow_force = (1e-8,0) # kg*nm/fs^2

md_steps_per_lb = 10
t_lb_step = md_steps_per_lb * md_timestep # fs
grid_res = 5 # grid resolution (angstroms)
assert int(args.size) % grid_res == 0, 'grid resolution must be a factor of system size'

num_x = int(int(args.size) / grid_res)
num_y = int(int(args.size) / grid_res)
box_vecs = box_vecs.value_in_unit(u.nanometer)
dim = box_vecs[0][0] # nm
dr = dim / num_x # nm
depth = 9
l_grid_shape = (depth, num_x, num_y)

gamma = 0.00025 # fs^-1
sim_time = 1.6 # us
num_lb_steps = int(sim_time * 1e9 / t_lb_step)
forcing = 'he'
vel_intvl = 100000 # number of lb steps between velocity records
t_vel_intvl = vel_intvl * t_lb_step
lip_vels = np.zeros((int(num_lb_steps/vel_intvl),len(top_head_idx),2))


# REPORTERS
os.chdir(trial_dir)
md_per_frame = 100000
reporter_name = f'{args.lipid}{args.size}{lb_str}'
xtc_reporter = md.reporters.XTCReporter(reporter_name+'.xtc', md_per_frame)
pdb_reporter = app.PDBReporter(reporter_name+'.pdb', md_per_frame, enforcePeriodicBox=True)
simulation.reporters.append(app.StateDataReporter(reporter_name+'.log',
                                                  1000,
                                                  step=True,
                                                  totalEnergy=True))
simulation.reporters.append(pdb_reporter)
simulation.reporters.append(xtc_reporter)


# INITIALIZE GRID AND FORCING SCHEME
fl_grid = lb.D2Q9FluidNVT(l_grid_shape, t_lb_step, dr, viscosity, vel, f_dens, flow_force)
og_rho_field = np.copy(fl_grid.rho)
og_u_field = np.copy(fl_grid.u)
if forcing == 'he':
    update_f = fl_grid.update_f_he()
elif forcing == 'guo':
    update_f = fl_grid.update_f_guo()


# RUN LB-MD
print('Running NVT LB-MD...')
print(f'Using {forcing} forcing')
tic = time.perf_counter()
for i in range(num_lb_steps):
    simulation.step(md_steps_per_lb)

    if args.lb:
        # get state records
        state_rec = simulation.context.getState(getPositions=True, getVelocities=True)
        positions = state_rec.getPositions(asNumpy=True).value_in_unit(u.nanometer)
        velocities = state_rec.getVelocities(asNumpy=True).value_in_unit(u.nanometer/u.femtosecond)
        head_pos = sim.fix_pbc(positions[top_head_idx], box_vecs)
        head_vel = velocities[top_head_idx]

        # update fluid grid
        idx, idx_n, dxy = lb.get_lipid_indices(head_pos, dr, dim, num_x)
        lip_force = fl_grid.update_imp(head_vel, idx, idx_n, dxy, gamma)
        fl_grid.update_macro()
        fl_grid.update_feq()
        update_f
        fl_grid.stream()

        # update external force
        x = (lip_force * (u.kilograms*u.nanometer)/(u.mole*u.femtoseconds**2)).value_in_unit(force_unit) # switch from explicit units to force_unit
        f3d = np.zeros((len(lip_force),3))
        f3d[:,0] = x[:,0]
        f3d[:,1] = x[:,1]
        [external.setParticleParameters(0, q[0], q[1]) for q in zip(np.arange(len(lip_force)), f3d)]
        external.updateParametersInContext(simulation.context)

        # record velocities
        if i % vel_intvl == 0:
            toc = time.perf_counter()
            print(f'LB step {i} complete in {toc-tic:0.4f} seconds')
            lip_vels[int(i/vel_intvl),...] = head_vel[:,:2]

simulation.saveState('equi.state')
simulation.saveCheckpoint('equi.chk')
np.save('lip_vels.npy',lip_vels)
toc = time.perf_counter()
time_run = toc - tic

print(f'LB-MD run complete in {time_run:0.4f} seconds\n')
rho_change = np.sum(fl_grid.rho - og_rho_field)
if rho_change != 0:
    print('Change in density field:\n', og_rho_field - fl_grid.rho)


# ANALYZE AND RECORD DATA
msd, traj_time = lba.msd_2d(traj=reporter_name+'.xtc',
                            topo='../topo_info/popc100_xtal.pdb',
                            frame_duration=int(md_per_frame*10),
                            start_frame=0)
plt.plot(traj_time/1e9,msd)
plt.xlabel('Time ($\mu$s)')
plt.ylabel('MSD (${nm}^2$)')
plt.savefig('msd.png')
plt.close()

diff_co, diff_list, err = lba.diffcoeff(msds=msd, traj_time=traj_time, dim=2, num_sub_samples=5, frame_frac_bounds=(0.5,1))
vaf = lba.calc_vaf(vels=lip_vels, frame_frac_bounds=(0.,1.))
plt.plot(range(vaf.shape[0])*t_vel_intvl/1e9,vaf)
plt.xlabel('Time ($\mu$s)')
plt.ylabel('VAF ($nm^2/fs^2$)')
plt.savefig('vaf.png')
plt.close()

sim.record_notes(args, time_run, md_steps_per_lb, sim_time, forcing, vel, flow_force, tr_idcs, diff_co, diff_list, err, vaf)
