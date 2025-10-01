import openmm.app as app
import openmm as mm
import openmm.unit as u
import os
import shutil
import martini_openmm as martini
import mdtraj as md
from mdtraj.reporters import XTCReporter
import argparse
from lb_analysis_package import calc_per_lipid_area
import sim_package as sim

### ARGPARSER ###
parser = argparse.ArgumentParser(description='Run LB-MD simulation.')
parser.add_argument('-l','--lipid', type=str, help='lipid type')
parser.add_argument('-s','--size', type=int, help='size of the system x/y dimension in angstroms')
parser.add_argument('-n','--num_steps', type=int, help='number of steps to run the simulation for')
parser.add_argument('-e','--ensemble', type=str, help='ensemble in which to run the simulation',
                    choices=['NVT','NPT','nvt','npt'], default='NVT')
args = parser.parse_args()
print(args,'\n')
lipid = args.lipid
size = args.size
num_steps = args.num_steps
ensemble = args.ensemble
path_name = f'{lipid}{size}{ensemble}_equilibration'
os.chdir('equilibrations')

# make equilibration directory
counter = 0
if not os.path.exists(path_name):
    os.mkdir(path_name)
else:
    while os.path.exists(path_name):
        counter += 1
        path_name = f'{lipid}{size}{ensemble}_equilibration_{counter}'
    os.mkdir(path_name)

# get number of lipids per leaflet
head_string = 'NC3'
file = md.load(f'../topo_info/{lipid}{size}_xtal.pdb')
topo = file.topology
table, bonds = topo.to_dataframe()
num_lip_per_leaf = int(len(table[table['name'] == head_string].index.values) / 2)

# setup the system
platform = mm.Platform.getPlatformByName('CUDA')
conf = app.PDBFile(f'../topo_info/{lipid}{size}_xtal.pdb')
box_vectors = conf.getTopology().getPeriodicBoxVectors()
epsilon_r = 15
top = martini.MartiniTopFile(
        f"../topo_info/{lipid}{size}_system.top",
        periodicBoxVectors=box_vectors,
        epsilon_r=epsilon_r)
system = top.create_system(nonbonded_cutoff=1.1 * u.nanometer)

integrator = mm.LangevinMiddleIntegrator(310 * u.kelvin,
                                10.0 / u.picosecond,
                                20 * u.femtosecond)
integrator.setRandomNumberSeed(1)

simulation = app.Simulation(top.topology,
                            system,
                            integrator,
                            platform)
simulation.context.setPositions(conf.getPositions())
simulation.context.reinitialize(True)

num_nvt_steps = num_steps
simulation.context.getState(getPositions=True, enforcePeriodicBox=True)


############ MINIMIZATION ############
print("Minimizing energy...")
simulation.minimizeEnergy(maxIterations=100000,tolerance=0.000001)


############ NVT EQUILIBRATION ############
print('Running NVT equilibration...')
simulation.context.setVelocitiesToTemperature(310*u.kelvin, 1)
os.chdir(path_name)

xtc_reporter = XTCReporter('equilibration.xtc',5000)
pdb_reporter = app.PDBReporter('equilibration.pdb',5000)
simulation.reporters.append(app.StateDataReporter("equilibration.log",
                                                  5000,
                                                  step=True,
                                                  totalEnergy=True))
simulation.reporters.append(xtc_reporter)
simulation.reporters.append(pdb_reporter)
simulation.step(num_nvt_steps)

state_rec = simulation.context.getState(getPositions=True)
posi = state_rec.getPositions(asNumpy=True).value_in_unit(u.nanometer)
box_vecs = state_rec.getPeriodicBoxVectors(asNumpy=True).value_in_unit(u.nanometer)
with open('eq.pdb', 'w') as f:
    app.PDBFile.writeFile(simulation.topology, simulation.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(), f)
simulation.context.setPositions(posi)
simulation.saveState('eq.state')

# calculate and plot area per lipid
calc_per_lipid_area(traj=f'equilibration.xtc',
			  lipids_per_leaf=num_lip_per_leaf,
              traj_range=[0,1.0], plot=True)

os.chdir('..')