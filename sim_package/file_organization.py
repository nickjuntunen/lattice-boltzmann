import os
from datetime import date


def create_dir(lipid_name: str, size_box: str, lb_status: bool):
        
    '''
    Create trial directories for (LB-)MD simulations.

    Parameters
    ----------
    lipid : str
        Lipid type.
    lb : bool
        Whether or not to implement the lattice boltzmann method.
    size : int
        Size of the system in angstroms.
    periodic_pdb : bool
        Whether or not the pdb file is periodic.
    '''
    if lb_status:
        lb_string = '_lb'
    else:
        lb_string = '_no_lb'
    trial_dir = f'{lipid_name}{size_box}_NVT{lb_string}'

    # create trial directories
    counter = 0
    if not os.path.exists(trial_dir):
        os.mkdir(trial_dir)

    else:
        while os.path.exists(trial_dir):
            counter += 1
            trial_dir = f'{lipid_name}{size_box}_NVT{lb_string}_{counter}'
        os.mkdir(trial_dir)

    print(f'Created trial directory: {trial_dir}')

    return lb_string, trial_dir


def record_notes(args1, time1, md_steps_per_lb1, sim_time1, forcing1, vel1, flow_force1, tracer_idcs1, diff_co1, diff_list1, err1, vaf_diffco1):
    today = str(date.today())
    with open(today+'_Notes.txt','w') as f:
        if args1.lb:
            f.write(f'LB-MD run for {args1.lipid}{args1.size} in NVT ensemble\n\n\n')
            f.write(f'LB-MD run complete in {time1:0.4f} seconds ({time1 / 3600:0.4f} hours)\n')
            f.write(f'Number of MD steps per LB step: {md_steps_per_lb1}\n')
            f.write(f'Total simulated time: {sim_time1} us\n')
            f.write(f'Forcing scheme: {forcing1} et al.\n')
            f.write(f'Initial velocity: {vel1} nm/fs\n')
            f.write(f'Constant flow force: {flow_force1} kg*nm/fs^2\n')
            f.write(f'Tracer head indices: {tracer_idcs1}\n')
            f.write('To track the above indices in Ovito, ions must be removed from the trajectory.\n\n\n')
            f.write(f'MSD diffusion coefficient calculations:\n')
            f.write(f'\tDiffusion coefficient: {diff_co1} nm^2/fs\n')
            f.write(f'\tDiffusion coefficient: {diff_co1 * 10 / 1e-5} *10^-5 cm^2/s\n')
            f.write(f'\tDiffusion coefficient list:\n\t{diff_list1}\n')
            f.write(f'\tStandard error: {err1} nm^2/fs\n\n\n')
            f.write(f'Velocity autocorrelation function calculations:\n')
            f.write(f'\tDiffusion coefficient: {vaf_diffco1} nm^2/fs\n')
            f.write(f'\tDiffusion coefficient: {vaf_diffco1 * 10 / 1e-5} *10^-5 cm^2/s\n')
        else:
            f.write(f'MD-only run for {args1.lipid}{args1.size} in NVT ensemble\n\n\n')
            f.write(f'MD-only run complete in {time1:0.4f} seconds ({time1 / 3600:0.4f} hours)\n')
            f.write(f'Total simulated time: {sim_time1} us\n')
            f.write(f'Tracer head indices: {tracer_idcs1}\n')
            f.write('To track the above indices in Ovito, ions must be removed from the trajectory.\n\n\n')
            f.write(f'MSD diffusion coefficient calculations:\n')
            f.write(f'\tDiffusion coefficient: {diff_co1} nm^2/fs\n')
            f.write(f'\tDiffusion coefficient: {diff_co1 * 10 / 1e-5} *10^-5 cm^2/s\n')
            f.write(f'\tDiffusion coefficient list:\n{diff_list1}\n')
            f.write(f'\tStandard error: {err1} nm^2/fs\n\n\n')
    return