# lattice-boltzmann

Implementation of the lattice Boltzmann method for coupling implicit solvent dynamics with a lipid bilayer modeled using Dry Martini for OpenMM.

Description of directories and files:

  topo_info: contains necessary membrane and Dry Martini topology files. Needs additions if new lipid type is being used.
  
  lb_analysis_package: functions to analyze the system (surface area per lipid, order parameter, etc.).
  
  equilibration.py: takes pre-made lipid bilayers and equilibrates them under the NPT-ensemble. Provide lipid type, membrane size, ensemble type, and number of equilibration steps.

  lb_fluid: holds fluid lattice object and associated methods for LB simulation

  oop_lbmd_run.py: object-oriented simulation wrapper

  lb_sim_package: file organization and periodic boundary condition helper functions

  Outdated: keep until sure the new method works\\
    lbmd_run.py (moved away from oop)
    lb_sim_package- fluid_grid_update.py and fluid_grid_init.py (moved into oop code)
