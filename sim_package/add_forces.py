import openmm as mm
import openmm.unit as u
import numpy as np


def tracer_spring(force_unit1, system1, simulation1, top_head_idx1 = None, bot_head_idx1 = None):
    '''
    Adds a weak harmonic spring force to tracer lipids.
    __________
    Input:
        force_unit1: unit of force
        system1: OpenMM system object
        simulation1: OpenMM simulation object
        top_head_idx1: list of indices of top leaflet head groups
        bot_head_idx1: list of indices of bottom leaflet head groups
    '''

    set_a = None
    set_b = None

    if top_head_idx1 is not None:
        mols_top = np.random.choice(top_head_idx1, 2, replace=False)
        spring1 = mm.CustomNonbondedForce('0.5*k*(r-r0)^2; k=k1*k2; r0=r01+r02')
        spring1.addPerParticleParameter('k')
        spring1.addPerParticleParameter('r0')
        set_a = set(())
        set_b = set(())

        for resi in simulation1.topology.residues():
            for at in resi.atoms():
                spring1.addParticle([0,0.1])

        for resi in simulation1.topology.residues():
            for at in resi.atoms():
                if at.index == mols_top[0]:
                    for at_f in resi.atoms():
                        set_a.add(at_f.index)
                        spring1.setParticleParameters(at.index, [1, 0.05])
                elif at.index == mols_top[1]:
                    for at_f in resi.atoms():
                        set_b.add(at_f.index)
                        spring1.setParticleParameters(at.index, [1, 0.05])

        spring1.addInteractionGroup(set_a, set_b)
        system1.addForce(spring1)
    else:
        print('No spring force added to top leaflet tracer particles.')

    if bot_head_idx1 is not None:
        mols_bot = np.random.choice(bot_head_idx1, 2, replace=False)
        spring2 = mm.CustomNonbondedForce('0.5*k*(r-r0)^2; k=k1*k2; r0=r01+r02')
        spring2.addPerParticleParameter('k')
        spring2.addPerParticleParameter('r0')
        set_a = set(())
        set_b = set(())

        for resi in simulation1.topology.residues():
            for at in resi.atoms():
                spring2.addParticle([0,0.1])

        for resi in simulation1.topology.residues():
            for at in resi.atoms():
                if at.index == mols_bot[0]:
                    for at_f in resi.atoms():
                        set_a.add(at_f.index)
                        spring2.setParticleParameters(at.index, [1, 0.05])
                elif at.index == mols_bot[1]:
                    for at_f in resi.atoms():
                        set_b.add(at_f.index)
                        spring2.setParticleParameters(at.index, [1, 0.05])

        spring2.addInteractionGroup(set_a, set_b)
        system1.addForce(spring2)
    else:
        print('No spring force added to bottom leaflet tracer particles.')

    simulation1.context.reinitialize(True)
    print(mols_top)
    print(mols_bot)
    return np.concatenate((mols_top, mols_bot))


def lb_shear(force_unit1, head_idcs, system1, simulation1):
    '''
    Creates a custom external force that acts on the head groups of the lipids.
    Use this to implement Lattice-Boltzmann shear.
    __________
    Input:
        head_idcs: list of indices of the head groups
        system1: OpenMM system object
        simulation1: OpenMM simulation object
    Output:
        external1: OpenMM custom external force object
    '''
    force_unit = u.kilojoules_per_mole / u.nanometer
    external1 = mm.CustomExternalForce('-forcex*x-forcey*y-forcez*z')
    external1.addPerParticleParameter('forcex')
    external1.addPerParticleParameter('forcey')
    external1.addPerParticleParameter('forcez')

    for i in range(len(head_idcs)):
        external1.addParticle(head_idcs[i], (0,0,0)*force_unit)

    system1.addForce(external1)
    simulation1.context.reinitialize(True)

    return external1 