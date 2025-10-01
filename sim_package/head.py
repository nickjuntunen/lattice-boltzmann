import numpy as np
import mdtraj as md

def get_indices(pdb_str1, lip):
    head = 'NC3'
    neck = 'PO4'
    tail = 'C5B'

    file = md.load(pdb_str1, lip)
    topo = file.topology
    coords = file.xyz[0]
    table, bonds = topo.to_dataframe()
    head_idx = table[table['name']==head].index.values
    neck_idx = table[table['name']==neck].index.values
    tail_idx = table[table['name']==tail].index.values
    top_head_idx = np.copy(head_idx)
    top_neck_idx = np.copy(neck_idx)

    del_idx = []
    for i in range(len(head_idx)):
        if coords[tail_idx[i]][2] > coords[head_idx[i]][2]:
            del_idx.append(i)

    top_head_idx = np.delete(top_head_idx,del_idx)
    top_neck_idx = np.delete(top_neck_idx,del_idx)
    bot_head_idx = head_idx[del_idx]

    lipid_bead_dict = {'popc':13}
    if lip not in lipid_bead_dict.keys():
        raise ValueError('Lipid type not recognized. Please add to lipid_bead_dict in head.py')

    lip_bds = lipid_bead_dict[lip]

    return top_head_idx, top_neck_idx, bot_head_idx, lip_bds