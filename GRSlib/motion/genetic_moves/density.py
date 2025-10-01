from ase.io import read,write
from ase import Atoms,Atom
from ase.ga.utilities import closest_distances_generator, CellBounds
from ase.ga.startgenerator import StartGenerator
from ase.data import atomic_numbers, atomic_names, atomic_masses, covalent_radii
from ase.neighborlist import primitive_neighbor_list

# Lowest level functions that can be used to modif structures, inherited class not needed since scoring will happen in
# motion/genetic.py. This collection of functions is mostly to avoid clustter and massive files where more abstract 
# things are happening.
def add_atom(atoms,symbols,tol = 0.5):
    #TODO Currently this is a copy/paste of the old code, needs work.
    blmin = closest_distances_generator(atom_numbers=[atomic_numbers[symbol] for symbol in symbols] + [atomic_numbers['Ne']], ratio_of_covalent_radii=0.5)
    def readd():
        symbol = vnp.random.choice(symbols)
        rnd_pos_scale = vnp.random.rand(1,3)
        rnd_pos = vnp.matmul(atoms.get_cell(),rnd_pos_scale.T)
        rnd_pos = rnd_pos.T[0]
        new_atom = Atom('Ne',rnd_pos)
        tst_atoms = atoms.copy()
        tst_atoms.append(new_atom)
        tst_atoms.wrap()
        rc = 5.
        
        atinds = [atom.index for atom in tst_atoms]
        at_dists = {i:[] for i in atinds}
        all_dists = []
        nl = primitive_neighbor_list('ijdD',pbc=tst_atoms.pbc,positions=tst_atoms.positions ,cell=atoms.get_cell(),cutoff=rc)
        bond_types = {i:[] for i in atinds}
        for i,j in zip(nl[0],nl[-1]):
            at_dists[i].append(j)
        for i,j in zip(nl[0],nl[1]):
            bond_types[i].append( (atomic_numbers[tst_atoms[i].symbol] , atomic_numbers[tst_atoms[j].symbol])  )
        return symbol, tst_atoms, at_dists, rnd_pos, bond_types
    symbol, tst_atoms , at_dists , rnd_pos, bond_types = readd()
    bondtyplst = list(bond_types.keys())
    syms = [tst_atom.symbol for tst_atom in tst_atoms]
    tst_id = syms.index('Ne')
    tst_dists = at_dists[tst_id]
    tst_bonds = bond_types[tst_id]
    conds = all([ vnp.linalg.norm(tst_dist) >=  blmin[(atomic_numbers[symbol] , tst_bonds[i][1])] for i,tst_dist in enumerate(tst_dists)])
    while not conds:
        symbol , tst_atoms, at_dists , rnd_pos, bond_types = readd()
        syms = [tst_atom.symbol for tst_atom in tst_atoms]
        tst_id = syms.index('Ne')
        tst_dists = at_dists[tst_id]
        tst_bonds = bond_types[tst_id]
        #conds = all([ vnp.linalg.norm(tst_dist) >= tol for tst_dist in tst_dists])
        #conds = all([ vnp.linalg.norm(tst_dist) >= blmin[tst_bonds[i]] for i,tst_dist in enumerate(tst_dists)])
        conds = all([ vnp.linalg.norm(tst_dist) >=  blmin[(atomic_numbers[symbol] , tst_bonds[i][1])]-tol for i,tst_dist in enumerate(tst_dists)])
    atoms.append(Atom(symbol,rnd_pos))
    return atoms

def remove_atom(atoms,symbols,tol = 0.5):
    #TODO Currently this is a copy/paste of the old code, needs work.
    blmin = closest_distances_generator(atom_numbers=[atomic_numbers[symbol] for symbol in symbols] + [atomic_numbers['Ne']], ratio_of_covalent_radii=0.5)
    return atoms

def change_cell():
    #TODO Currently this is a copy/paste of the old code, needs work.
    return atoms
