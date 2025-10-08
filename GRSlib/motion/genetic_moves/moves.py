from ase.io import read,write
from ase import Atoms,Atom
from ase.ga.utilities import closest_distances_generator, CellBounds
from ase.ga.startgenerator import StartGenerator
from ase.data import atomic_numbers, atomic_names, atomic_masses, covalent_radii
from ase.neighborlist import primitive_neighbor_list

# Lowest level functions that can be used to modif structures, inherited class not needed since scoring will happen in
# motion/genetic.py. This collection of functions is mostly to avoid clustter and massive files where more abstract 
# things are happening.

class GenMoves():
    
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

    def perturb_one_atom(atoms,scale=0.5,max_attempt=100,apply_to='ase'):
        #TODO Currently this is a copy/paste of the old code, needs work.
        if apply_to == 'ase':
            cutoffs = ase.neighborlist.natural_cutoffs(atoms)
            sym_num_map = {sym:atomic_numbers[sym] for sym in atoms.symbols}
            #nl = ase.neighborlist.neighbor_list('ijd', atoms, max(cutoffs))
            nl = ase.neighborlist.neighbor_list('ijd', atoms, cutoffs)
            #nl.update(atoms)
            #indices, offsets = nl.get_neighbors(0)`
            Zs = [ atomic_numbers[sym] for sym in list(set(atoms.symbols))]
            blmin = closest_distances_generator(atom_numbers=Zs,
                                                ratio_of_covalent_radii=0.5)
            good_pert = False
            nattempt =0
            new_atoms = atoms.copy()
            while not good_pert and nattempt < max_attempt:
                new_atoms = atoms.copy()
                pert_ind = np.random.randint(0,len(atoms))
                perturbation = np.random.rand(1,3)[0]
                posneg = 2.*(perturbation - np.min(perturbation))/np.ptp(perturbation)-1
                posneg *= scale
                new_atoms[pert_ind].x += posneg[0]
                new_atoms[pert_ind].y += posneg[1]
                new_atoms[pert_ind].z += posneg[2]
                nl = ase.neighborlist.neighbor_list('ijd', new_atoms, cutoffs)
                sym_num_map = {sym:atomic_numbers[sym] for sym in new_atoms.symbols}
                pair_dist_flags = [ blmin[tuple(sorted([sym_num_map[new_atoms.symbols[i]], sym_num_map[ new_atoms.symbols[nl[1][i_ind]]] ]))] > nl[2][i_ind] for i_ind, i in enumerate(nl[0])]
                if not any(pair_dist_flags):
                    good_pert = True
                nattempt += 1
            return new_atoms

            """
            new_atoms = atoms.copy()
            pert_ind = np.random.randint(0,len(atoms))
            perturbation = np.random.rand(1,3)[0]
            posneg = 2.*(perturbation - np.min(perturbation))/np.ptp(perturbation)-1
            posneg *= scale
            new_atoms[pert_ind].x += posneg[0]
            new_atoms[pert_ind].y += posneg[1]
            new_atoms[pert_ind].z += posneg[2]
            return new_atoms
            """

        elif apply_to == 'raw_positions':
            new_atoms = atoms.copy()
            pert_ind = np.random.randint(0,len(atoms))
            perturbation = np.random.rand(1,3)[0]
            posneg = 2.*(perturbation - np.min(perturbation))/np.ptp(perturbation)-1
            posneg *= scale
            new_atoms[pert_ind][0] += posneg[0]
            new_atoms[pert_ind][1] += posneg[1]
            new_atoms[pert_ind][2] += posneg[2]
            return new_atoms

    def perturb_N_atoms(atoms,scale=0.5,max_attempt = 100, fraction=0.25):
        #TODO Currently this is a copy/paste of the old code, needs work.
        cutoffs = ase.neighborlist.natural_cutoffs(atoms)
        sym_num_map = {sym:atomic_numbers[sym] for sym in atoms.symbols}
        #nl = ase.neighborlist.neighbor_list('ijd', atoms, max(cutoffs))
        nl = ase.neighborlist.neighbor_list('ijd', atoms, cutoffs)
        #nl.update(atoms)
        #indices, offsets = nl.get_neighbors(0)`
        Zs = [ atomic_numbers[sym] for sym in list(set(atoms.symbols))]
        blmin = closest_distances_generator(atom_numbers=Zs,
                                            ratio_of_covalent_radii=0.5)

        good_pert = False
        nattempt =0
        new_atoms = atoms.copy()
        while not good_pert and nattempt < max_attempt:
            pert_inds = np.random.choice(range(len(atoms)),size=int(len(atoms)*fraction) )
            for pert_ind in pert_inds:
                new_atoms = atoms.copy()
                perturbation = np.random.rand(1,3)[0]
                posneg = 2.*(perturbation - np.min(perturbation))/np.ptp(perturbation)-1
                posneg *= scale
                new_atoms[pert_ind].x += posneg[0]
                new_atoms[pert_ind].y += posneg[1]
                new_atoms[pert_ind].z += posneg[2]
            nl = ase.neighborlist.neighbor_list('ijd', new_atoms, cutoffs)
            sym_num_map = {sym:atomic_numbers[sym] for sym in new_atoms.symbols}
            pair_dist_flags = [ blmin[tuple(sorted([sym_num_map[new_atoms.symbols[i]], sym_num_map[ new_atoms.symbols[nl[1][i_ind]]] ]))] > nl[2][i_ind] for i_ind, i in enumerate(nl[0])]
            if not any(pair_dist_flags):
                good_pert = True
            nattempt += 1
        if not good_pert:
            print ("WARNING: this mutation has a bad neighbor distance - try changing your scale parameter")
        return new_atoms

    def flip_one_atom(atoms,types,endpoint_compositions=False):
        #TODO Currently this is a copy/paste of the old code, needs work.
        if endpoint_compositions:
            new_atoms = atoms.copy()
            flip_ind = np.random.randint(0,len(atoms))
            flip_current = new_atoms[flip_ind].symbol
            excluded = [typ for typ in types if typ != flip_current]
            flip_to_ind = np.random.randint(0,len(excluded))
            flip_to_type = excluded[flip_to_ind]
            new_atoms[flip_ind].symbol = flip_to_type
        else:
            itr = 0
            comps_dct = get_comp(atoms,types)
            comp_vals = list(comps_dct.values())
            while itr == 0 or any([icomp == 0.0 for icomp in comp_vals]):
                new_atoms = atoms.copy()
                flip_ind = np.random.randint(0,len(atoms))
                flip_current = new_atoms[flip_ind].symbol
                excluded = [typ for typ in types if typ != flip_current]
                flip_to_ind = np.random.randint(0,len(excluded))
                flip_to_type = excluded[flip_to_ind]
                new_atoms[flip_ind].symbol = flip_to_type
                comps_dct = get_comp(new_atoms,types)
                comp_vals = list(comps_dct.values())
                itr += 1
        return new_atoms

    def flip_N_atoms(atoms,types,fraction=None,endpoint_compositions=False):
        #TODO Currently this is a copy/paste of the old code, needs work.
        if endpoint_compositions:
            fraction = np.random.rand()
            pert_inds = np.random.choice(range(len(atoms)),size=int(len(atoms)*fraction) )
            new_atoms = atoms.copy()
            for pert_ind in pert_inds:
                flip_ind = np.random.randint(0,len(atoms))
                flip_current = new_atoms[flip_ind].symbol
                excluded = [typ for typ in types if typ != flip_current]
                flip_to_ind = np.random.randint(0,len(excluded))
                flip_to_type = excluded[flip_to_ind]
                new_atoms[flip_ind].symbol = flip_to_type
        else:
            comps_dct = get_comp(atoms,types)
            comp_vals = list(comps_dct.values())
            itr = 0
            while itr == 0 or any([icomp == 0.0 for icomp in comp_vals]):
                fraction = np.random.rand()
                pert_inds = np.random.choice(range(len(atoms)),size=int(len(atoms)*fraction) )
                new_atoms = atoms.copy()
                for pert_ind in pert_inds:
                    flip_ind = np.random.randint(0,len(atoms))
                    flip_current = new_atoms[flip_ind].symbol
                    excluded = [typ for typ in types if typ != flip_current]
                    flip_to_ind = np.random.randint(0,len(excluded))
                    flip_to_type = excluded[flip_to_ind]
                    new_atoms[flip_ind].symbol = flip_to_type
                comps_dct = get_comp(new_atoms,types)
                comp_vals = list(comps_dct.values())
                itr += 1

        return new_atoms
