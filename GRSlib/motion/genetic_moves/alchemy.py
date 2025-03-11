

def get_comp(atoms,symbols):
    comps = {symbol: 0.0 for symbol in symbols}
    counts = {symbol: 0 for symbol in symbols}
    atsymbols = [atom.symbol for atom in atoms]
    for atsymbol in atsymbols:
        counts[atsymbol] +=1
    for symbol in symbols:
        comps[symbol] = counts[symbol]/len(atoms)
    return comps

def flip_one_atom(atoms,types,endpoint_compositions=False):
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
