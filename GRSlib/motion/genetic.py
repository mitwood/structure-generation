#placeholder for genetic moves of (xyz) 
import numpy as np
#from ase.build import bulk
#from ase.io import read,write
#from ase.ga.utilities import closest_distances_generator, CellBounds
#from ase.ga.startgenerator import StartGenerator
#from ase import Atoms,Atom
#from ase.data import atomic_numbers

def starting_generation(pop_size,all_species,cell,data_type,nchem,**kwargs):
    pop = []
    if data_type == 'ase':
        volume = np.dot(cell[2],np.cross(cell[0],cell[1]))
        dsize = int(pop_size/nchem)
        sort_specs = sorted(all_species)
        nats = len(all_species)
        uspecs = list(set(sort_specs))
        if len(uspecs) == 1:
            block_sets = [ [(uspecs[0],len(sort_specs))] ]* nchem
        else:
            block_sets = []
            for iii in range(nchem):
                natoms_per_type = generate_random_integers(len(sort_specs), len(uspecs))
                block_sets.append( [ tuple([uspecs[ij],natoms_per_type[ij]]) for ij in range(len(uspecs)) ] )
                
        # Generate a dictionary with the closest allowed interatomic distances
        Zs = [ atomic_numbers[sym] for sym in list(set(all_species))]
        blmin = closest_distances_generator(atom_numbers=Zs,ratio_of_covalent_radii=0.5)

        natoms = len(all_species)

        slab = Atoms('',cell=cell, pbc=True)

        for block in block_sets:
            # Initialize the random structure generator
            #sg = StartGenerator(slab, all_species, blmin, box_volume=volume,
            sg = StartGenerator(slab, block, blmin,number_of_variable_cell_vectors=0) 
            # Generate N random structures and add them to the database
            for i in range(dsize):
                pop.append(sg.get_new_candidate())

    elif typ == 'lattice':
        try:
            parent = kwargs['parent']
            sc = kwargs['s'] # ( 2,2,1 ) or some other tuple that defines supercell
            #atoms = parent*sc
            occ_concs = kwargs['occupation_concentrations'] # {'Mg':0.25,'Ca':0.75} or ['Ca','Mg'] for random sampling
        except KeyError:
            try:
                typ = kwargs['type'] # a SINGLE parent element: e.g. Mg OR Ta OR Ni ...
                lattice = kwargs['lattice'] # 'bcc','fcc',... (just cubic for now to keep it simple)
                a = kwargs['a'] # lattice constant
                #syms = [s for s in atoms.symbols]
                sc = kwargs['s'] # ( 2,2,1 ) or some other tuple that defines supercell
                occ_concs = kwargs['occupation_concentrations']
            except KeyError:
                raise ValueError(" need to supply the type, lattice, lattice constant, and supercell factor to use 'lattice' generation type: example kwargs={ 'type':'Mg', 'lattice':'bcc', 'a':4.25, 's':(2,2,2)}")
            assert lattice in ['bcc','fcc','sc'], "must have cubic lattice in starting_generation=lattice type"
            parent = bulk(typ,lattice,a,cubic=True)
            atoms = parent*sc
            for i in range(pop_size):
                occs = generate_occs(occ_concs, len(atoms))
                print ('occs in initial pop %d' %i, occs)
                newatoms = atoms.copy()
                newatoms.symbols = occs
                pop.append(newatoms)
            
    return pop

def mutation_type_from_prob(choice_probs):
    choices = list(choice_probs.keys())
    probs = list(choice_probs.values())
    mut_typ = np.random.choice(choices,p=probs)
    return mut_typ

def generate_random_integers(sum_value, n):
    # Generate a list of n - 1 random integers
    random_integers = [np.random.randint(0, sum_value) for _ in range(n - 1)]
    # Ensure the sum of the random integers is less than or equal to sum_value
    random_integers.sort()
    # Calculate the Nth integer to ensure the sum is equal to sum_value
    random_integers.append(sum_value - sum(random_integers))
    
    return random_integers

def generate_occs(target, natoms):
    if type(target) == dict:
        counts = {k: round(v * natoms) for k, v in target.items()}
        diff = natoms - sum(counts.values())
        while diff != 0:
            for k in counts:
                if diff > 0:
                    counts[k] += 1
                    diff -= 1
                elif diff < 0 and counts[k] > 0:
                    counts[k] -= 1
                    diff += 1
        occs = [k for k, v in counts.items() for _ in range(v)]
        np.random.shuffle(occs)
    elif type(target) == list:
        occs = np.random.choice(target,natoms).tolist()
    return occs

