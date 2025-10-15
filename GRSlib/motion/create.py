#from GRSlib.parallel_tools import ParallelTools
from GRSlib.motion.create_helper.ase_tools import ASETools
from ase import Atoms,Atom
from ase.build import bulk
import numpy as np
import random

# Parent class for creating new structures using ASE tools. This does not require a scoring function, unlike the other
# parent classes (Gradient, Genetic). Returns dictionaries of 

#TODO need to make heavy use of converting between ASE and LAMMPS-Data objects

class Create:

    def __init__(self, pt, config, convert):
        self.pt = pt #ParallelTools()
        self.config = config #Config()
        self.convert = convert #Convert()
        #pop_size,all_species,cell,data_type,nchem
        # Any number of variables to create structures can be stashed here in self. instead of 
        # passing them into the individual functions.

    def starting_generation(self, *args):
        #More of a super function that will call a bunch of the ones below
        start_type = "from_"+self.config.sections["GENETIC"].start_type
        fn_start = getattr(Create, start_type) 
        population = fn_start(self,args) #Not a big fan of this method, feels clunky to switch start points.

        #TODO Currently this is a copy/paste of the old code, needs work.
        """
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
                    population.append(sg.get_new_candidate())

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
                    population.append(newatoms)
        """                
        return population

    #Starting point types:
    def from_template(self,*args):
        #More of a super function that will call a bunch of the ones below
#        print("Starting population using provided template")
        population = []
        duplicate = self.convert.lammps_to_ase(args[0][0])
        for candidate in range(self.config.sections["GENETIC"].population_size):
            population.append(duplicate)
        return population
    
    def from_lattice(self,*args):
        #More of a super function that will call a bunch of the ones below
#        print("Starting population using provided lattice type")
        population = []
        return population
    
    def from_random(self,*args):
        #More of a super function that will call a bunch of the ones below
        #print("Starting population of random low energy structures of provided elements")
        population = []
        # From types, find cell
        num_ele = len(self.config.sections["BASIS"].elements)
        for item in range(num_ele):
            trial_struct = bulk(self.config.sections["BASIS"].elements[item]) #ase.build.bulk, provides a guess at cell
            prim_trial = ASETools.get_primitive_cell(trial_struct)
            supercell = ASETools.get_cube_supercell(prim_trial, self.config.sections["GENETIC"].min_atoms, 
                                               self.config.sections["GENETIC"].max_atoms)
            population.append(supercell) 
            #There is now a supercell of size between min_atoms and max_atoms of the minimum energy structures of each element
        for item in range(num_ele):
            for remaining in range(int(round(int(self.config.sections["GENETIC"].population_size) - num_ele)/num_ele)):
                new_candidate = ASETools.get_random_pos(population[item], self.config.sections["GENETIC"].min_atoms,
                                                        self.config.sections["GENETIC"].max_atoms, self.config.sections["BASIS"].elements[item])
                population.append(new_candidate)
        return population

