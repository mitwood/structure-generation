import numpy as np
#from ase.build import bulk
#from ase.io import read,write
#from ase.ga.utilities import closest_distances_generator, CellBounds
#from ase.ga.startgenerator import StartGenerator
from ase import Atoms,Atom
from GRSlib.motion.create_helper.ase_tools import ASETools
from GRSlib.motion.genetic_moves.moves import GenMoves
#from ase.data import atomic_numbers
import random

class Genetic:
#TODO need to make heavy use of converting between ASE and LAMMPS-Data objects
#TODO each function call here needs to end with scoring call

    def __init__(self, pt, config, scoring):
        self.pt = pt #ParallelTools()
        self.config = config #Config()
        self.scoring = scoring

    def crossover(self, parent1, parent2):
        #TODO Currently this is a copy/paste of the old code, needs work.

        # 1) Takes in a pair of structures, tourny selection should give the most recent winner
        #    and a (random?) second structure from the winners circle
        # 2) Crossover will be the 'merger' of these two structures, which for now is the cell from last winner and 
        #    a spliced together set of atoms based on some random dividing line in the atom ids. 
        # 3) Check the spliced cell for accuracy in chemical composition, flip_atoms until close to desired
        # 4) Now generate the remaining population_size - 1  structures as perturbations of the spliced cell. 
        # 5) Return the population 
        parent1_natom = len(parent1.numbers)
        parent2_natom = len(parent2.numbers)
        cross_point = np.random.randint(1, parent1_natom-1)
        if (parent1_natom - cross_point) > parent2_natom:
            cross_point = parent1_natom-parent2_natom
            child1 = parent1[:cross_point] + parent2[parent2_natom:]
            child2 = parent1[cross_point:] + parent2[parent2_natom:]
        else:
            child1 = parent1[:cross_point] + parent2[cross_point:]
            child2 = parent2[:cross_point] + parent1[cross_point:]

        if self.config.sections["GENETIC"].composition is not None:
            #There was a constraint on the allowed compositions, get comps ans see if it is off
            #by more than one atom swap worth.
            comps_child1 = self.get_comp(child1,self.config.sections["BASIS"].elements)
            comps_child2 = self.get_comp(child2,self.config.sections["BASIS"].elements)
            for ele_types in comps_child1:
                pass
        return parent1
    """
    if target_comps or types==None:
        assert len(parent1) == len(parent2), "parents must have the same length"
        #TODO Need a way to propose another parent, or force a choice of size
        #Call back to tourny selection and get a new candidate
        psize = len(parent1)
        if inputseed != None:
            np.random.seed(inputseed)
        cross_point = np.random.randint(1, psize-1)
        child1 = parent1[:cross_point] + parent2[cross_point:]
        child2 = parent2[:cross_point] + parent1[cross_point:]
    else:
        assert len(parent1) == len(parent2), "parents must have the same length"
        #TODO Need a way to propose another parent, or force a choice of size
        #Call back to tourny selection and get a new candidate
        comps_dct1 = self.get_comp(parent1,types)
        comp_vals1 = list(comps_dct1.values())
        comps_dct2 = self.get_comp(parent2,types)
        comp_vals2 = list(comps_dct2.values())
        while itr == 0 or any([icomp == 0.0 for icomp in comp_vals1]) or any([icomp == 0.0 for icomp in comp_vals2]):
            psize = len(parent1)
            if inputseed != None:
                np.random.seed(inputseed)
            cross_point = np.random.randint(1, psize-1)
            child1 = parent1[:cross_point] + parent2[cross_point:]
            child2 = parent2[:cross_point] + parent1[cross_point:]
            comps_dct1 = self.get_comp(child1,types)
            comp_vals1 = list(comps_dct1.values())
            comps_dct2 = self.get_comp(child2,types)
            comp_vals2 = list(comps_dct2.values())
            itr += 1
    return [child1, child2]
    """
    def mutation(self, parent):
        #TODO Currently this is a copy/paste of the old code, needs work.
        #TODO from config, find the set of choices, roll dice and then call down to helper functions in genetic_moves
        # 1) Takes in a structures, tourny selection should give the most recent winner
        # 2) Generate the remaining population_size - 1  structures as perturbations of the given cell. 
        # 3) Return the population 
        
        chosen_mutation  = 'add_atom'
        mutation_options = self.config.sections["GENETIC"].mutation_types
        print(mutation_options)
        print(random.choices(list(mutation_options.keys()), weights=mutation_options.values(), k=self.config.sections["GENETIC"].population_size))
        mutation_array = random.choices(list(mutation_options.keys()), weights=mutation_options.values(), k=self.config.sections["GENETIC"].population_size)
        for mutation in mutation_array:
            event = getattr(Genetic, chosen_mutation)
            event(parent, self.pt, self.config, self.scoring)


        return parent
        """
        if 'flip' in mutation_type:
            return mutation_types[mutation_type](current_atoms,types)
        else:
            return mutation_types[mutation_type](current_atoms,scale)
        
        """

    def get_comp(self, atoms, symbols):
        comps = {symbol: 0.0 for symbol in symbols}
        counts = {symbol: 0 for symbol in symbols}
        atsymbols = [atom.symbol for atom in atoms]
        for atsymbol in atsymbols:
            counts[atsymbol] +=1
        for symbol in symbols:
            comps[symbol] = counts[symbol]/len(atoms)
        return comps


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

