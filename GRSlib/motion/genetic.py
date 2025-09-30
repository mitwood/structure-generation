import numpy as np
#from ase.build import bulk
#from ase.io import read,write
#from ase.ga.utilities import closest_distances_generator, CellBounds
#from ase.ga.startgenerator import StartGenerator
#from ase import Atoms,Atom
#from ase.data import atomic_numbers

class Genetic:

    def __init__(self, pt, config):
        self.pt = pt #ParallelTools()
        self.config = config #Config()
    
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

