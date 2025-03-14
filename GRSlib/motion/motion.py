from GRSlib.parallel_tools import ParallelTools
from GRSlib.motion.scoring import Scoring
import numpy as np
import random

#Scoring has to be a class within motion because we want a consistent reference for scores, ans this
#refrence will be LAMMPS using a constructed potential energy surface from the representation loss function

class Gradient:

    def __init__(self, data, current_desc, target_desc, pt, config):
        self.pt = pt #ParallelTools()
        self.config = config #Config()
        #Bring in the target and current descriptors here, will be with self. then
        #descriptors_flt = descriptors.flatten()
        self.current_desc = current_desc
        self.target_desc = target_desc
        self.data = data
        self.n_elements = self.config.sections['BASIS'].numtypes
        if self.n_elements > 1:
            current_desc = current_desc.flatten()
            target_desc = target_desc.flatten()

    def fire_min(self):
        #Will construct a set of additional commands to send to LAMMPS before scoring
        add_cmds=\
        """min_style  fire
        min_modify integrator eulerexplicit tmax 10.0 tmin 0.0 delaystep 5 dtgrow 1.1 dtshrink 0.5 alpha0 0.1 alphashrink 0.99 vdfmax 100000 halfstepback no initialdelay no
        minimize 1e-6 1e-6 %s %s""" % (self.config.sections['MOTION'].nsteps, self.config.sections['MOTION'].nsteps)

        before_score, after_score = Scoring.add_cmds_before_score(add_cmds)
        return before_score, after_score

    def line_min(self):
        #Will construct a set of additional commands to send to LAMMPS before scoring
        add_cmds=\
        """min_style  cg
        min_modify dmax 0.05 line quadratic
        minimize 1e-6 1e-6 %s %s""" % (self.config.sections['MOTION'].nsteps, self.config.sections['MOTION'].nsteps)

        before_score, after_score = Scoring.add_cmds_before_score(add_cmds)
        return before_score, after_score

    def box_min(self):
        #Will construct a set of additional commands to send to LAMMPS before scoring
        add_cmds=\
        """min_style  cg
        min_modify dmax 0.05 line quadratic
        fix box all box/relax iso 0.0 vmax 0.001
        minimize 1e-6 1e-6 %s %s""" % (self.config.sections['MOTION'].nsteps, self.config.sections['MOTION'].nsteps)

        before_score, after_score = Scoring.add_cmds_before_score(add_cmds)
        return before_score, after_score

    def run_then_min(self):
        #Will construct a set of additional commands to send to LAMMPS before scoring
        add_cmds=\
        """velocity all create %s 4928459 dist gaussian
        fix nve all nve
        fix lan all langevin %s %s 1.0 48279
        run %s
        unfix nve
        unfix lan
        min_style  fire
        min_modify integrator eulerexplicit tmax 10.0 tmin 0.0 delaystep 5 dtgrow 1.1 dtshrink 0.5 alpha0 0.1 alphashrink 0.99 vdfmax 100000 halfstepback no initialdelay no
        minimize 1e-6 1e-6 %s %s""" % (self.config.sections['MOTION'].temperature, self.config.sections['MOTION'].temperature, 
                                       self.config.sections['MOTION'].temperature, self.config.sections['MOTION'].nsteps, 
                                       self.config.sections['MOTION'].nsteps, self.config.sections['MOTION'].nsteps, self.config.sections['MOTION'].nsteps)

        before_score, after_score = Scoring.add_cmds_before_score(add_cmds)
        return before_score, after_score

class Genetic:

    def __init__(self, pt, config, **kwargs):
        self.pt = pt #ParallelTools()
        self.config = config #Config()

    def crossover(parent1, parent2, **kwargs):
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

    def mutation(self, **kwargs):
        # TODO from config, find the set of choices, roll dice and then call down to density/perturbation/alchemy
        chosen_mutation  = 'add_atom'
        (chosen_mutation)()


        if 'flip' in mutation_type:
            return mutation_types[mutation_type](current_atoms,types)
        else:
            return mutation_types[mutation_type](current_atoms,scale)
        

        #This def can be cleaner, with a func call dependent on choice
        #if '' == add_atom:
        #    density.add_atom()
        #if '' == del_atom:
        #    density.remove_atom()
        #if '' == change_cell:
        #    density.change_cell()
        # ...

    def tournament_selection(self, **kwargs):
        scores = []
        for candidate in len(population):
            scores.append(Scoring.get_score(population[candidate]))
        selection = population.copy()

        # Pick 2 indicies to compare and add the best of
        # to the selection list (e.g. perform a tournament)
        #Allow for a scoring anomoly at some low rate? else return min(scores)?
        for round in len(selection)-1:
            compare_pair = np.random.randint(0, len(selection), 2)
            if scores[compare_pair[0]] <= scores[compare_pair[1]]:
                loser = compare_pair[1]
                selection.pop(loser)
            else:
                loser = compare_pair[0]
                selection.pop(loser)
        if np.random.rand() < self.config.sections['GENETIC'].mutation_rate:
            nextgen_selection = self.mutation(selection) #Will mutation only take in one structure?
        else:
            nextgen_selection = self.crossover(selection) #Should have two structures
        return nextgen_selection

    def unique_tournament_selection(self, **kwargs):
        #This should be the default since we dont want to send duplicates the crossover/mutation
        scores = []
        for candidate in len(population):
            scores.append(Scoring.get_score(population[candidate]))
        selection = []
        for candidate in population:
            if candidate not in selection:
                selection.append(candidate)
        # The structures to chose from should now only contain unique candidates, determined by score        

        # Pick 2 indicies to compare and add the best of
        # to the selection list (e.g. perform a tournament)
        #Allow for a scoring anomoly at some low rate? else return min(scores)?
        for round in len(selection)-1:
            compare_pair = np.random.randint(0, len(selection), 2)
            if scores[compare_pair[0]] <= scores[compare_pair[1]]:
                loser = compare_pair[1]
                selection.pop(loser)
            else:
                loser = compare_pair[0]
                selection.pop(loser)
        if np.random.rand() < self.config.sections['GENETIC'].mutation_rate:
            nextgen_selection = self.mutation(selection) #Will mutation only take in one structure?
        else:
            nextgen_selection = self.crossover(selection) #Should have two structures
        return nextgen_selection

    def get_comp(self, atoms, symbols):
        comps = {symbol: 0.0 for symbol in symbols}
        counts = {symbol: 0 for symbol in symbols}
        atsymbols = [atom.symbol for atom in atoms]
        for atsymbol in atsymbols:
            counts[atsymbol] +=1
        for symbol in symbols:
            comps[symbol] = counts[symbol]/len(atoms)
        return comps

