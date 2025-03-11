from GRSlib.parallel_tools import ParallelTools
from GRSlib.motion.scoring import Scoring
import numpy as np

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
        print(np.shape(current_desc),np.shape(target_desc))
        #self.loss_ff = 0.0 #set to a constant term to initialise

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
        print(np.shape(current_desc),np.shape(target_desc))
        #self.loss_ff = 0.0 #set to a constant term to initialise

    def crossover(p1, p2, inputseed = None, types=None, endpoint_compositions=False):
        if endpoint_compositions or types==None:
            assert len(p1) == len(p2), "parents must have the same length"
            psize = len(p1)
            if inputseed != None:
                np.random.seed(inputseed)
            cross_point = np.random.randint(1, psize-1)
            c1 = p1[:cross_point] + p2[cross_point:]
            c2 = p2[:cross_point] + p1[cross_point:]
        else:
            assert len(p1) == len(p2), "parents must have the same length"
            itr = 0
            comps_dct1 = get_comp(p1,types)
            comp_vals1 = list(comps_dct1.values())
            comps_dct2 = get_comp(p2,types)
            comp_vals2 = list(comps_dct2.values())
            while itr == 0 or any([icomp == 0.0 for icomp in comp_vals1]) or any([icomp == 0.0 for icomp in comp_vals2]):
                psize = len(p1)
                if inputseed != None:
                    np.random.seed(inputseed)
                cross_point = np.random.randint(1, psize-1)
                c1 = p1[:cross_point] + p2[cross_point:]
                c2 = p2[:cross_point] + p1[cross_point:]
                comps_dct1 = get_comp(c1,types)
                comp_vals1 = list(comps_dct1.values())
                comps_dct2 = get_comp(c2,types)
                comp_vals2 = list(comps_dct2.values())
                itr += 1
        return [c1, c2]

    def mutation(current_atoms,mutation_type='perturb_N',types=['Ag'],scale=0.5):
        mutation_types = {
        'perturb_one' : perturb_one_atom,
        'perturb_N' : perturb_N_atoms,
        'flip_one' : flip_one_atom,
        'flip_N' : flip_N_atoms,
        }
        if 'flip' in mutation_type:
            return mutation_types[mutation_type](current_atoms,types)
        else:
            return mutation_types[mutation_type](current_atoms,scale)

    def tournament_selection(population, scores, k=3, inputseed=None):
        if inputseed != None:
            np.random.seed(inputseed)
        selection_ix = np.random.randint(len(population))
        for ix in np.random.randint(0, len(population), k-1):
            # check if better (e.g. perform a tournament)
            if scores[ix] < scores[selection_ix]:
                selection_ix = ix
        return population[selection_ix]

    def variable_tournament_selection(population, scores, k=6, inputseed=None):
        if inputseed != None:
            np.random.seed(inputseed)
        selection_ix = np.random.randint(len(population))
        for ix in np.random.randint(0, len(population), k-1):
            # check if better AND dissimilar (e.g. perform a tournament)
            if scores[ix] < scores[selection_ix] and scores[ix] != scores[selection_ix]:
                selection_ix = ix
        return population[selection_ix]


