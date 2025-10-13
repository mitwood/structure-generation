#from GRSlib.parallel_tools import ParallelTools
#from GRSlib.motion.scoring_factory import scoring
from GRSlib.motion.scoring import Scoring
from GRSlib.motion.genetic import Genetic
from GRSlib.motion.create import *
import numpy as np
import random, shutil, os, glob

# Two types of motion (aka changes) can be applied to a structure 1) Gradients of the loss function (energy/score) 
# yielding continuous changes, or 2) discrete moves that include (atom addition/removal, chemical identities).
# New types can be added as classes if there is need, follow class inheritence of existing motion types.
# These classes require a scoring function, which is inherited after initializing in GRS.py. 

# A key difference between Gradient and Optimize is that the former operates on a single structure at a time, while
# the latter will create a dictionary of structures and their scores to be evaluated. 

class Gradient:

    def __init__(self, pt, config, scoring):
        self.pt = pt #ParallelTools()
        self.config = config #Config()
        self.scoring = scoring

    def fire_min(self,data):
        #Will construct a set of additional commands to send to LAMMPS before scoring
        add_cmds=\
        """delete_atoms overlap 0.3 all all
        compute cluster all cluster/atom  0.3
        compute max all reduce max c_cluster
        variable exit equal c_max
        fix halt all halt 10 v_exit > 1 error soft
        min_style  fire
        min_modify integrator eulerexplicit tmax 10.0 tmin 0.0 delaystep 5 dtgrow 1.1 dtshrink 0.5 alpha0 0.1 alphashrink 0.99 vdfmax 100000 halfstepback no initialdelay no
        dump 1 all custom 1 minimize_fire.dump id type x y z fx fy fz
        displace_atoms all random 0.1 0.1 0.1 %s units box
        minimize 1e-6 1e-6 %s %s
        write_data %s_last.data""" % (np.random.randint(low=1, high=99999),self.config.sections['GRADIENT'].nsteps, self.config.sections['GRADIENT'].nsteps, self.config.sections['TARGET'].job_prefix)
        before_score, after_score = self.scoring.add_cmds_before_score(add_cmds,data)
        end_data = self.config.sections['TARGET'].job_prefix + "_last.data"
        return before_score, after_score, end_data

    def line_min(self,data):
        #Will construct a set of additional commands to send to LAMMPS before scoring
        add_cmds=\
        """delete_atoms overlap 0.3 all all
        compute cluster all cluster/atom  0.3
        compute max all reduce max c_cluster
        variable exit equal c_max
        fix halt all halt 10 v_exit > 1 error soft
        min_style  cg
        min_modify dmax 0.05 line quadratic
        dump 1 all custom 1 minimize_line.dump id type x y z fx fy fz
        displace_atoms all random 0.1 0.1 0.1 %s units box
        minimize 1e-6 1e-6 %s %s
        write_data %s_last.data
        """ % (np.random.randint(low=1, high=99999),self.config.sections['GRADIENT'].nsteps, self.config.sections['GRADIENT'].nsteps, self.config.sections['TARGET'].job_prefix)
        before_score, after_score = self.scoring.add_cmds_before_score(add_cmds,data)
        end_data = self.config.sections['TARGET'].job_prefix + "_last.data"
        return before_score, after_score, end_data

    def box_min(self,data):
        #Will construct a set of additional commands to send to LAMMPS before scoring
        add_cmds=\
        """delete_atoms overlap 0.3 all all
        compute cluster all cluster/atom  0.3
        compute max all reduce max c_cluster
        variable exit equal c_max
        fix halt all halt 10 v_exit > 1 error soft
        in_style  cg
        min_modify dmax 0.05 line quadratic
        dump 1 all custom 1 minimize_box.dump id type x y z fx fy fz
        fix box all box/relax iso 0.0 vmax 0.001
        displace_atoms all random 0.1 0.1 0.1 %s units box
        minimize 1e-6 1e-6 %s %s
        write_data %s_last.data""" % (np.random.randint(low=1, high=99999),self.config.sections['GRADIENT'].nsteps, self.config.sections['GRADIENT'].nsteps, self.config.sections['TARGET'].job_prefix)
        before_score, after_score = self.scoring.add_cmds_before_score(add_cmds,data)
        end_data = self.config.sections['TARGET'].job_prefix + "_last.data"
        return before_score, after_score, end_data

    def temp_min(self,data):
        #Will construct a set of additional commands to send to LAMMPS before scoring
        add_cmds=\
        """compute cluster all cluster/atom  0.3
        compute max all reduce max c_cluster
        variable exit equal c_max
        fix halt all halt 10 v_exit > 1 error soft
        velocity all create %s %s dist gaussian
        fix nve all nve
        fix lan all langevin %s %s 1.0 48279
        run %s
        unfix nve
        unfix lan
        delete_atoms overlap 0.3 all all
        min_style  fire
        min_modify integrator eulerexplicit tmax 10.0 tmin 0.0 delaystep 5 dtgrow 1.1 dtshrink 0.5 alpha0 0.1 alphashrink 0.99 vdfmax 100000 halfstepback no initialdelay no
        dump 1 all custom 1 run_minimize.dump id type x y z fx fy fz
        minimize 1e-6 1e-6 %s %s
        write_data %s_last.data""" % (self.config.sections['GRADIENT'].temperature, np.random.randint(low=1, high=99999), 
                                      self.config.sections['GRADIENT'].temperature, self.config.sections['GRADIENT'].temperature, 
                                      self.config.sections['GRADIENT'].nsteps, self.config.sections['GRADIENT'].nsteps, 
                                      self.config.sections['GRADIENT'].nsteps, self.config.sections['GRADIENT'].nsteps, 
                                      self.config.sections['TARGET'].job_prefix)

        before_score, after_score = Scoring.add_cmds_before_score(add_cmds,data)
        end_data = self.config.sections['TARGET'].job_prefix + "_last.data"
        return before_score, after_score, end_data

class Optimize:

    def __init__(self, pt, config, scoring, convert):
        self.pt = pt #ParallelTools()
        self.config = config #Config()
        self.scoring = scoring
        self.convert = convert
        self.create = Create.starting_generation(self, self.pt, self.config)
        self.gradmove = Gradient(pt, config, scoring)

    def unique_tournament_selection(self, *args):
        #More of a super function that will call a bunch of the ones below
        #This should be the default since we dont want to send duplicates the crossover/mutation
        self.genetic = Genetic(self.pt, self.config,self.convert,self.scoring,self.gradmove)       
        starting_generation = Create.starting_generation(self)
        scores = []

        for candidate in range(len(starting_generation)):
            file_name = self.config.sections['TARGET'].job_prefix+"_Cand%sGen%s.lammps-data"%(candidate,'Init')
            lammps_data = self.convert.ase_to_lammps(starting_generation[candidate],file_name)
            #Honestly I would prefer scores as a dictonary of Key:Item pairs, TODO later.
            scores.append(['Init', candidate, file_name, self.scoring.get_score(lammps_data)])
#            shutil.move(lammps_data, self.config.sections['TARGET'].job_prefix + "_Cand%sGen%s.data"%(candidate,0))
        
        for iteration in range(self.config.sections['GENETIC'].ngenerations):               
#            selection = np.unique(scores[:2])#Cull candidates for uniqueness. 0: generation, 1: id, 2: file-name, 3: score
            selection = scores.copy() 
            for round in range(len(selection)-2): #-2 because we want to keep the best and second-best for crossover
                compare_pair = np.random.randint(0, len(selection), 2)
                #This is where a dictionary of score/selection would be nice and clean instead of fixed index references.
                if scores[compare_pair[0]][3] <= scores[compare_pair[1]][3]:
                    loser = compare_pair[1]
                    selection.pop(loser)
                else:
                    loser = compare_pair[0]
                    selection.pop(loser)
            #At the last round, hold onto the runner up for a possible crossover
            print(iteration,selection)
            if selection[0][3] <= selection[0][3]:
                winner = selection[0]
                runner_up = selection[1]
            else:
                winner = selection[1]
                runner_up = selection[0]

            atoms_winner = self.convert.lammps_to_ase(winner[2])
            atoms_runner_up = self.convert.lammps_to_ase(runner_up[2])

            #Winning candidate is then appended to winners circle list : [generation, ase.Atoms, score]
            try:
                gen_winners = np.c_[gen_winners, winner] #appends arrays along the first axis (row-wise)
            except:
                gen_winners = winner
            #Now setup for the next iteration of the tournament
            scores = [] 
#            for file in glob.glob(self.config.sections['TARGET'].job_prefix + "_Cand*Gen*"):
#                if file not in gen_winners:
#                    os.remove(file)

            if np.random.rand() < float(self.config.sections['GENETIC'].mutation_rate):
                batch = self.genetic.mutation(atoms_winner) #Will mutation only take in one structure?
            else:
                batch = self.genetic.crossover(atoms_winner, atoms_runner_up) #Should have two structures

            for candidate in range(len(batch)):
                file_name = self.config.sections['TARGET'].job_prefix+"_Cand%sGen%s.lammps-data"%(candidate,iteration)
                lammps_data = self.convert.ase_to_lammps(starting_generation[candidate],file_name)
                scores.append([iteration, candidate, file_name, self.scoring.get_score(lammps_data)])
                #shutil.move(lammps_data, self.config.sections['TARGET'].job_prefix + "_Cand%sGen%s.data"%(candidate,iteration))

        return gen_winners
        #End of tournament returns winners circle list to GRS.py -> (convert.ASEtoLAMMPS + write score output)

    def latin_hyper(self, **kwargs):
        #placeholder for equal sampling accross input space of generated strucutres
        pass

    def sim_anneal(self, **kwargs):
        #placeholder for simulated annealing of generated strucutres
        pass

    def lib_optimizer(self, **kwargs):
        #placeholder, possible for DAKOTA or pyMOO coupling?
        pass
    
