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

    def none_min(self,data):
        #Will construct a set of additional commands to send to LAMMPS before scoring
        add_cmds=\
        """run 0
        write_data %s_last.data""" % (self.config.sections['TARGET'].job_prefix)
        before_score, after_score = self.scoring.add_cmds_before_score(add_cmds,data)
        end_data = self.config.sections['TARGET'].job_prefix + "_last.data"
        return before_score, after_score, data
    
    def fire_min(self,data,tole=0.0,tolf=0.0):
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
        minimize %1.1E %1.1E %s %s
        write_data %s_last.data""" % (tole,tolf,self.config.sections['GRADIENT'].nsteps, self.config.sections['GRADIENT'].nsteps, self.config.sections['TARGET'].job_prefix)
        before_score, after_score = self.scoring.add_cmds_before_score(add_cmds,data)
        end_data = self.config.sections['TARGET'].job_prefix + "_last.data"
        return before_score, after_score, end_data

    def line_min(self,data,tole=0.0,tolf=1.e-10,dmax=0.5):
        #Will construct a set of additional commands to send to LAMMPS before scoring
        add_cmds=\
        """delete_atoms overlap 0.3 all all
        compute cluster all cluster/atom  0.3
        compute max all reduce max c_cluster
        variable exit equal c_max
        fix halt all halt 10 v_exit > 1 error soft
        min_style  cg
        min_modify dmax %2.2f line quadratic
        dump 1 all custom 1 minimize_line.dump id type x y z fx fy fz
        minimize %1.1E %1.1E %s %s
        write_data %s_last.data
        """ % (dmax,tole,tolf, self.config.sections['GRADIENT'].nsteps, self.config.sections['GRADIENT'].nsteps, self.config.sections['TARGET'].job_prefix)
        before_score, after_score = self.scoring.add_cmds_before_score(add_cmds,data)
        end_data = self.config.sections['TARGET'].job_prefix + "_last.data"
        return before_score, after_score, end_data

    def box_min(self,data,tole=0.0,tolf=0.0,dmax=0.5):
        #Will construct a set of additional commands to send to LAMMPS before scoring
        add_cmds=\
        """delete_atoms overlap 0.3 all all
        compute cluster all cluster/atom  0.3
        compute max all reduce max c_cluster
        variable exit equal c_max
        fix halt all halt 10 v_exit > 1 error soft
        min_style  cg
        min_modify dmax %2.2f line quadratic
        dump 1 all custom 1 minimize_box.dump id type x y z fx fy fz
        fix box all box/relax aniso 0.0 couple none nreset 50 vmax %2.2f
        minimize %1.1E %1.1E %s %s
        write_data %s_last.data""" % (dmax,dmax,tole,tolf,self.config.sections['GRADIENT'].nsteps, self.config.sections['GRADIENT'].nsteps, self.config.sections['TARGET'].job_prefix)
        before_score, after_score = self.scoring.add_cmds_before_score(add_cmds,data)
        end_data = self.config.sections['TARGET'].job_prefix + "_last.data"
        return before_score, after_score, end_data

    def temp_min(self,data):
        #Will construct a set of additional commands to send to LAMMPS before scoring
        add_cmds=\
        """delete_atoms overlap 0.3 all all
        compute cluster all cluster/atom 0.3
        compute max all reduce max c_cluster
        variable exit equal c_max
        fix halt all halt 10 v_exit > 1 error soft
        velocity all create %s %s dist gaussian
        fix nve all nve
        fix lan all langevin %s %s 1.0 48279
        run %s
        unfix nve
        unfix lan
        write_data %s_last.data
        min_style  fire
        min_modify integrator eulerexplicit tmax 10.0 tmin 0.0 delaystep 5 dtgrow 1.1 dtshrink 0.5 alpha0 0.1 alphashrink 0.99 vdfmax 100000 halfstepback no initialdelay no
        dump 1 all custom 1 run_minimize.dump id type x y z fx fy fz
        minimize 1e-6 1e-6 %s %s
        write_data %s_last.data""" % (self.config.sections['GRADIENT'].temperature, np.random.randint(low=1, high=99999), 
                                      self.config.sections['GRADIENT'].temperature, self.config.sections['GRADIENT'].temperature, 
                                      self.config.sections['GRADIENT'].nsteps, self.config.sections['TARGET'].job_prefix, self.config.sections['GRADIENT'].nsteps, 
                                      self.config.sections['GRADIENT'].nsteps, self.config.sections['TARGET'].job_prefix)
        before_score, after_score = self.scoring.add_cmds_before_score(add_cmds,data)
        end_data = self.config.sections['TARGET'].job_prefix + "_last.data"
        return before_score, after_score, end_data

class Optimize:

    def __init__(self, pt, config, scoring, convert):
        self.pt = pt #ParallelTools()
        self.config = config #Config()
        self.scoring = scoring
        self.convert = convert
        self.create = Create(self.pt, self.config, self.convert)
        self.gradmove = Gradient(pt, config, scoring)

    def find_min_runner_up(self,selection,random_x_second=None):
        population = selection.copy()
        population.sort(key = lambda x:x[3],reverse=False)
        mp_inds = {ii:population[ii][1] for ii in range(len(population))}
        #print(mp_inds)
        #print('sort_pop',population)
        #print('win,runner_up',population[0],population[1])
        if random_x_second == None:
            return population[0],population[1]
        else:
            second = np.random.randint(1,random_x_second-1)
            try:
                return population[0],population[second]
            except IndexError:
                return population[0],population[0]
        
        #print(selection[mp_inds[0]],selection[mp_inds[1]])
    
    def tournament_selection_N(self,population_in,k=3,seed=None):
        population = population_in.copy()
        scores = [p[3] for p in population]
        scores_a = np.array(scores)
        #lowest_k_minus_1_scores = scores.copy().sort(key=lambda x : x[3],reverse=False)[:k-1]
        if seed != None:
            np.random.seed(seed)
        selection_ix = np.random.choice(range(0,len(population)),k-1,replace=False)
        not_ix = [j for j in range(0,len(population)) if j not in selection_ix]
        selection_jx = np.random.choice(not_ix,k-1,replace=False)
        mask_ix = scores_a[selection_jx] <= scores_a[selection_ix]
        #mask_ix = scores_a[selection_jx] < scores_a[selection_ix]
        selection_ix = selection_jx[mask_ix]
        updated = [population[ix] for ix in selection_ix]
        return updated

    def filter_unique(self,selection):
        used = []
        updated = []
        for s in selection:
            if s[1] not in used:
                updated.append(s)
                used.append(s[1])
        return updated     

    def unique_selection(self, data):
        #More of a super function that will call a bunch of the ones below
        #This should be the default since we dont want to send duplicates the crossover/mutation
        ki=3 #default
        #ki =4
        self.genetic = Genetic(self.pt, self.config,self.convert,self.scoring,self.gradmove)    
        starting_generation = Create.starting_generation(self,data)
        scores = []
        gen_winners = []
        for candidate in range(len(starting_generation)):
            file_name = self.config.sections['TARGET'].job_prefix+"_Cand%sGen%s.lammps-data"%(candidate,'Init')
            lammps_data = self.convert.ase_to_lammps(starting_generation[candidate],file_name)
            #Honestly I would prefer scores as a dictonary of Key:Item pairs, TODO later.
            scores.append(['Init', candidate, file_name, self.scoring.get_score(lammps_data)])
#            shutil.move(lammps_data, self.config.sections['TARGET'].job_prefix + "_Cand%sGen%s.data"%(candidate,0))
        for iteration in range(self.config.sections['GENETIC'].ngenerations):       
            population_in = scores.copy()
            print('pop in',population_in)
            #iterate selection method (good place to sub in different selection methods in the future)
            selected_sets = [self.tournament_selection_N(population_in,k=ki,seed=None) for idx in range(len(starting_generation))]
            print('raw sets',selected_sets)
            #remove empty selections (TODO remove empty selection solution)
            selected_sets = [s for s in selected_sets if len(s) >=2]
            selected= [item for sublist in selected_sets for item in sublist]
            #filter for uniqueness (no repeats of candidates from population in selection)
            print('selected pre filter',selected)
            filtered = self.filter_unique(selected)
            if len(filtered) > ki:
                selected = filtered
            else:
                print('not enough unique candidates found. using repeats')
            print('selected post filter',selected)
            #selected = self.filter_unique(selected)
            """
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
            """
            #NOTE set random_x_second to None to do (winner + runner_up)
            winner, runner_up = self.find_min_runner_up(selected,random_x_second=ki)
            #At the last round, hold onto the runner up for a possible crossover
            #TODO We will probably need to implement different pairing methods rather than just (winner + runner_up)
            #     especially if having trouble finding the right solution. adding some randomness helps avoid 
            #     convergence to local minimum.
            #     we could also try (winner + random) , (winner + random_top_10_percent) , (random + random)
            print("Iteration:",iteration, "Winner:",winner, "Second:",runner_up)
            with open("scoring_%s.txt"%self.config.sections['TARGET'].job_prefix, "a") as f:
                print(iteration, winner, runner_up, file=f)
            print('winner',winner,winner[2])
            atoms_winner = self.convert.lammps_to_ase(winner[2])
            atoms_runner_up = self.convert.lammps_to_ase(runner_up[2])

            #Winning candidate is then appended to winners circle list 
            gen_winners.append(winner) #appends arrays along the first axis (row-wise)
            #Now setup for the next iteration of the tournament
            """
            #NOTE this is one possible way we can do a loop over structures to get mutations from different candidates & not always the lowest loss candidate
            scores = []
            for iset,set_pair in enumerate(selected_sets):
                winner, runner_up = self.find_min_runner_up(selected,random_x_second=ki)
                atoms_winner = self.convert.lammps_to_ase(set_pair[0][2])
                atoms_runner_up = self.convert.lammps_to_ase(set_pair[1][2]) #TODO update for multi element list
                if np.random.rand() < float(self.config.sections['GENETIC'].mutation_rate):
                    batch = self.genetic.mutation(atoms_winner,single=True) #Will mutation only take in one structure?
                else:
                    batch = self.genetic.crossover(atoms_winner, atoms_runner_up) #Should have two structures

                for candidate in range(len(batch)):
                    #file_name = self.config.sections['TARGET'].job_prefix+"_Cand%sGen%s.lammps-data"%(candidate,iteration)
                    file_name = self.config.sections['TARGET'].job_prefix+"_Cand%dGen%s.lammps-data"%(iset,iteration)
                    #NOTE: is lammps_data here always pulled from starting_generation? if so
                    #      it needs to be updated so that the candidate is pulled from the 'current_generation'
                    #      so far, it is unclear to me if the starting generation is just repeatedly operated on or
                    #      if the generation is being updated and operated on.
                    lammps_data = self.convert.ase_to_lammps(starting_generation[candidate],file_name) #NOTE question in comment above here
                    #scores.append([iteration, candidate, file_name, self.scoring.get_score(lammps_data)])
                    score_win = self.scoring.get_score(self.convert.ase_to_lammps(atoms_winner,file_name +'-win'))
                    score_conv = self.scoring.get_score(lammps_data)
                    #TODO help james understand why the 'winner' score not the lowest candidate score in line below
                    print('score comp', candidate, len(batch), score_conv, score_win)
                    scores.append([iteration, candidate, file_name, score_conv])
                    #shutil.move(lammps_data, self.config.sections['TARGET'].job_prefix + "_Cand%sGen%s.data"%(candidate,iteration))
            """
            #Now setup for the next iteration of the tournament
            scores = []
            if np.random.rand() < float(self.config.sections['GENETIC'].mutation_rate):
                batch = self.genetic.mutation(atoms_winner) #Will mutation only take in one structure?- TODO enable mutate >1 structures
            else:
                batch = self.genetic.crossover(atoms_winner, atoms_runner_up) #Should have two structures
                #batch = self.genetic.crossover_ASE(atoms_winner, atoms_runner_up) #crossover function from ASE
            #print('batch i',batch)
            for candidate in range(len(batch)):
                file_name = self.config.sections['TARGET'].job_prefix+"_Cand%sGen%s.lammps-data"%(candidate,iteration)
                lammps_data = self.convert.ase_to_lammps(batch[candidate],file_name)
                score_conv = self.scoring.get_score(lammps_data)
                scores.append([iteration, candidate, lammps_data, score_conv])
                print('gen %d cand %d score %f' % (iteration,candidate,score_conv))
                shutil.move(lammps_data, self.config.sections['TARGET'].job_prefix + "_Cand%sGen%s.lammps-data"%(candidate,iteration))
            current_generation = []
        for file in glob.glob(self.config.sections['TARGET'].job_prefix + "_Cand*Gen*"):
            
            if file not in  [row[2] for row in gen_winners]:
                os.remove(file)

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
    
