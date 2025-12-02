from mpi4py import MPI
from GRSlib.GRS import GRS
import random, copy, os, glob, shutil
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

settings = \
{
"BASIS":
    {
    "descriptor": "ACE",
    "numTypes": 1,
    "elements": "W", #Needs to be a dict?
    "rcutfac": 5.5,
    "lambda": 1.4,
    "ranks": "1 2 3",
    "lmax": "0 0 0",
    "lmin": "0 0 0",
    "nmax": "8 1 1",
    "nmaxbase": 8,
    "bzeroflag": 0
    },
"SCORING":
    {
    "score_type": "moments",
    "strength_target": 1.0, 
    "strength_prior": 0.0, 
    "moments": "mean stdev" ,
    "moments_coeff": "1.0 0.01",
    "moments_bonus": "0 0" ,
    },
"TARGET":
    {
    "target_fname": "fcc.data",
    "target_fdesc": "fcc.npy",
    "start_fname": "bcc.data",
    "prior_fdesc": "prior.npy",
    "job_prefix": "BCCtoFCC"
    },
"GRADIENT":
    {
    "soft_strength": 1.0,
    "ml_strength": 1.0,
    "nsteps": 1000,
    "temperature": 100.0,
    "min_type": "temp"
    },
"GENETIC":
    {
    "start_type": "random",  #Can be random or template right now. If template, starting generation is ["TARGET"].start_fname
    "mutation_rate": 0.5,
    "mutation_types": {"perturb": 0.3, "change_ele": 0.0, "atom_count" : 0.30, "volume" : 0.30, "minimize" : 0.05, "ortho_cell" : 0.05}, 
    "population_size": 100,
    "ngenerations": 10,
    "max_atoms": 50,
    "min_atoms": 10,
    "max_length_aspect": 2.0,
    "max_angle_aspect": 2.0,
    "density_ratio": 1.1,
    "composition_constraint": {'W':1.0}
    }
}

grs = GRS(settings,comm=comm)

tourny_winners = []
score = grs.get_score(settings["TARGET"]["start_fname"])
print("     Starting Score:",score)

scores, best_struct = grs.genetic_move(settings["TARGET"]["target_fname"])

score = grs.get_score(best_struct)
print("     Best Score:",score, "from",best_struct)
renamed_best = "Winner-0_"+best_struct
shutil.move(best_struct, renamed_best)
tourny_winners.append(renamed_best)

print("Setting priors with:",tourny_winners)
grs.set_prior(tourny_winners)
grs.config.sections['SCORING'].strength_target = 0.0
grs.config.sections['SCORING'].strength_prior = 1.0
#grs.config.view_state('SCORING')
score = grs.get_score(renamed_best)
print("    Score wrt Priors (best candidates of each gen):",score)

grs.config.sections['GENETIC'].start_type = "template"

for i in range(10):
    for file in glob.glob(grs.config.sections['TARGET'].job_prefix + "_Cand*Gen*"):
        if file not in  [row[2] for row in tourny_winners]:
            os.remove(file)
    grs.config.sections['SCORING'].strength_target = 1.0
    grs.config.sections['SCORING'].strength_prior = 0.0

    scores, best_struct = grs.genetic_move(renamed_best)

    score = grs.get_score(best_struct)
    print("     Best Score:",score, "from",best_struct)
    renamed_best = "Winner-%s_"%i+best_struct
    shutil.move(best_struct, renamed_best)
    tourny_winners.append(renamed_best)

    print("     Ending Score:",score, "from",renamed_best)   

    print("Setting priors with:",tourny_winners)
    grs.set_prior(tourny_winners)
    grs.config.sections['SCORING'].strength_target = 0.0
    grs.config.sections['SCORING'].strength_prior = 1.0
    #grs.config.view_state('SCORING')
    score = grs.get_score(renamed_best)
    print("    Score wrt Priors (best candidates of each gen):",score)

