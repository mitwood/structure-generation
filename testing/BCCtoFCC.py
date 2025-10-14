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
    "lmax": "0 3 3",
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
    "mutation_rate": 0.50,
    "mutation_types": {"perturb": 0.2, "change_ele": 0.0, "atom_count" : 0.30, "volume" : 0.20, "minimize" : 0.2, "ortho_cell" : 0.10}, 
    "population_size": 40,
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

score = grs.get_score(settings["TARGET"]["start_fname"])
print("     Starting Score:",score)

updated_struct = settings["TARGET"]["start_fname"]
grs.set_prior([updated_struct])

scores, best_struct = grs.genetic_move(updated_struct)

updated_struct = grs.gradient_move(best_struct)
score = grs.get_score(updated_struct)
print("     Ending Score:",score)

#updated_struct = grs.update_start(updated_struct,"MinScore")
#grs.set_prior(glob.glob(settings['TARGET']["job_prefix"]+"*.data"))

exit()
