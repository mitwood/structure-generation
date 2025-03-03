from time import time
from mpi4py import MPI
from GRSlib.GRS import GRS
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

settings = "../GenericInput.in"
grs = GRS(settings, comm=comm)

attributes = [attr for attr in dir(grs) if not attr.startswith('__')]
print("attr of grs:")
print(attributes)

attributes = [attr for attr in dir(grs.config) if not attr.startswith('__')]
print("attr of grs.config:")
print(attributes)

attributes = [attr for attr in dir(grs.config.sections) if not attr.startswith('__')]
print("attr of grs.config.sections:")
print(attributes)

attributes = [attr for attr in dir(grs.config.sections.values) if not attr.startswith('__')]
print("attr of grs.config.sections.values")
print(attributes)

#print(grs.config.sections["BASIS"].descriptor)
print("!")
exit()
