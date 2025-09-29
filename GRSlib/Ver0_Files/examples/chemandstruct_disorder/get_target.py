from opt_tools import *
from lammps_ACE_basis import *
import sys
av,var = build_target(sys.argv[1],save_all=True)

np.save('target_descriptors.npy',av)
np.save('target_var_descriptors.npy',var)
