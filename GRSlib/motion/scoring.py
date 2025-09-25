#from GRSlib.parallel_tools import ParallelTools
#from GRSlib.motion.lossfunc.moments import Moments
#from GRSlib.motion.lossfunc import Gradient
from GRSlib.converters.sections.lammps_base import Base, _extract_compute_np
import lammps, lammps.mliap
from lammps.mliap.loader import *
from functools import partial
import numpy as np

#Scoring has to be a class within motion because we want a consistent reference for scores, and this
#refrence will be LAMMPS using a constructed potential energy surface from the representation loss function.
#Sub-classes of Scoring will be versions of this representation loss function (Moments, Entropy, etc), allowing
#for custom verions to be added without trouble.

class Scoring:

#    def __init__(self, pt, config, data, loss_ff, **kwargs):
    def __init__(self, pt, config, loss_func, data, descriptors):
        self.pt = pt #ParallelTools()
        self.config = config #Config()
        self.data = data
        self.descriptors = descriptors
        self.loss_func = loss_func
        self.loss_func.__init__(self.pt, self.config, self.descriptors) #Initialize loss function, get ready to send to scoring
        self.loss_func(self.pt, self.config, self.descriptors) #Call loss function, get ready to send to scoring
        self.lmp = self.pt.initialize_lammps('log.lammps',0)
        lammps.mliap.activate_mliappy(self.lmp)

    def construct_lmp(self):
        #Generates the major components of a lammps script needed for a scoring call
#        me = self.lmp.extract_setting("world_rank")
#        nprocs = self.lmp.extract_setting("world_size")
#        cmds = ["-screen", "none", "-log", "none"]
#        self.lmp = lammps(cmdargs = cmds)
        self.lmp = self.pt.initialize_lammps('log.lammps',0)
        lammps.mliap.activate_mliappy(self.lmp)
        construct_string=\
        """
        units metal
        atom_style atomic
        read_data {}
        pair_style hybrid/overlay soft 1.0 mliap model mliappy LATER descriptor ace coupling_coefficients.yace
        pair_coeff * * soft {}
        pair_coeff * * mliap {}
        neighbor 2.3 bin
        neigh_modify one 10000
        thermo 10
        thermo_style custom step etotal temp press
        """
        init_lmp=construct_string.format(self.data, self.config.sections["MOTION"].soft_strength, (" ".join(str(x) for x in self.config.sections['BASIS'].elements)))
        #TODO make the possibility to import any reference potential to be used with the mliap one
        self.lmp.commands_string(init_lmp)
        lammps.mliap.load_model(self.loss_func)
        self.lmp.command("run 0")
              
    def get_atomic_energies(self):
        #Return as array per-atom energies for the set of potentials applied
        lammps.mliap.activate_mliappy(self.lmp)
        self.construct_lmp()
        self.lmp.command("compute peatom all pe/atom")
        self.lmp.command("run 0")
        num_atoms = self.lmp.extract_global("natoms")
        atom_energy = _extract_compute_np(self.lmp, "peatom", 0, 2, (num_atoms, 1))
#        del self.lmp
        return atom_energy

    def get_norm_forces(self):
        #Return as array per-atom forces 
        self.construct_lmp()
        self.lmp.command("compute fatom all property/atom fx fy fz")
        self.lmp.command("run 0")
        num_atoms = self.lmp.extract_global("natoms")
        atom_forces = _extract_compute_np(self.lmp, "fatom", 0, 2, (num_atoms, 3))        
#        del self.lmp
        return atom_forces

    def get_score(self):
        self.construct_lmp()
        self.lmp.command("run 0")
        score = self.lmp.get_thermo("pe") # potential energy
#        del self.lmp
        return score

    def add_cmds_before_score(self,string):
        self.construct_lmp()
        before_score = self.get_score()
        self._extract_commands(string)
        self.lmp.commands_string("run 0")
        after_score = self.lmp.get_thermo("pe") # potential energy
#        del self._lmp
        return before_score, after_score

    def _extract_commands(self,string):
        #Can be given a block of text where it will split them into individual commands
        add_lmp_lines = [x for x in string.splitlines() if x.strip() != '']
        for line in add_lmp_lines:
                self.lmp.command(line)
