#from GRSlib.parallel_tools import ParallelTools
#from GRSlib.motion.lossfunc.moments import Moments
#from GRSlib.motion.lossfunc import Gradient
from GRSlib.converters.sections.lammps_base import Base, _extract_compute_np
import lammps, lammps.mliap
from lammps.mliap.loader import *
from functools import partial
from ase.data import atomic_masses, atomic_numbers
from ase.io import read
import numpy as np

#Scoring has to be a class within motion because we want a consistent reference for scores, and this
#refrence will be LAMMPS using a constructed potential energy surface from the representation loss function.
#Sub-classes of Scoring will be versions of this representation loss function (Moments, Entropy, etc), allowing
#for custom verions to be added without trouble.

class Scoring:

#    def __init__(self, pt, config, data, loss_ff, **kwargs):
    def __init__(self, pt, config, loss_func, descriptors):
        self.pt = pt #ParallelTools()
        self.config = config #Config()
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
        types = self.config.sections["BASIS"].elements
        Z_of_type = {i+1:atomic_numbers[ele] for i,ele in enumerate(types)}
        tmp_ats = read(self.data,format='lammps-data',Z_of_type=Z_of_type)
        has_types = []
        for at in tmp_ats:
            if at.symbol not in has_types:
                has_types.append(at.symbol)
        #TODO check this when some elements not in self.data (e.g. do we need list below if we only have Ca instead of both Ca and Mg in self.data)
        #masses_per_typ = {typ+1:atomic_masses[atomic_numbers[ele]] for typ,ele in enumerate(has_types)}
        masses_per_typ = {typ+1:atomic_masses[atomic_numbers[ele]] for typ,ele in enumerate(types)}
        self.lmp = self.pt.initialize_lammps('log.lammps',0)
        lammps.mliap.activate_mliappy(self.lmp)
        #NOTE thermo modify norm yes to make score magnitude (and soft contribution) independent of system size
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
        thermo_modify norm yes
        """
        init_lmp=construct_string.format(self.data, self.config.sections["GRADIENT"].soft_strength, (" ".join(str(x) for x in self.config.sections['BASIS'].elements)))
        mass_str = ""
        for typ,mass in masses_per_typ.items():
            mass_str = mass_str + "mass     %d  %f \n" % (typ,mass)
        init_lmp = init_lmp + mass_str
        #TODO make the possibility to import any reference potential to be used with the mliap one
        self.lmp.commands_string(init_lmp)
        lammps.mliap.load_model(self.loss_func)
        self.lmp.command("run 0")
              
    def get_atomic_energies(self):
        #Return as array per-atom energies for the set of potentials applied
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

    def get_score(self,data):
        self.data = data
        num_atoms = len(read(data,format='lammps-data'))
        self.construct_lmp()
        self.lmp.command("run 0")
        score = self.lmp.get_thermo("pe") # potential energy
        #TODO, do we still need this if thermo #NO WE DONT
        #score /= num_atoms
#        del self.lmp
        return score

    def add_cmds_before_score(self,string,data):
        self.data = data
        self.construct_lmp()
        before_score = self.get_score(data)
        num_atoms = len(read(data,format='lammps-data'))
        try:
            self._extract_commands(string)
            self.lmp.commands_string("run 0")
            after_score = self.lmp.get_thermo("pe") # potential energy
            #after_score /= num_atoms
        except:
            print("LAMMPS Crashed, reported score will be prior to motion.")
            after_score = before_score
#        del self.lmp
        return before_score, after_score

    def _extract_commands(self,string):
        #Can be given a block of text where it will split them into individual commands
        add_lmp_lines = [x for x in string.splitlines() if x.strip() != '']
        for line in add_lmp_lines:
                self.lmp.command(line)
