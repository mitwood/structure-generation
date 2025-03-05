from GRSlib.parallel_tools import ParallelTools
from GRSlib.motion import moments
from jax import grad, jit

#Scoring has to be a class within motion because we want a consistent reference for scores, ans this
#refrence will be LAMMPS using a constructed potential energy surface from the representation loss function



class Scoring:

    def __init__(self, data, pt, config):
        self.pt = pt #ParallelTools()
        self.config = config #Config()
        #Bring in the target and current descriptors here, will be with self. then
        #descriptors_flt = descriptors.flatten()
        self.current_desc = None
        self.target_desc = None
        self.first_mom = None
        self.second_mom = None
        self.third_mom = None
        self.fourth_mom = None
        self.loss_ff = 0.0 #set to a constant term to initialise

    @partial(jit, static_argnums=(0))
    def loss_function(self,pt,config):
        #construct the Jax graph to hand off to lammps
        if (any(self.config.sections['SCORING'].moments) == 'mean'):
            print("Adding mean to loss function force field")
            self.first_mom = moments.first_moment(self)
        if (any(self.config.sections['SCORING'].moments) == 'stdev'):
            print("Adding standard deviation to loss function force field")
            self.second_mom = moments.second_moment(self)
        if (any(self.config.sections['SCORING'].moments) == 'skew'):
            print("Adding skewness to loss function force field")
            self.third_mom = moments.third_moment(self)
        if (any(self.config.sections['SCORING'].moments) == 'kurt'):
            print("Adding kurtosis to loss function force field")
            self.fourth_mom = moments.fourth_moment(self)
        set_of_moments = [self.first_mom, self.second_mom, self.third_mom, self.fourth_mom]
        for item in set_of_moments:
            if item!=None:
                self.loss_ff += item
        return self.loss_ff, grad(self.loss_ff)

    def get_atomic_energies(self):
        #Return as array per-atom energies for the set of potentials applied
        self._lmp = self.pt.initialize_lammps(self.config.args.lammpslog, printlammps)
        lammps.mliap.activate_mliappy(self._lmp)

        self._lmp.command("clear")
        self._lmp.command("units metal")
        self._lmp.command("atom_style atomic")

        lammps.mliap.load_model(self.model)
        del self._lmp

    def get_norm_forces(self):
        #Return as array per-atom forces 
        self._lmp = self.pt.initialize_lammps(self.config.args.lammpslog, printlammps)
        del self._lmp

    def get_score(self):
        #Return as array unweighted scores per moment

        self.lmp.commands_string("run 0")
        result = self.lmp.get_thermo("pe") # potential energy

