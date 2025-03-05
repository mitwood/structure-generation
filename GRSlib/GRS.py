from GRSlib.parallel_tools import ParallelTools
from GRSlib.io.input import Config
from GRSlib.converters.convert_factory import convert

import random


class GRS:
    """ 
    >Big comment goes here explaining what the code does
    Args:
        input (str): Optional dictionary or path to input file when using library mode; defaults to 
                     None for executable use.
        comm: Optional MPI communicator when using library mode; defaults to None.

    Attributes:
        pt (:obj:`class` ParallelTools): Instance of the ParallelTools class for helping MPI 
                                         communication and shared arrays.
        config (:obj:`class` Config): Instance of the Config class for initializing settings, 
                                      initialized with a ParallelTools instance.
        >Update once more of the code structure is fleshed out
    """
    def __init__(self, input=None, comm=None, arglist: list=[]):
        self.comm = comm
        # Instantiate ParallelTools and Config instances belonging to this GRS instance.
        # NOTE: Each proc in `comm` creates a different `pt` object, but shared arrays still share 
        #       memory within `comm`.
        self.pt = ParallelTools(comm=comm)
        self.pt.all_barrier()
        self.config = Config(self.pt, input, arguments_lst=arglist)

        # Instantiate other backbone attributes.
#        self.basis = basis(self.config.sections["BASIS"].descriptor, self.pt, self.config) if "BASIS" in self.config.sections else None

        self.convert = convert(self.config.sections['BASIS'].descriptor,self.pt,self.config) 
        #^ Initialized with the appropiate subclass method defined by user input

        # Check LAMMPS version if using nonlinear solvers.
        if (hasattr(self.pt, "lammps_version")):
            if (self.pt.lammps_version < 20220915):
                raise Exception(f"Please upgrade LAMMPS to 2022-09-15 or later to use MLIAP based structure searching.")

        #Start by converting the target structure to descriptors
        if self.config.sections['TARGET'].target_fname is None:
            print('Target structure not found or undefined')
        else:
            target_descriptors = self.convert.run_lammps_single(self.config.sections['TARGET'].target_fname)

    def __del__(self):
        """Override deletion statement to free shared arrays owned by this instance."""
        self.pt.free()
        del self

    def __setattr__(self, name: str, value):
        """
        Override set attribute statement to prevent overwriting important attributes of an instance.
        """
        protected = ("pt", "config")
        if name in protected and hasattr(self, name):
            raise AttributeError(f"Overwriting {name} is not allowed; instead change {name} in place.")
        else:
            super().__setattr__(name, value)

    def convert_to_desc(self):
        """
        Accepts a structure (xyz) as input and will return descriptors (D), optionally will convert
        between file types (xyz=lammps-data, ase.Atoms, etc)
        """
        @self.pt.single_timeit
        def convert_to_desc():
            #Pass data to, and do something with the functs of convert
            self.convert.run_lammps_single(self.data)
            print("Called Convert To Descriptors")
        self.descriptors = convert_to_desc()

    def propose_structure(self):
        """
        Propose new structure from random, ase, or templates.
        """
        @self.pt.single_timeit
        def propose_structure():
            #1)
            print("Called Propose_Structure")
        propose_structure()

    def genetic_move(self):
        """
        Hybridize or mutate a structure using a set of moves sampled via a genetic algorithm
        """
        @self.pt.single_timeit
        def genetic_move():
            #1) Propose a set of structures from templates, ase, or random (or read in a list of ase.Aatoms objects)
            #2) Score each of the candidates (ase_to_lammps -> run_single)
            #3) Hybridize, Mutate based on set of rules and probabilities
            #4) Store socring information with best-of-generation and best-overall isolated
            #5) Loop until generation limit or scoring residual below threshold
            print("Called Genetic_Move")
        genetic_move()

    def gradient_move(self):
        """
        Accepts a structure (xyz, ase.Atoms) as input and will return updated structure (xyz, ase.Atoms) that 
        has been modified by motion of atoms on the loss function potential
        """
        @self.pt.single_timeit
        def gradient_move():
            #1) Take in target descriptors, convert to moments of descriptor distribution
            #2) Take in current descriptors, convert to moments of descriptor distribution
            #3) Construct a fictitious potential energy surface based on difference in moments
            #4) Assemble a LAMMPS input script that overlaps potentials and runs dynamics
            #5) Return an updated structure and scoring on the difference in moments
            print("Called Gradient_Move")
        gradient_move()

    def baseline_training(self):
        """
        Accepts a structure (xyz, ase.Atoms) as input and will return updated structure (xyz, ase.Atoms) that 
        has been modified by motion of atoms on the loss function potential
        """
        @self.pt.single_timeit
        def baseline_training():
            #This is more of a 'super' function because it will call many other routines to give the result of a
            #baseline training set with many (hundreds? thousands?) candidate structures. Should return a score
            #of the training diversity based on moments of the descriptor distribution.
            print("Called Baseline_Training")
        baseline_training()

    def write_output(self):
        @self.pt.single_timeit
        def write_output():
            if not self.config.args.perform_fit:
                return
            self.output.output(self.solver.fit, self.solver.errors)

            #self.output.write_lammps(self.solver.fit)
            #self.output.write_errors(self.solver.errors)
        write_output()
