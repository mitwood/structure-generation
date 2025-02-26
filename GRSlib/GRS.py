from GRS.parallel_tools import ParallelTools
from GRS.io.input import Config
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
        if self.config.args.verbose:
            self.pt.single_print(f"GRS instance hash: {self.config.hash}")
        # Instantiate other backbone attributes.
        self.basis = basis(self.config.sections["BASIS"].descriptor, self.pt, self.config) if "BASIS" in self.config.sections else None
            #Currently first arg (basis type) will default to ACE, but this could be generalized

#        self.calculator = calculator(self.config.sections["CALCULATOR"].calculator, self.pt, self.config) \
#            if "CALCULATOR" in self.config.sections else None
#        self.solver = solver(self.config.sections["SOLVER"].solver, self.pt, self.config) \
#            if "SOLVER" in self.config.sections else None
#        self.output = output(self.config.sections["OUTFILE"].output_style, self.pt, self.config) \
#            if "OUTFILE" in self.config.sections else None

        self.fit = None
        self.multinode = 0

        # Check LAMMPS version if using nonlinear solvers.
        if (hasattr(self.pt, "lammps_version")):
            if (self.pt.lammps_version < 20220915):
                raise Exception(f"Please upgrade LAMMPS to 2022-09-15 or later to use MLIAP based structure searching.")
          
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


    def FnName(self):
        """
        
        
        """
    
    def scrape_configs(self, delete_scraper: bool = False):
        """
        Scrapes configurations of atoms and creates an instance attribute list of configurations called `data`.
        
        Args:
            delete_scraper: Boolean determining whether the scraper object is deleted or not after scraping. Defaults 
                            to False. Since scraper can retain unwanted memory, we delete it in executable mode.
        """
        @self.pt.single_timeit
        def scrape_configs():
            self.scraper.scrape_groups()
            self.scraper.divvy_up_configs()
            self.data = self.scraper.scrape_configs()
            if delete_scraper:
                del self.scraper
        scrape_configs()

    def process_configs(self, data: list=None, allgather: bool=False, delete_data: bool=False):
        """
        Calculate descriptors for all configurations in the :code:`data` list and stores info in the shared arrays.
        
        Args:
            data: Optional list of data dictionaries to calculate descriptors for. If not supplied, we use the list 
                  owned by this instance.
            allgather: Whether to gather distributed lists to all processes to just to head proc. In some cases, such as 
                       processing configs once and then using that data on multiple procs, we must allgather.
            delete_data: Whether the data list is deleted or not after processing.Since `data` can retain unwanted 
                         memory after processing configs, we delete it in executable mode.
        """

        if data is not None:
            data = data
        elif hasattr(self, "data"):
            data = self.data
        else:
            raise NameError("No list of data dictionaries to process.")

        # Zero distributed index before parallel loop over configs.
        self.calculator.distributed_index = 0

        @self.pt.single_timeit
        def process_configs():
            self.calculator.allocate_per_config(data)
            # Preprocess the configs if nonlinear fitting.
            if (not self.solver.linear):
                if self.config.args.verbose: 
                    self.pt.single_print("Nonlinear solver, preprocessing configs.")
                self.calculator.preprocess_allocate(len(data))
                for i, configuration in enumerate(data):
                    self.calculator.preprocess_configs(configuration, i)
            # Allocate shared memory arrays.
            self.calculator.create_a()
            # Calculate descriptors.
            if (self.solver.linear):
                for i, configuration in enumerate(data):
                    # TODO: Add option to print descriptor calculation progress on single proc.
                    #if (i % 1 == 0):
                    #   self.pt.single_print(i)
                    self.calculator.process_configs(configuration, i)
            else:
                for i, configuration in enumerate(data):
                    self.calculator.process_configs_nonlinear(configuration, i)
            # Delete instance-owned data dictionary to save memory.
            if delete_data and hasattr(self, "data"):
                del self.data
            # Gather distributed lists in `self.pt.fitsnap_dict` to root proc.
            self.calculator.collect_distributed_lists(allgather=allgather)
            # Optional extra steps.
            if self.solver.linear:
                self.calculator.extras()

        process_configs()

    def perform_fit(self):
        """Solve the machine learning problem with descriptors as input and energies/forces/etc as 
           targets"""
        @ self.pt.single_timeit
        def fit():
            if not self.config.args.perform_fit:
                return
            elif self.fit is None:
                if self.solver.linear:
                    self.solver.perform_fit()
                else:
                    # Perform nonlinear fitting on 1 proc only.
                    if(self.pt._rank==0):
                        self.solver.perform_fit()
            else:
                self.solver.fit = self.fit
                
        # If not performing a fit, keep in mind that the `configs` list is None 
        # for nonlinear models. Keep this in mind when performing error 
        # analysis.
        
        def fit_gather():
            self.solver.fit_gather()

        @self.pt.single_timeit
        def error_analysis():
            self.solver.error_analysis()

        fit()
        fit_gather()
        error_analysis()

    def write_output(self):
        @self.pt.single_timeit
        def write_output():
            if not self.config.args.perform_fit:
                return
            self.output.output(self.solver.fit, self.solver.errors)

            #self.output.write_lammps(self.solver.fit)
            #self.output.write_errors(self.solver.errors)
        write_output()
