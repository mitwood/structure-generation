from GRSlib.parallel_tools import ParallelTools
from GRSlib.io.input import Config
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
        self.basis = basis(self.config.sections["BASIS"].descriptor, self.pt, self.config) if "BASIS" in self.config.sections else None
        #self.convert = convert(self.pt,self.config)

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

    def write_output(self):
        @self.pt.single_timeit
        def write_output():
            if not self.config.args.perform_fit:
                return
            self.output.output(self.solver.fit, self.solver.errors)

            #self.output.write_lammps(self.solver.fit)
            #self.output.write_errors(self.solver.errors)
        write_output()
