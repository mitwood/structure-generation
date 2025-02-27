import configparser
import argparse
import sys
from pickle import HIGHEST_PROTOCOL
from GRSlib.io.sections.section_factory import new_section
from pathlib import Path
import random


class Config():
    """ 
    Class for storing input settings in a `config` instance. The `config` instance is first created 
    in `io/output.py`. If given a path to an input script, we use Python's native ConfigParser 
    to parse the settings. If given a nested dictionary, the sections are determined from the 
    first keys and specific settings from the nested keys.

    Args:
        pt: A ParallelTools instance.
        input: Optional input can either be a filename or a dictionary.
        arguments_lst: List of args that can be supplied at the command line.

    Attributes:
        infile: String for optional input filename. Defaults to None.
        indict: Dictionary for optional input dictionary of settings, to replace input file. Defaults 
            to None.
        
    """

    def __init__(self, pt, input=None, arguments_lst: list = []):
        self.pt = pt
        self.input = input
        # Input file (infile) and dictionary (indict) set to None by default and get set in 
        # parse_config.
        self.infile = None
        self.indict = None
        self.default_protocol = HIGHEST_PROTOCOL
        self.args = None
        self._original_config = None
        self.parse_cmdline(arguments_lst=arguments_lst)
        self.sections = {}
        self.parse_config()

        # Generate random 128 bit hash to identify this fit on rank 0.
        if self.pt._rank == 0:
            self.hash = f"{random.getrandbits(128):032x}"
        else:
            self.hash = None

    def parse_cmdline(self, arguments_lst: list = []):
        """ Parse command line args if using executable mode, or a list if using library mode. """
        parser = argparse.ArgumentParser(prog="fitsnap3")
        if (self.input is None):
            parser.add_argument("infile", action="store",
                                help="Input file with bispectrum etc. options")

        # Optional args.
        parser.add_argument("--lammpslog", "-l", action="store_true", dest="lammpslog",
                            help="Write logs from LAMMPS. Logs will appear in current working directory.")
        parser.add_argument("--overwrite", action="store_true", dest="overwrite",
                            help="Allow overwriting existing files")
        parser.add_argument("--verbose", "-v", action="store_true", dest="verbose",
                            default=False, help="Show more detailed information about processing")
        parser.add_argument("--screen", "-sc", action="store_false", dest="screen",
                            help="Print fitsnap output to screen.")
        if arguments_lst:
            # If arg list is not empty we are feeding in arguments with library mode, and args should be parse 
            # according to this list.
            self.args = parser.parse_args(arguments_lst)
        else:
            # If arguments list is empty, we can parse the args like usual with executable mode.
            self.args = parser.parse_args()


    def parse_config(self):
        self._original_config = configparser.ConfigParser(inline_comment_prefixes='#')
        self._original_config.optionxform = str
        if self.input is not None:
            if (isinstance(self.input, str)):
                self.infile = self.input
            elif (isinstance(self.input, dict)):
                self.indict = self.input
        else:
            if not Path(self.args.infile).is_file():
                raise FileNotFoundError("Input file not found")
            self.infile = self.args.infile

        if (self.infile is not None):
            # We have an input file.
            self._original_config.read(self.infile)

            # continue setting up self._original_config
            infile_folder = str(Path(self.infile).parent.absolute())
            file_name = self.infile.split('/')[-1]
            if not Path(infile_folder+'/'+file_name).is_file():
                raise RuntimeError("Input file {} not found in {}", file_name, infile_folder)

    def convert_to_dict(self, original_input=False):
        """
        Convert the current config (settings) object to a dictionary. Note that datatypes may not be preserved.

        Args:
            original_input: optional, set to True to return the original input 

        Returns:
            config_dict: Python dictionary containing the same elements as the original or current config (settings) object.
        """
        if original_input:
            config_dict = {s:dict(self._original_config.items(s)) for s in self._original_config.sections()}
        else:
            config_dict = {s:vars(self.sections[s]) for s in self.sections}
            return config_dict
