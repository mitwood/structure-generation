from os import path, listdir, stat
import numpy as np
from random import random, seed, shuffle
from copy import copy
from ase.io import read, write, lammpsdata


class Convert:

    def __init__(self, name, pt, config):
        self.pt = pt #ParallelTools()
        self.config = config #Config()

    def ase_to_lammps(self,data):
        """
        Takes in an ase.Atoms object and writes a lammps-data, returns the file name
        """
        fname = '%s.lammps-data' % data.symbols
        write(fname, data, format='lammps-data', masses=True)
        return fname

    def lammps_to_ase(self,data):
        """
        Takes in a lammps-data file and returns an ase.Atoms object
        """
        ase_data = read(target_fname,format='lammps-data')
        return ase_data

    def lammps_ace(self,data):
        """
        Takes in an lammps-data file and converts to descriptors in the ACE basis set
        """

    def lammps_snap(self,data):
        """
        Takes in an lammps-data file and converts to descriptors in the SNAP basis set
        """

    def lammps_custom(self,data):
        """
        Takes in an lammps-data file and converts to descriptors in a custom basis set
        """

