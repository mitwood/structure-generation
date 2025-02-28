from os import path, listdir, stat
import numpy as np
from random import random, seed, shuffle
from fitsnap3lib.units.units import convert
from copy import copy


class Convert:

    def __init__(self, pt, config):
        self.pt = pt #ParallelTools()
        self.config = config #Config()

    def ase_to_lammps(self):
        """
        Takes in an ase.Atoms object and converts to lammps-data
        """

    def lammps_to_descriptors(self):
        """
        Takes in an lammps-data object and converts to descriptors in the defined basis set
        """

