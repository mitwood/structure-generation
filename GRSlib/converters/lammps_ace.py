from GRSlib.converters.convert import Convert
from GRSlib.converters.lammps_base import Base, _extract_compute_np
import numpy as np

class Ace(Convert):

    def __init__(self, name, pt, config):
        super().__init__(name, pt, config)
        self._data = {}
        self._i = 0
        self._lmp = None
        self._row_index = 0
        self.pt.check_lammps()
        self.make_input()

    def _prepare_lammps(self):
        self._set_structure()
        # this is super clean when there is only one value per key, needs reworking
        # self._set_variables(**_lammps_variables(config.sections["ACE"].__dict__))

        self._lmp.command(f"variable rcutfac equal {max(self.config.sections['BASIS'].rcutfac)}")
        self._lmp.command(f"pair_style 	zero {max(self.config.sections['BASIS'].rcutfac)}")
        self._lmp.command("pair_coeff 	* *")
        self._set_computes()
        self._set_neighbor_list()

    def _set_computes(self):
        # everything is handled by LAMMPS compute pace (similar format as compute snap)

        numtypes = len(self.config.sections['BASIS'].elements)
        if not self.config.sections['BASIS'].bikflag:
            base_pace = "compute pace all pace coupling_coefficients.yace 0 0"
        elif (self.config.sections['BASIS'].bikflag and not self.config.sections['BASIS'].dgradflag):
            base_pace = "compute pace all pace coupling_coefficients.yace 1 0"
        elif (self.config.sections['BASIS'].bikflag and self.config.sections['BASIS'].dgradflag):
            base_pace = "compute pace all pace coupling_coefficients.yace 1 1"
        self._lmp.command(base_pace)

    def make_input(self):
        Base._initialize_lammps(self)
        Base._set_structure(self)
        self._lmp.command(f"read_data {self.config.sections['TARGET'].start_fname}")
        self._prepare_lammps()
        self._set_computes()
        Base._run_lammps(self)

    def _collect_lammps_single(self):
        num_atoms = self._data["NumAtoms"]
        num_types = self.config.sections['ACE'].numtypes
        with open('coupling_coefficients.yace','r') as readcoeff:
            lines = readcoeff.readlines()
            elemline = [line for line in lines if 'elements' in line][0]
            elemstr = elemline.split(':')[-1]
            elemstr2 = elemstr.replace('[','')
            elemstr3 = elemstr2.replace(']','')
            elemstr4 = elemstr3.replace(',','')
            elems = elemstr4.split()
            nelements = len(elems)
            desclines = [line for line in lines if 'mu0' in line]
        
        ncols_pace = int(len(desclines)/nelements)
        nrows_pace = num_atoms
        lmp_pace = _extract_compute_np(self._lmp, "pace", 0, 2, (nrows_pace, ncols_pace))

        if (np.isinf(lmp_pace)).any() or (np.isnan(lmp_pace)).any():
            self.pt.single_print('WARNING! Applying np.nan_to_num()')
            lmp_pace = np.nan_to_num(lmp_pace)
        if (np.isinf(lmp_pace)).any() or (np.isnan(lmp_pace)).any():
            raise ValueError('Nan in computed data of file')

        return lmp_pace
        