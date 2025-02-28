from GRSlib.converters.lammps_base import LammpsBase, _extract_compute_np
import numpy as np

class LammpsPace(LammpsBase):

    def __init__(self, name, pt, config):
        super().__init__(name, pt, config)
        self._data = {}
        self._i = 0
        self._lmp = None
        self._row_index = 0
        self.pt.check_lammps()

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
        numtypes = len(self.config.sections['CONSTRAINT'].type_map)

        # everything is handled by LAMMPS compute pace (similar format as compute snap)

        if not self.config.sections['BASIS'].bikflag:
            base_pace = "compute pace all pace coupling_coefficients.yace 0 0"
        elif (self.config.sections['BASIS'].bikflag and not self.config.sections['BASIS'].dgradflag):
            base_pace = "compute pace all pace coupling_coefficients.yace 1 0"
        elif (self.config.sections['BASIS'].bikflag and self.config.sections['BASIS'].dgradflag):
            base_pace = "compute pace all pace coupling_coefficients.yace 1 1"
        self._lmp.command(base_pace)

    def _collect_lammps_single(self):
        num_atoms = self._data["NumAtoms"]
        num_types = self.config.sections['ACE'].numtypes

        nrows_pace =
        ncols_pace = 
        lmp_pace = _extract_compute_np(self._lmp, "pace", 0, 2, (nrows_pace, ncols_pace))

        if (np.isinf(lmp_pace)).any() or (np.isnan(lmp_pace)).any():
            self.pt.single_print('WARNING! Applying np.nan_to_num()')
            lmp_pace = np.nan_to_num(lmp_pace)
        if (np.isinf(lmp_pace)).any() or (np.isnan(lmp_pace)).any():
            raise ValueError('Nan in computed data of file')

        return lmp_pace
        