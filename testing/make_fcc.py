from ase.io import read,write
from ase.build import bulk

atoms = bulk('W','fcc',a=4.15,cubic=True)

write('fcc.data',atoms,format='lammps-data')
