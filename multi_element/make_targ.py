from ase.build import bulk
from ase.io import read,write
from ase import Atoms,Atom

#atoms = bulk('MgO','rocksalt',a=4.3,cubic=True)
atoms = bulk('Mg','fcc',a=4.3,cubic=True)
syms = [s for s in atoms.symbols]
#atoms.symbols = ['Ca', 'O', 'Mg', 'O', 'Mg', 'O', 'Mg', 'O']
atoms.symbols = ['Ca', 'Mg', 'Mg', 'Mg']
write('conventional.cif',atoms)
write('conventional.data',atoms,format='lammps-data',masses=True)
#['Mg', 'O', 'Mg', 'O', 'Mg', 'O', 'Mg', 'O']
big_cell = atoms * (1,1,1)

write('supercell_target.cif',big_cell)
write('supercell_target.data',big_cell,format='lammps-data',masses=True)


satoms = bulk('Mg','fcc',a=4.3,cubic=True)
satoms.symbols = ['Ca', 'Ca', 'Ca', 'Mg']
starting_struct = satoms * (1,1,2)

write('starting.cif',starting_struct)
write('starting.data',starting_struct,format='lammps-data',masses=True)

