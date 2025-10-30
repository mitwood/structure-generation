import spglib as spg
from ase.ga.utilities import closest_distances_generator, CellBounds
from ase.io import read,write
from ase.data import atomic_numbers
from ase.lattice.cubic import *
from ase.lattice.tetragonal import *
from ase.lattice.orthorhombic import *
from ase.lattice.monoclinic import *
from ase.lattice.triclinic import *
from ase.lattice.hexagonal import *
from ase.neighborlist import *
from ase.build import bulk,make_supercell
from ase import Atoms,Atom
import numpy as np
import random

class ASETools():
    #Helper functions :
    def diamond_struct():
        """A factory for creating diamond lattices."""
        xtal_name = 'diamond'
        bravais_basis = [[0, 0, 0], [0.25, 0.25, 0.25]]

    def GraphiteFactory():
        bravais_basis=[[0, 0, 0], [0.5, 0.5, 0.5]]

    def optimal_bond_to_latparam(optimal_bond_length,atoms,lattice_params,tol=0.05):
        verbose = False
        tst_atoms = atoms.copy()
        fac = 1.1
        cut = (optimal_bond_length + tol)
        dists = primitive_neighbor_list('d',pbc=tst_atoms.pbc,positions=tst_atoms.positions ,cell=tst_atoms.get_cell(),cutoff=cut)
        while len(dists) == 0:
            cut *=fac
            dists = primitive_neighbor_list('d',pbc=tst_atoms.pbc,positions=tst_atoms.positions ,cell=tst_atoms.get_cell(),cutoff=cut)
            #print('num dists',cut,len(dists))
        current_bond_length = np.average(dists)
        #print(current_bond_length,optimal_bond_length)
        while current_bond_length > optimal_bond_length + tol or current_bond_length < optimal_bond_length - tol:
            new_ds = primitive_neighbor_list('d',pbc=tst_atoms.pbc,positions=tst_atoms.positions ,cell=tst_atoms.get_cell(),cutoff=optimal_bond_length + tol)
            current_bond_length = np.average(new_ds) #TODO nan to num this for 0 length new_ds array?
            if current_bond_length > optimal_bond_length:
                cell = tst_atoms.get_cell()
                cell_vec_sizes = [np.linalg.norm(v) for v in cell]
                mx = max(cell_vec_sizes)
                mx_ind = cell_vec_sizes.index(mx)
                isize = cell_vec_sizes[mx_ind]
                iratio = optimal_bond_length/isize
                istep = 0.002 #(  iratio * 0.1 ) #1/5 th of the size
                assert istep < 1., "check your step size %f" % istep
                new_cell = (1 - istep) * cell
                tst_atoms.set_cell(new_cell,scale_atoms=True)
            #elif current_bond_length < optimal_bond_length :
            else:
                cell = tst_atoms.get_cell()
                cell_vec_sizes = [np.linalg.norm(v) for v in cell]
                mx = max(cell_vec_sizes)
                mx_ind = cell_vec_sizes.index(mx)
                isize = cell_vec_sizes[mx_ind]
                iratio = optimal_bond_length/isize
                istep = 0.002 #(  iratio * 0.1 ) #1/5 th of the size
                assert istep < 1., "check your step size %f" % istep
                new_cell = (1 + istep) * cell
                tst_atoms.set_cell(new_cell,scale_atoms=True)
            tmpcut = optimal_bond_length + tol
            mx_itr = 20
            itri=0
            while np.isnan(current_bond_length) and itri < mx_itr:
                new_ds = primitive_neighbor_list('d',pbc=tst_atoms.pbc,positions=tst_atoms.positions ,cell=tst_atoms.get_cell(),cutoff=tmpcut)
                current_bond_length = current_bond_length = np.average(new_ds)
                tmpcut += 0.1
                itri +=1
            if verbose:
                print ('istep',istep,current_bond_length, optimal_bond_length,tst_atoms.get_cell())
            #new_ds = primitive_neighbor_list('d',pbc=tst_atoms.pbc,positions=tst_atoms.positions ,cell=tst_atoms.get_cell(),cutoff=optimal_bond_length + tol)
            #current_bond_length = np.average(new_ds)
        
        return tst_atoms


    def get_cell_type(atoms,parent_only = False):
        cell = atoms.get_cell()
        scpos = atoms.get_scaled_positions()
        if parent_only:
            numbers = [atoms.numbers[0]]*len(atoms)
        else:
            numbers = atoms.numbers
        spgcell = (cell,scpos,numbers)
        spacegroup = spg.get_spacegroup(spgcell, symprec=1e-5, angle_tolerance=-1.0, symbol_type=0)

        spacegroup_to_tuple = {
    'Pm-3m (221)' : ('cubic','sc'),
    'Im-3m (229)' : ('cubic','bcc'),
    'Fm-3m (225)' : ('cubic','fcc'),
    'R-3m (166)' : ('cubic','diamond'),
    'Fd-3m (227)' : ('cubic','diamond'),
    'P4/mmm (123)' : ('tetragonal','st'),
    'I4/mmm (139)' : ('tetragonal','ct'),
    'Pmmm (47)' : ('orthorhombic','so'),
    'Cmmm (65)' : ('orthorhombic','baco'),
    'Fmmm (69)' : ('orthorhombic','fco'),
    'Immm (71)' : ('orthorhombic','boco'),
    'P2/m (10)' : ('monoclinic','sm'),
    'C2/m (12)' : ('monoclinic','bcm'),
    'P-1 (2)' : ('triclinic','t'),
    'P1 (1)' : ('triclinic','t'),
    'P6/mmm (191)' : ('hexagonal','h'),
    'P6_3/mmc (194)' : ('hexagonal','hcp'),
    'P6_3/mmc (194)' : ('hexagonal','hgr')
        }
        try:
            result = spacegroup_to_tuple[spacegroup]
        except KeyError:
            result = ('triclinic','t')
        return result

    def get_primitive_cell(atoms):
        cell = atoms.get_cell()
        scpos = atoms.get_scaled_positions()
        numbers = atoms.numbers
        spgcell = (cell,scpos,numbers)
        symmetry = spg.get_symmetry(spgcell, symprec=1e-5)
        lattice, scaled_positions, numbers = spg.standardize_cell(spgcell, to_primitive=True, no_idealize=False, symprec=1e-5)
        new_prim = Atoms(numbers)
        new_prim.set_cell(lattice)
        new_prim.set_scaled_positions(scaled_positions)
        new_prim.set_pbc(True)
        return new_prim

    # quick function to get supercell from primitive cell for any system
    #TODO. Hermite Normal Form supercells in like Gus Hart has
    # would be the most comprehensive way to do this (would contain cubic and primitive multiples)
    def get_any_supercell(atoms,min_natoms,max_natoms,num=1):
        natoms_in = len(atoms)
        rough_n3 = int(natoms_in**(1/3))
        tups_over = [p for p in itertools.product(range(1,rough_n3+4),range(1,rough_n3+4),range(1,rough_n3+4))]
        natoms_over = [natoms_in * p[0]*p[1]*p[2] for p in tups_over]
        tups = [tup for itup,tup in enumerate(tups_over) if natoms_over[itup] < max_natoms and natoms_over[itup] > min_natoms]
        natoms = [natoms_in * p[0]*p[1]*p[2] for p in tups]
        sizes = sorted(list(set(natoms)))
        if num ==1:
            random_sc_mult_i = np.random.choice(range(len(tups)))
            random_sc_mult = tups[random_sc_mult_i]
            scell = atoms*random_sc_mult
            return scell
        else:
            #random_sc_mult_i = np.random.choice(range(len(tups)),num,replace=False)
            #random_sc_mults = [tups[random_sc_mult_ii] for random_sc_mult_ii in random_sc_mult_i]
            #scells = [atoms*random_sc_mult for random_sc_mult in random_sc_mults]
            grouped = {sz:[tup for tup in tups if (tup[0]*tup[1]*tup[2]*natoms_in) == sz] for sz in sizes}
            this_size = np.random.choice(sizes)
            try:
                random_sc_mult_i=np.random.choice(range(len(grouped[this_size])),num,replace=False)
                if len(random_sc_mult_i) != num:
                    random_sc_mult_i=np.random.choice(range(len(grouped[this_size])),num)
            except:
                random_sc_mult_i=np.random.choice(range(len(grouped[this_size])),num)
            random_sc_mults = [grouped[this_size][random_sc_mult_ii] for random_sc_mult_ii in random_sc_mult_i]
            scells = [atoms*random_sc_mult for random_sc_mult in random_sc_mults]
            print('obtained, target',len(scells),num)
            return scells
    #TODO limit where this can be applied. I am not sure if it will work with hexagonal phases
    #   and others that Coreen has been working on implementing. As far as I see, it only works
    #   for cubic crystals.
    def get_cube_supercell(atoms, min_natoms, max_natoms):
        cell = atoms.get_cell()
        scpos = atoms.get_scaled_positions()
        numbers = atoms.numbers
        while True:
            selected_natoms = random.randint(int(min_natoms), int(max_natoms))
            closest_cube = round(float(selected_natoms/len(numbers))**(1./3.))
            if (len(numbers)*closest_cube**3.0 < int(max_natoms)) or (len(numbers)*closest_cube**3.0 > int(min_natoms)):
                break
            else:
                print("Closest cube of primitive cell out of min/max, rerolling")
        replicated = make_supercell(atoms, P=[[closest_cube,0,0],[0,closest_cube,0],[0,0,closest_cube]],  wrap=True, order='atom-major', tol=1e-05) #ase.build
        return replicated
        
    def get_random_pos(atoms, min_natoms, max_natoms, ele_type):
        cell = atoms.get_cell() #Carry over the cell size from the supecell
        cell_lenx, cell_leny, cell_lenz = atoms.cell.cellpar()[0],atoms.cell.cellpar()[1],atoms.cell.cellpar()[2]
        selected_natoms = random.randint(int(min_natoms), int(max_natoms))
        new_positions = []
        for atom in range(selected_natoms):
            tmp_x,tmp_y,tmp_z = np.random.uniform(low=0.0,high=cell_lenx),np.random.uniform(low=0.0,high=cell_leny),np.random.uniform(low=0.0,high=cell_lenz)
            new_positions.append((tmp_x,tmp_y,tmp_z))
        return Atoms(ele_type+str(selected_natoms),positions=new_positions, cell=cell, pbc=[1,1,1])
        
    def lattice_func(pltup):
        valid_tups = [ ('cubic','sc'),
    ('cubic','bcc'),
    ('cubic','fcc'),
    ('cubic','diamond'),
    ('tetragonal','st'),
    ('tetragonal','ct'),
    ('orthorhombic','so'),
    ('orthorhombic','baco'),
    ('orthorhombic','fco'),
    ('orthorhombic','boco'),
    ('monoclinic','sm'),
    ('monoclinic','bcm'),
    ('triclinic','t'),
    ('hexagonal','h'),
    ('hexagonal','hcp'),
    ('hexagonal','hgr')]
        all_tup_str = ' '.join( '("%s" , "%s")'%B for B in valid_tups)
        assert pltup in valid_tups, "(%s,%s) is not a valid structure tuple, please enter one of the following: %s" % (pltup + (all_tup_str,))
        # cubic structures
        if pltup == ('cubic','sc'):
            return SimpleCubic
        elif pltup == ('cubic','bcc'):
            return BodyCenteredCubic
        elif pltup == ('cubic','fcc'):
            return FaceCenteredCubic
        elif pltup == ('cubic','diamond'):
            return Diamond
        # tetragonal structures
        elif pltup == ('tetragonal','st'):
            return SimpleTetragonal
        elif pltup == ('tetragonal','ct'):
            return CenteredTetragonal
        # orthorhombic structures
        elif pltup == ('orthorhombic','so'):
            return SimpleOrthorhombic
        elif pltup == ('orthorhombic','baco'):
            return BaseCenteredOrthorhombic
        elif pltup == ('orthorhombic','fco'):
            return FaceCenteredOrthorhombic
        elif pltup == ('orthorhombic','boco'):
            return BodyCenteredOrthorhombic
        # monoclinic structures
        elif pltup == ('monoclinic','sm'):
            return SimpleMonoclinic
        elif pltup == ('monoclinic','bcm'):
            return BaseCenteredMonoclinic
        # triclinic structure
        elif pltup == ('triclinic','t'):
            return Triclinic
        # hexagonal structures
        elif pltup == ('hexagonal','h'):
            return Hexagonal
        elif pltup == ('hexagonal','hcp'):
            return HexagonalClosedPacked
        elif pltup == ('hexagonal','hgr'):
            return Graphite

    def get_prim_structs(elem_list, multiatom=False):
        all_ats =[]
        for elem in elem_list:
            atis = bulk(elem)
            suggested_bond_len = 2*np.average(natural_cutoffs(atis))
            #print (elem,get_cell_type(atis),suggested_bond_len)
            for tup in valid_tups:
                this_func = lattice_func(tup)
                atoms =this_func(size=(1,1,1), symbol=elem, pbc=(1,1,1), latticeconstant=lattice_params_schema[tup])
                prim_atoms = get_primitive_cell(atoms)
                starting_atoms = optimal_bond_to_latparam(optimal_bond_length=suggested_bond_len,atoms=prim_atoms,lattice_params=None,tol=0.05)
                all_ats.append(starting_atoms)
        return all_ats

    def compress_expand(strct, a, prefix=None, axes = [0,1,2],stepsize=0.03, nsteps = 2 ):
        this_a = a # lattice constant
        compressed_expanded = {}
        steps_up = np.linspace(this_a, this_a + (stepsize*nsteps), nsteps )
        steps_above = steps_up[1:]
        steps_down = np.linspace(this_a - (stepsize*nsteps), this_a, nsteps + 1 )
        steps_below = steps_down[:-1]
        all_steps = np.append(steps_below,steps_above)

        cell = strct.get_cell()
        scaled_pos = strct.get_scaled_positions()
        for stepind,istep in enumerate(all_steps):
            new_atoms = Atoms(strct.symbols)
            new_cell = cell.copy()
            for axis in axes:
                new_cell[axis] +=  cell[axis] - istep
            new_atoms.set_cell(new_cell)
            new_atoms.set_scaled_positions(scaled_pos)
            new_atoms.set_pbc(True)
            #compressed_expanded[icrystal][istrct].append(new_atoms)
            if prefix == None:
                prefix_out = 'eos_%d' % stepind
            else:
                prefix_out = prefix + '_eos_%d' % stepind
            write('%s.cif' % prefix_out  , new_atoms)

#These live outside the defined functions
    allowed_lattice_params = {
    ('cubic','sc'):[('a',)],
    ('cubic','bcc'):[('a',)],
    ('cubic','fcc'):[('a',)],
    ('cubic','diamond'):[('a',)],
    ('tetragonal','st'):[('a','c/a')],
    ('tetragonal','ct'):[('a','c/a')],
    ('orthorhombic','so'):[('a','b/a','c/a')],
    ('orthorhombic','baco'):[('a','b/a','c/a')],
    ('orthorhombic','fco'):[('a','b/a','c/a')],
    ('orthorhombic','boco'):[('a','b/a','c/a')],
    ('monoclinic','sm'):[('a', 'b/a', 'c/a', 'alpha')],
    ('monoclinic','bcm'):[('a', 'b/a', 'c/a', 'alpha')],
    ('triclinic','t'):[('a', 'b/a', 'c/a', 'alpha', 'beta', 'gamma')],
    ('hexagonal','h'):[('a','c/a')],
    ('hexagonal','hcp'):[('a','c/a')],
    ('hexagonal','hgr'):[('a','c/a')]
    }

    lattice_params_schema = {
    ('cubic','sc'):2.0,#{'a':2.0},
    ('cubic','bcc'):4.0,#{'a':4.0},
    ('cubic','fcc'):4.0,#{'a':4.0},
    ('cubic','diamond'):4.0,#{'a':4.0},
    ('tetragonal','st'):{'a':4.0,'c/a':4/3},
    ('tetragonal','ct'):{'a':3.0,'c/a':4/3},
    ('orthorhombic','so'):{'a':2.0, 'b/a':1.2, 'c/a':1.3},
    ('orthorhombic','baco'):{'a':4.0, 'b/a':1.2, 'c/a':1.3},
    ('orthorhombic','fco'):{'a':4.0, 'b/a':1.2, 'c/a':1.3},
    ('orthorhombic','boco'):{'a':4.0, 'b/a':1.2, 'c/a':1.3},
    ('monoclinic','sm'):{'a':4.0, 'b/a':1.2, 'c/a':1.3, 'alpha':70 },
    ('monoclinic','bcm'):{'a':4.0, 'b/a':1.2, 'c/a':1.3, 'alpha':70 },
    ('triclinic','t'):{'a':4.0, 'b/a':1.2, 'c/a':1.3, 'alpha':70., 'beta':40., 'gamma':100. },
    ('hexagonal','h'):{'a':2.8,'c/a':1.5},
    ('hexagonal','hcp'):{'a':2.8,'c/a':1.5},
    ('hexagonal','hgr'): {'a':2.8,'c/a':1.5},
    }

    valid_tups = [ ('cubic','sc'),
    ('cubic','bcc'),
    ('cubic','fcc'),
    ('cubic','diamond'),
    ('tetragonal','st'),
    ('tetragonal','ct'),
    ('orthorhombic','so'),
    ('orthorhombic','baco'),
    ('orthorhombic','fco'),
    ('orthorhombic','boco'),
    ('monoclinic','sm'),
    ('monoclinic','bcm'),
    ('triclinic','t'),
    ('hexagonal','h'),
    ('hexagonal','hcp')
    ]
    
    bravais_phases = {valid_tup[1]:valid_tup for valid_tup in valid_tups}

    elem_list = [
    'C','Si',
    'Fe','Co','Ni','Cu','Zn',
    'Zr','Nb','Mo',
    'Ru','Pd','Ag' ,
    'Hf','Ta','W',
    'Ir','Pt','Au',
    ]

