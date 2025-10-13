from ase.io import read,write
from ase import Atoms,Atom
from ase.ga.utilities import closest_distances_generator, CellBounds
from ase.ga.startgenerator import StartGenerator
from ase.data import atomic_numbers, atomic_names, atomic_masses, covalent_radii
from ase.neighborlist import primitive_neighbor_list
import numpy as np
import random
from collections import Counter

# Lowest level functions that can be used to modif structures, inherited class not needed since scoring will happen in
# motion/genetic.py. This collection of functions is mostly to avoid clustter and massive files where more abstract 
# things are happening.

class GenMoves():
    
    def atom_count(atoms,config):
        #From the user input of density ratio, figure out the bounds of number of atoms to add/remove
        #Roll dice on the +/- bounds and adjust the ase.atom object that is read in.
        cell = atoms.get_cell() #Carry over the cell size from the supecell
        cell_lenx, cell_leny, cell_lenz = atoms.cell.cellpar()[0],atoms.cell.cellpar()[1],atoms.cell.cellpar()[2]
        natoms = len(atoms.get_atomic_numbers())
        chem_comp = list(atoms.symbols) #atoms.get_chemical_formula(mode='all')
        elements = list(Counter(chem_comp).keys()) #same as set(chem_comp)
        ele_counts = list(Counter(chem_comp).items()) #counts per unique element
        new_positions = atoms.get_positions()
        max_deladd = np.abs(natoms - round(natoms*config.sections["GENETIC"].density_ratio))
        change_count = random.randint(-max_deladd,max_deladd)
        if change_count < 0:
            for i in range(abs(change_count)):
                del atoms[random.randint(0,len(atoms.get_atomic_numbers())-1)] #Needs to be atoms.get_atomic_numbers not natoms
            chem_comp = list(atoms.symbols) #atoms.get_chemical_formula(mode='all')
            elements = list(Counter(chem_comp).keys()) #same as set(chem_comp)
            ele_counts = list(Counter(chem_comp).items()) #counts per unique element
            sym_comp = ""
            for ele in range(len(elements)):
                sym_comp += elements[ele]+str(ele_counts[ele][1])
            new_positions = atoms.get_positions()
            new_atoms = Atoms(symbols=sym_comp,positions=new_positions, cell=cell, pbc=[1,1,1])
        else:
            for i in range(abs(change_count)):
                tmp_x,tmp_y,tmp_z = np.random.uniform(low=0.0,high=cell_lenx),np.random.uniform(low=0.0,high=cell_leny),np.random.uniform(low=0.0,high=cell_lenz)
                new_positions = np.append(new_positions,[[tmp_x,tmp_y,tmp_z]], axis=0)
                chem_comp.append(random.choice(list(set(chem_comp))))
            sym_comp = ""
            elements = list(Counter(chem_comp).keys()) #same as set(chem_comp)
            ele_counts = list(Counter(chem_comp).items()) #counts per unique element
            for ele in range(len(elements)):
                sym_comp += elements[ele]+str(ele_counts[ele][1])
            new_atoms = Atoms(symbols=sym_comp,positions=new_positions, cell=cell, pbc=[1,1,1])
        return new_atoms

    def volume(atoms,config):
        #From the user input of density ratio, take cube root and roll dice for lx,ly,lz,alpha,beta,gamma
        cell = atoms.get_cell() #Carry over the cell size from the supecell
        scaled_positions = atoms.get_scaled_positions()
        atom_symbols = atoms.symbols
        scale_matrix = np.eye(3,dtype=float)*0.5*(config.sections["GENETIC"].density_ratio)**(1./3.)
        new_cell = np.matmul(cell,scale_matrix)
        new_positions = np.matmul(scaled_positions,new_cell)
        new_atoms = Atoms(atom_symbols,positions=new_positions, cell=new_cell, pbc=[1,1,1])
        return new_atoms

    def perturb(atoms,config):
        new_cell = atoms.get_cell() #Carry over the cell size from the supecell
        atom_symbols = atoms.symbols
        new_positions = atoms.get_positions()
        atom_length = (1/(3.)**(1./2.))*(len(atoms.numbers())/atoms.get_volume())*(1./3.) # Linear distance from average atomic volume, becomes max displacement distance
        change_count = random.randint(1,round(len(atoms.numbers())/2)) #Perturb up to one-half the atom positions
        for i in range(change_count):
            pertub_id = random.randint(0,len(atoms.numbers())-1)
            new_positions[id][0] += np.random.uniform(low=-atom_length,high=atom_length)
            new_positions[id][1] += np.random.uniform(low=-atom_length,high=atom_length)
            new_positions[id][2] += np.random.uniform(low=-atom_length,high=atom_length)

        new_atoms = Atoms(atom_symbols,positions=new_positions, cell=new_cell, pbc=[1,1,1])
        return new_atoms

    def change_ele(atoms,config):

        chem_comp = atoms.get_chemical_formula(mode='all')
        elements = Counter(chem_comp).keys() #same as set(chem_comp)
        ele_counts = Counter(chem_comp).items()/len(atoms.numbers()) #counts per unique element

        itr = 0
        while itr == 0 or any([icomp == 0.0 for icomp in ele_counts]):
            fraction = np.random.rand()
            pert_inds = np.random.choice(range(len(atoms)),size=int(len(atoms)*fraction) )
            new_atoms = atoms.copy()
            for pert_ind in pert_inds:
                flip_ind = np.random.randint(0,len(atoms))
                flip_current = new_atoms[flip_ind].symbol
                excluded = [typ for typ in elements if typ != flip_current]
                flip_to_ind = np.random.randint(0,len(excluded))
                flip_to_type = excluded[flip_to_ind]
                new_atoms[flip_ind].symbol = flip_to_type
            elements = Counter(chem_comp).keys() #same as set(chem_comp)
            ele_counts = Counter(chem_comp).items()/len(atoms.numbers()) #counts per unique element
            itr += 1

        return new_atoms

    def minimize(atoms,config):
        #Do nothing because it will be relaxed upon returning to genetic.py
        return atoms    