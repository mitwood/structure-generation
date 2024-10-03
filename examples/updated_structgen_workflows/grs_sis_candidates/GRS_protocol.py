import sys

#-----------------------------------------------------------------------
#initialize ACE fingerprints
from lammps_ACE_basis import *
from opt_tools import *
# define the ACE descriptor set
elements = ['Cr','Fe']
z_of_type = {1:24,2:26}
# number of bonds in fingerprints (descriptor rank)
ranks = [1,2,3]
# angular character of fingerprints
lmax = [0,3,3]
# minimum l per rank (for trimming descriptor list)
lmin = [0,0,0]
# radial character of fingerprints
nmax = [8,1,1]

#NOTE uncomment to generate coupling_coefficients.yace file if not there yet
#make_ACE_functions(elements,ranks,lmax,lmin,nmax)
#-----------------------------------------------------------------------

global_var_weight = 0.00005

#-----------------------------------------------------------------------
# set target structure (From file)
from opt_tools import *
#NOTE can be built from ASE atoms object using function from jmgoff & cmmulle
#descriptors_target_1,descriptors_target_2 = build_target(start)
# OR average and variance can be read in directly as numpy arrays
descriptors_target_1 = np.load('target_descriptors.npy')
descriptors_target_2 = np.load('target_var_descriptors.npy')
#-----------------------------------------------------------------------

# GRS lammps interface
#-----------------------------------------------------------------------
import jax.numpy as np
import numpy as vnp #import regular numpy for 'random' functions
from jax import grad, jit, vmap
from jax import random
from functools import partial
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import lammps
import lammps.mliap
from lammps.mliap.loader import *
import pandas as pd
import copy
import pickle
import os
import shutil
import random

#count descriptors from yace file
n_descs=get_desc_count('coupling_coefficients.yace')
elems=get_desc_count('coupling_coefficients.yace',return_elems=True)
nelements = len(elems)

data_path="./StructureDump"
cross_weight=-0.001000 #closeness to target as a joint distribution min/max just flip this sign (pos being minimize).
self_weight=-0.001000 #closeness to target within an atomic configuration. min/max just flip this sign (pos being minimize). 

# set relative strength of core repulsions vs GSQS effective potential
soft_strength=2.0
ml_strength=1.0

# total number of configurations
n_totconfig=1001
# max and minimum sizes for cells
cell_multiples = [(1,1,1),(1,1,2),(1,2,2),(2,2,2),(2,2,3),(2,3,3)]
cell_multiples_fcc = [(1,1,1),(1,1,2),(1,2,2),(2,2,2)]
cell_size_bcc = [int(2* np.prod(np.array(celli))) for celli in cell_multiples]
cell_size_fcc = [int(4* np.prod(np.array(celli))) for celli in cell_multiples_fcc]
maxcellsize=max([max(cell_size_bcc),max(cell_size_fcc)])+1
mincellsize=min([min(cell_size_bcc),min(cell_size_fcc)])
#  mask for your descriptors if you want to make a GRS effective potential on fewer (including all of them here)
mask = list(range(n_descs))
target=vnp.zeros((len(mask)))

# fraction to multiply supercells by to introduce random variations in volume (e.g., a value of 0.1 adds 0.1*np.random.rand()*a to cell vector a, 0.1*np.random.rand()*b to cell vector b, 0.1*np.random.rand()*c to cell vector c
global_volume_variation = 0.0

#composition in dictionary form: e.g. ( W:0.5, Be:0.5 )
target_comps = {'Cr':0.49,'Fe':0.51}
# extra objective function variables to target exact matches of descriptors 
q1_wt=0.0
q1_cross_wt=0.0
q2_wt=0.0
q2_cross_wt=0.0
# flag to perform minimization with variable cell lengths
box_rlx = False
# flag to run at temperature first
run_temp = True
# flag to randomize run length 
rand_run = True
# flag to use fire style minimization (good for phase transitions & large minimizations)
fire_min = False
# flag to use linestyle quadratic minimization
line_min = True
# flag to randomize the step size in CG or other minimization algorithm
random_move_size=True
# flag to dump the full relaxation trajectory of each GRS (can generate a lot of data)
dump_relax=False
# flag to use randomized compositions for elements in the dictionary: target_comps = {'Cr':1.0 }
randomize_comps=False
if not fire_min and box_rlx:
    min_typ_global = 'box'
else:
    min_typ_global = 'min'

class GRSModel:
    def __init__(self, n_elements, n_descriptors_tot, mask):
        self.mask=mask
        self.n_descriptors=n_descriptors_tot
        self.n_descriptors_keep=len(mask)*n_elements
        self.n_elements=n_elements
        self.n_params=1
        self.sum_of_products=vnp.zeros((self.n_descriptors_keep,self.n_descriptors_keep))
        self.sum=vnp.zeros((self.n_descriptors_keep,))
        self.sumsq=vnp.zeros((self.n_descriptors_keep,))
        self.q_count=0
        self.first_moment_grad=grad(self.first_moment)
        self.V_grad=grad(self.V)
        self.K_self=self_weight
        self.K_cross=cross_weight
        self.first_mom_weight_cross = q1_cross_wt
        self.first_mom_weight = q1_wt
        self.second_mom_weight_cross = q2_cross_wt
        self.second_mom_weight = q2_wt
        self.whitening=np.identity(self.n_descriptors_keep)
        self.mode=  "update"
        self.data=[]

    def set_mode_update(self):
        self.mode="update"

    def set_mode_run(self):
        self.mode="run"

    def update(self,d):
        if self.n_elements > 1:
            dft=d.flatten()
        else:
            dft = d
        self.q_count += dft.shape[0]
        self.sum+=vnp.sum(dft,axis=0)
        self.sumsq+=vnp.sum(dft*dft,axis=0)
        self.set_mode_run()

    # \mathcal{M}_1^{intra} contribution to Q_{GRS}^{intra}
    @partial(jit, static_argnums=(0))
    def first_moment_cross(self, descriptors):
        avgs = self.sum/self.q_count
        abs_diffs_1 = np.abs(avgs - descriptors_target_1)
        abs_diffs_1 = np.array(abs_diffs_1)
        abs_diffs_1 = np.nan_to_num(abs_diffs_1)
        is_zero = np.isclose(abs_diffs_1,np.zeros(abs_diffs_1.shape))
        is_zero = np.array(is_zero,dtype=int)
        bonus=-np.sum(is_zero*self.first_mom_weight_cross)
        tst_residual_1 = np.sum(abs_diffs_1) +bonus
        return tst_residual_1

    # \mathcal{M}_2^{intra} contribution to Q_{GRS}^{intra}
    @partial(jit, static_argnums=(0))
    def second_moment_cross(self, descriptors):
        vrs = self.sumsq/self.q_count 
        abs_diffs_2 = np.abs(vrs - descriptors_target_2)
        abs_diffs_2 = np.array(abs_diffs_2)
        abs_diffs_2 = np.nan_to_num(abs_diffs_2)
        is_zero = np.isclose(abs_diffs_2,np.zeros(abs_diffs_2.shape))
        is_zero = np.array(is_zero,dtype=int)
        bonus=-np.sum(is_zero*self.second_mom_weight_cross)
        tst_residual_2 = np.sum(abs_diffs_2) +bonus
        return tst_residual_2

    # \mathcal{M}_1 contribution to Q_{GRS}
    @partial(jit, static_argnums=(0))
    def first_moment(self, descriptors):
        avgs = np.average(descriptors,axis=0)
        abs_diffs_1 = np.abs(avgs - descriptors_target_1)
        abs_diffs_1 = np.array(abs_diffs_1)
        abs_diffs_1 = np.nan_to_num(abs_diffs_1)
        is_zero = np.isclose(abs_diffs_1,np.zeros(abs_diffs_1.shape))
        is_zero = np.array(is_zero,dtype=int)
        bonus=-np.sum(is_zero*self.first_mom_weight)
        tst_residual_1 = np.sum(abs_diffs_1) +bonus
        return tst_residual_1

    # \mathcal{M}_2 contribution to Q_{GRS}
    @partial(jit, static_argnums=(0))
    def second_moment(self, descriptors):
        vrs = np.var(descriptors,axis=0)
        abs_diffs_2 = np.abs(vrs - descriptors_target_2)
        abs_diffs_2 = np.array(abs_diffs_2)
        abs_diffs_2 = np.nan_to_num(abs_diffs_2)
        is_zero = np.isclose(abs_diffs_2,np.zeros(abs_diffs_2.shape))
        is_zero = np.array(is_zero,dtype=int)
        bonus=-np.sum(is_zero*self.second_mom_weight)
        tst_residual_2 = np.sum(abs_diffs_2) +bonus
        return tst_residual_2

    #GRS effective potential (combining various contributions)
    @partial(jit, static_argnums=(0))
    def V(self,descriptors,weights=[1.0,global_var_weight]):
        if self.n_elements >= 1:
            descriptors_flt = descriptors.flatten()
        else:
            descriptors_flt = descriptors
        vi = ((weights[0]*self.first_moment(descriptors_flt)) + (weights[1]*self.second_moment(descriptors_flt)))
        vj = ((weights[0]*self.first_moment_cross(descriptors_flt)) + (weights[1]*self.second_moment_cross(descriptors_flt)))
        return self.K_self*vi + self.K_cross*vj

    def __call__(self, elems, descriptors, beta, energy):
        self.last_descriptors=descriptors.copy()
        if self.mode=="run":
            b=descriptors[:,self.mask]
            ener=self.V(b)
            energy[:]=0
            energy[0]=ener
            b=self.V_grad(b)
            if not np.all(np.isfinite(b)):
                print("GRAD ERROR!")

            beta[:,:]=0
            beta[:,self.mask]=b

        if self.mode=="update":
            b=descriptors[:,self.mask]
            self.update(b)

class GRSSampler:
    def __init__(self, model, before_loading_init):
        self.model=model
        self.lmp = lammps.lammps(cmdargs=['-screen','none'])
        lammps.mliap.activate_mliappy(self.lmp)
        self.lmp.commands_string(before_loading_init)
        lammps.mliap.load_model(em)

    def update_model(self):
        self.model.set_mode_update()
        self.lmp.commands_string("variable etot equal etotal")
        self.lmp.commands_string("variable ptot equal press")
        self.lmp.commands_string("variable pairp equal epair")
        self.lmp.commands_string("variable numat equal atoms")
        self.lmp.commands_string("run 0")
        self.lmp.commands_string('print "${etot} ${pairp} ${ptot} ${numat} " append Summary.dat screen no')
        self.model.set_mode_run()

    def run(self,cmd=None):
        if cmd==None:
            self.lmp.commands_string("run 0")
        else:
            self.lmp.commands_string(cmd)

#begin the actual optimization loop
try:
    shutil.rmtree(data_path)
except:
    pass
try:
    os.mkdir(data_path)
except:
    pass

#initialize lammps python module and GSQS
lmpi = lammps.lammps(cmdargs=['-screen','none'])
activate_mliappy(lmpi)
em=GRSModel(nelements,n_descs,mask=mask)

# 0th candidate using internal bulk_template (defaults to ASE atoms object low-energy structure)
template = bulk_template(elems[0],desired_size=(1,1,1),volfrac=1.0,cubic=True)
g = internal_generate_cell(0,desired_size=vnp.random.choice(range(mincellsize,maxcellsize)),template=template,desired_comps=target_comps,use_template=template,min_typ=min_typ_global,soft_strength=soft_strength)
sampler=GRSSampler(em,g)
sampler.update_model()

# make comprehensive composition sampling candidates using algorithm for "generating derivative superstructures"
#   from icet
template_sis_all = bulk_sis_template(elems,cellmaxmult=6,crystal_types=['bcc'],lattice_constants=[2.88])
template_sis_fcc_all = bulk_sis_template(elems,cellmaxmult=6,crystal_types=['fcc'],lattice_constants=[3.56])
used_sis_fcc = []
used_sis = []
remaining_sis_fcc = list(range(len(template_sis_fcc_all)))
remaining_sis = list(range(len(template_sis_all)))

i=1
while i <= n_totconfig:
    print(i,"/",n_totconfig,"Using indicies :",mask)
    if len(remaining_sis_fcc) == 0:
        remaining_sis_fcc = list(range(len(template_sis_fcc_all)))
    if len(remaining_sis) == 0:
        remaining_sis = list(range(len(template_sis_all)))
    sizes = cell_multiples
    sizes_fcc= cell_multiples_fcc
    ind_select = vnp.random.choice(range(len(sizes)))
    desired_size=sizes[ind_select]
    ind_select_fcc = vnp.random.choice(range(len(sizes_fcc)))
    desired_size_fcc = sizes_fcc[ind_select_fcc]
    cell_scale = 1 + (vnp.random.uniform(-1,1)*global_volume_variation)
    #other conventional cubic candidate structures defined by 'template' variable
    template = bulk_template(elems[0],desired_size,volfrac=cell_scale,cubic=True)
    template_fcc = bulk_template(elems[0],desired_size,volfrac=cell_scale,override_lat='fcc',override_a=3.56,cubic=True)
    #NOTE 'ind_select_*' are random integers to select pregenerated candidates the two lines below ensure that
    #  pregenerated SIS candidates are selected such that exhaustive compositional sampling is enforced
    ind_select_sis = vnp.random.choice(remaining_sis) 
    ind_select_sis_fcc = vnp.random.choice(remaining_sis_fcc)
    #NOTE uncomment to instead select random sis candidates without enforcing exhaustive compositional sampling
    #ind_select_sis = vnp.random.choice(range(len(template_sis_all)))
    #ind_select_sis_fcc = vnp.random.choice(range(len(template_sis_fcc_all)))
    template_sis = template_sis_all[ind_select_sis]
    template_sis_fcc = template_sis_fcc_all[ind_select_sis_fcc]
    # specify candidate types accessed and respective probabilities for candidates
    candidate_typ = vnp.random.choice([0,1,2,3,4],p=[0.05,0.05,0.0,0.45,0.45])
    if not randomize_comps:
        if candidate_typ == 0:
            g = internal_generate_cell(i,desired_size=vnp.random.choice(range(mincellsize,maxcellsize)),template=template,desired_comps=target_comps,use_template=template,min_typ=min_typ_global,soft_strength=soft_strength)
        if candidate_typ == 1:
            g = internal_generate_cell(i,desired_size=vnp.random.choice(range(mincellsize,maxcellsize)),template=template_fcc,desired_comps=target_comps,use_template=template_fcc,min_typ=min_typ_global,soft_strength=soft_strength)
        elif candidate_typ == 2:
            g = internal_generate_cell(i,desired_size=vnp.random.choice(range(mincellsize,maxcellsize)),template=None,desired_comps=target_comps,use_template=None,min_typ=min_typ_global,soft_strength=soft_strength)
        elif candidate_typ == 3:
            idx = remaining_sis.index(ind_select_sis)
            del remaining_sis[idx]
            g = internal_generate_cell(i,desired_size=vnp.random.choice(range(mincellsize,maxcellsize)),template=template_sis,desired_comps=target_comps,use_template=template_sis,min_typ=min_typ_global,soft_strength=soft_strength,sis_freeze=True)
        elif candidate_typ == 4:
            idx = remaining_sis_fcc.index(ind_select_sis_fcc)
            del remaining_sis_fcc[idx]
            g = internal_generate_cell(i,desired_size=vnp.random.choice(range(mincellsize,maxcellsize)),template=template_sis_fcc,desired_comps=target_comps,use_template=template_sis_fcc,min_typ=min_typ_global,soft_strength=soft_strength,sis_freeze=True)
        
    else:
        target_comps_rnd = rand_comp(target_comps)
        if candidate_typ == 0:
            g = internal_generate_cell(i,desired_size=vnp.random.choice(range(mincellsize,maxcellsize)),template=template,desired_comps=target_comps_rnd,use_template=template,min_typ=min_typ_global,soft_strength=soft_strength)
        if candidate_typ == 1:
            g = internal_generate_cell(i,desired_size=vnp.random.choice(range(mincellsize,maxcellsize)),template=template_fcc,desired_comps=target_comps_rnd,use_template=template_fcc,min_typ=min_typ_global,soft_strength=soft_strength)
        elif candidate_typ == 2:
            g = internal_generate_cell(i,desired_size=vnp.random.choice(range(mincellsize,maxcellsize)),template=None,desired_comps=target_comps_rnd,use_template=None,min_typ=min_typ_global,soft_strength=soft_strength)
        elif candidate_typ == 3:
            idx = remaining_sis.index(ind_select_sis)
            del remaining_sis[idx]
            g = internal_generate_cell(i,desired_size=vnp.random.choice(range(mincellsize,maxcellsize)),template=None,desired_comps=target_comps,use_template=None,min_typ=min_typ_global,soft_strength=soft_strength,sis_freeze=False)
        elif candidate_typ == 4:
            idx = remaining_sis_fcc.index(ind_select_sis_fcc)
            del remaining_sis_fcc[idx]
            g = internal_generate_cell(i,desired_size=vnp.random.choice(range(mincellsize,maxcellsize)),template=None,desired_comps=target_comps,use_template=None,min_typ=min_typ_global,soft_strength=soft_strength,sis_freeze=True)
    sampler=GRSSampler(em,g)
    em.K_cross=cross_weight
    em.K_self=self_weight
    if run_temp:
        sampler.run('velocity all create 300.0 4928459 loop geom')
        sampler.run("fix  a1  all nve")
        if rand_run:
            run_nsteps = vnp.random.choice([0,10,100],p=[0.33,0.43,0.24])
        elif not rand_run:
            run_nsteps = 10
        sampler.run("run %d " % run_nsteps)
        sampler.run("unfix  a1")
    if fire_min:
        sampler.run("min_style  fire")
        sampler.run("min_modify integrator eulerexplicit tmax 10.0 tmin 0.0 delaystep 5 dtgrow 1.1 dtshrink 0.5 alpha0 0.1 alphashrink 0.99 vdfmax 100000 halfstepback no initialdelay no")
    if line_min:
        sampler.run("min_style  cg")
        if random_move_size:
            move_size = vnp.random.choice([0.0005,0.005,0.05]) 
        elif not random_move_size:
            move_size = 0.05
        sampler.run("min_modify  dmax %f line backtrack"%move_size)

    if dump_relax:
        sampler.run("dump    a%d all xyz 1 dump_%d.xyz" %(i,i))
    sampler.run("minimize 1e-12 1e-12 10000 100000")
    if dump_relax:
        sampler.run("undump    a%d"%i)
    sampler.run("write_data %s/sample.%i.dat " % (data_path,i) )
    sampler.update_model()

    i+=1
