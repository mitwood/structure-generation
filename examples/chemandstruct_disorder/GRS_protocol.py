# grs library imports
from opt_tools import *
from lammps_ACE_basis import *
# other imports
from jax import grad, jit, vmap
from jax import random
import jax.numpy as np
# import regular numpy for numpy.random functions
import numpy as vnp
from functools import partial
import lammps
import lammps.mliap
from lammps.mliap.loader import *
import os
import shutil
import sys

#-----------------------------------------------------------------------
# Define the descriptor set to be used, from '.yace' descriptor file 
#   for lammps. These can be generated within if FitSNAP is in your 
#   PYTHONPATH
#-----------------------------------------------------------------------

# define the possible element types
elements = ['Nb','Ni']
# number of bonds in fingerprints (descriptor rank)
#ranks = [1,2,3]
# angular character of fingerprints
#lmax = [0,3,3]
# minimum l per rank (for trimming descriptor list)
#lmin = [0,0,0]
# radial character of fingerprints
#nmax = [3,1,1]
#make_ACE_functions(elements,ranks,lmax,lmin,nmax)
# 'make_ACE_functions' will generate a 'coupling_coefficients.yace' file
# containing ACE descriptor hyperparameters

#-----------------------------------------------------------------------
#count descriptors from yace file
n_descs=get_desc_count('coupling_coefficients.yace')
elems=get_desc_count('coupling_coefficients.yace',return_elems=True)
nelements = len(elems)
#-----------------------------------------------------------------------

#-----------------------------------------------------------------------
# set target structure (From file read by ASE)
#av,var = build_target(sys.argv[1],save_all=True)
av = np.load('target_descriptors.npy')
var = np.load('target_var_descriptors.npy')
#for now, use first and second moments of target 
#   distribution (average and variance)
descriptors_target_1 = av
descriptors_target_2 = var
#-----------------------------------------------------------------------


# extra objective function variables to target exact matches of descriptors
q1_wt=0.0
q1_cross_wt=0.0
q2_wt=0.0
q2_cross_wt=0.0

cross_weight=1.000000 #closeness to target as a joint distribution min/max just flip this sign (pos being minimize).
self_weight=1.000000 #closeness to target within an atomic configuration. min/max just flip this sign (pos being minimize). 

soft_strength=1.0
ml_strength=1.0


#-----------------------------------------------------------------------
# define starting points for GRS structures (a.k.a. candidate types)
#-----------------------------------------------------------------------
#composition in dictionary form: e.g. { W:0.5, Be:0.5 }
target_comps = {'Ni':0.88,'Nb':0.12}

volume_variation = 0.4 # fraction to vary candidate cell volume by  

# total number of configurations
n_totconfig = int(sys.argv[2])

# where to save structures
data_path="./OptimizedGRS"

# variables for nearly crystalline candidates
#cell_multiples = [(1,1,2),(2,2,2),(2,3,3), (3,3,3)]
cell_multiples = [(3,3,4)]
cell_multiples_fcc = [(2,3,3)]
cell_size_bcc = [int(2* np.prod(np.array(celli))) for celli in cell_multiples]
cell_size_fcc = [int(4* np.prod(np.array(celli))) for celli in cell_multiples_fcc]

#variables for hermite normal form supercells (randomly populated cells)
maxcellsize=73
mincellsize=40#min([min(cell_size_bcc),min(cell_size_fcc)])

# candidate sampling probabilities:

candidate_typs = [0,1,2] # corresponding hull-like supercell, fcc-like supercells, and random hermite normal form cells, respectively
#   NOTE hull-like supercell is supercell resembling the ground state crystal structure
#   for Ta - this is BCC, for carbon, diamond, etc.
candidate_probs = [1.0,0.00,0.0] #probability to sample a candidate type from 'candidate_typs'

#-----------------------------------------------------------------------
# define parameters for minimization of GRS loss function
#-----------------------------------------------------------------------
run_temp = True # flag to run at temperature/perform annealing first
box_rlx = False # flag to perform minimization with variable cell lengths
fire_min = False# flag to use fire style minimization

line_min = True # flag to use linestyle quadratic minimization
random_move_size = True #flag to use variable minimization step sizes
#min_step_sizes = [0.05,0.1, 0.5] #if random_move_size=True - the step sizes used in line_min
min_step_sizes = [0.05,0.1, 0.5] #if random_move_size=True - the step sizes used in line_min
dump_relax = False #flag to dump full minimization trajectory of GRS from LAMMPS
randomize_comps = False # flag to use unconstrained compositions (only consider for 2+ element systems)
if not fire_min and box_rlx:
    min_typ_global = 'box'
else:
    min_typ_global = 'min'


#-----------------------------------------------------------------------
# Define the GRS model & effective potential
#-----------------------------------------------------------------------
class GRSModel:
    def __init__(self, n_elements, n_descriptors_tot, mask=None):
        if mask != None:
            self.mask=mask
        else:
            self.mask = vnp.array(range(n_descriptors_tot))
        self.n_descriptors=n_descriptors_tot
        self.n_descriptors_keep=len(self.mask)*n_elements
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

    #match for mean descriptor value for the set of structures
    @partial(jit, static_argnums=(0))
    def first_moment_cross(self, descriptors):
        avgs = self.sum/self.q_count
        abs_diffs_1 = np.abs(avgs - descriptors_target_1)
        abs_diffs_1 = np.array(abs_diffs_1)
        is_zero = np.isclose(abs_diffs_1,np.zeros(abs_diffs_1.shape))
        is_zero = np.array(is_zero,dtype=int)
        bonus=-np.sum(is_zero*self.first_mom_weight_cross)
        tst_residual_1 = np.average(abs_diffs_1) +bonus
        return tst_residual_1

    #match for variance of descriptor value for the set of structures
    @partial(jit, static_argnums=(0))
    def second_moment_cross(self, descriptors):
        vrs = self.sumsq/self.q_count 
        abs_diffs_2 = np.abs(vrs - descriptors_target_2)
        abs_diffs_2 = np.array(abs_diffs_2)
        is_zero = np.isclose(abs_diffs_2,np.zeros(abs_diffs_2.shape))
        is_zero = np.array(is_zero,dtype=int)
        bonus=-np.sum(is_zero*self.second_mom_weight_cross)
        tst_residual_2 = np.average(abs_diffs_2) +bonus
        cpy = tst_residual_2.copy()
        return tst_residual_2

    # match of mean descriptor values within the current structure only
    @partial(jit, static_argnums=(0))
    def first_moment(self, descriptors):
        avgs = np.average(descriptors,axis=0)
        abs_diffs_1 = np.abs(avgs - descriptors_target_1)
        abs_diffs_1 = np.array(abs_diffs_1)
        is_zero = np.isclose(abs_diffs_1,np.zeros(abs_diffs_1.shape))
        is_zero = np.array(is_zero,dtype=int)
        bonus=-np.sum(is_zero*self.first_mom_weight)
        tst_residual_1 = np.average(abs_diffs_1) +bonus
        return tst_residual_1

    # match of descriptor variance values within the current structure only
    @partial(jit, static_argnums=(0))
    def second_moment(self, descriptors):
        vrs = np.var(descriptors,axis=0)
        abs_diffs_2 = np.abs(vrs - descriptors_target_2)
        abs_diffs_2 = np.array(abs_diffs_2)
        is_zero = np.isclose(abs_diffs_2,np.zeros(abs_diffs_2.shape))
        is_zero = np.array(is_zero,dtype=int)
        bonus=-np.sum(is_zero*self.second_mom_weight)
        tst_residual_2 = np.average(abs_diffs_2) +bonus
        return tst_residual_2

    #effective potential
    @partial(jit, static_argnums=(0))
    def V(self,descriptors,weights=[1.0,0.00005]):
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
        lammps.mliap.load_model(grs_modl)

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

try:
    shutil.rmtree(data_path)
except:
    pass
try:
    os.mkdir(data_path)
except:
    pass

lmpi = lammps.lammps(cmdargs=['-screen','none'])
activate_mliappy(lmpi)
grs_modl=GRSModel(nelements,n_descs)
# use internal structure generator to build candidates
template = bulk_template(elems[0],desired_size=cell_multiples[-1],volfrac=1.0,cubic=True)
template.rattle(stdev=0.1)
g = internal_generate_cell(0,desired_size=vnp.random.choice(range(mincellsize,maxcellsize)),template=template,desired_comps=target_comps,use_template=template,min_typ=min_typ_global,soft_strength=soft_strength)
print ('model',grs_modl)
sampler=GRSSampler(grs_modl,g)
sampler.update_model()

i=1
while i <= n_totconfig:
    print(i,"/",n_totconfig)
    sizes = cell_multiples
    sizes_fcc= cell_multiples_fcc
    ind_select = vnp.random.choice(range(len(sizes)))
    desired_size=sizes[ind_select]
    ind_select_fcc = vnp.random.choice(range(len(sizes_fcc)))
    desired_size_fcc = sizes_fcc[ind_select_fcc]
    cell_scale = 1 + (vnp.random.uniform(0,1)*volume_variation)
    template = bulk_template(elems[0],desired_size,volfrac=cell_scale,cubic=True)
    #template.rattle(stdev=0.5)
    template_fcc = bulk_template(elems[1],desired_size,volfrac=cell_scale,cubic=True)
    candidate_typ = vnp.random.choice(candidate_typs,p=candidate_probs)
    if not randomize_comps:
        if candidate_typ == 0:
            g = internal_generate_cell(i,desired_size=vnp.random.choice(range(mincellsize,maxcellsize)),template=template,desired_comps=target_comps,use_template=template,min_typ=min_typ_global,soft_strength=soft_strength)
        if candidate_typ == 1:
            g = internal_generate_cell(i,desired_size=vnp.random.choice(range(mincellsize,maxcellsize)),template=template_fcc,desired_comps=target_comps,use_template=template_fcc,min_typ=min_typ_global,soft_strength=soft_strength)
        elif candidate_typ == 2:
            g = internal_generate_cell(i,desired_size=vnp.random.choice(range(mincellsize,maxcellsize)),template=None,desired_comps=target_comps,use_template=None,min_typ=min_typ_global,soft_strength=soft_strength,volfrac=cell_scale)
    else:
        target_comps_rnd = rand_comp(target_comps)
        if candidate_typ == 0:
            g = internal_generate_cell(i,desired_size=vnp.random.choice(range(mincellsize,maxcellsize)),template=template,desired_comps=target_comps_rnd,use_template=template,min_typ=min_typ_global,soft_strength=soft_strength)
        if candidate_typ == 1:
            g = internal_generate_cell(i,desired_size=vnp.random.choice(range(mincellsize,maxcellsize)),template=template_fcc,desired_comps=target_comps_rnd,use_template=template_fcc,min_typ=min_typ_global,soft_strength=soft_strength)
        elif candidate_typ == 2:
            g = internal_generate_cell(i,desired_size=vnp.random.choice(range(mincellsize,maxcellsize)),template=None,desired_comps=target_comps_rnd,use_template=None,min_typ=min_typ_global,soft_strength=soft_strength,volfrac=cell_scale)
    sampler=GRSSampler(grs_modl,g)
    grs_modl.K_cross=cross_weight
    grs_modl.K_self=self_weight
    if run_temp:
        sampler.run("fix 1mc all atom/swap 1 2 29494 300.0 ke yes types 1 2")
        sampler.run("fix 1rt all nvt temp 10000.0 1.0 0.001") 
        sampler.run("run 1000")
        sampler.run("unfix 1mc")
        sampler.run("fix 2mc all atom/swap 3 2 29494 100.0 ke yes types 1 2")
        sampler.run("run 500")
        sampler.run("unfix 1rt")
    
    if fire_min:
        sampler.run("min_style  fire")
        sampler.run("min_modify integrator eulerexplicit tmax 10.0 tmin 0.0 delaystep 5 dtgrow 1.1 dtshrink 0.5 alpha0 0.1 alphashrink 0.99 vdfmax 10000 halfstepback no initialdelay no")
    if line_min:
        sampler.run("min_style  cg")
        if random_move_size:
            move_size = vnp.random.choice(min_step_sizes)
        elif not random_move_size:
            move_size = 0.05
        sampler.run("min_modify  dmax %f line backtrack"%move_size)

    if dump_relax:
        sampler.run("dump    a%d all xyz 1 dump_%d.xyz" %(i,i))
    sampler.run("minimize 1e-6 1e-6 3000 30000")
    if dump_relax:
        sampler.run("undump    a%d"%i)
    sampler.run("write_data %s/sample.%i.dat " % (data_path,i) )
    sampler.update_model()

    i+=1
