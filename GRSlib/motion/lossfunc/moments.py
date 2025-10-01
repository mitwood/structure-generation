from GRSlib.motion.scoring import Scoring
import jax.numpy as jnp
import numpy as np
from jax import grad, jit
from functools import partial

class Moments(Scoring):
    def __init__(self, *args): #pt, config, target_desc, prior_desc):
        self.pt, self.config, descriptors = args
        self.target_desc = descriptors.get('target',None).copy() 
        self.prior_desc = descriptors.get('prior',None).copy()
        self.n_descriptors = np.shape(self.target_desc)[1]
        self.mask = list(range(self.n_descriptors))
        self.n_params = 1 #Variables LAMMPS needs to know about
        self.n_elements = self.config.sections['BASIS'].numtypes #Variables LAMMPS needs to know about
        self.loss_ff_grad = grad(self.construct_loss)
        self.grad_loss = grad(self.construct_loss)
        self.mode = "update"

    def __call__(self, *args):
        #The arguments that this function brings in are super improtant and are expected by LAMMPS MLIAP package.
        #They are (elems, current_desc, beta, energy)
        #Integer values of LAMMPS atom types are in elems
        #Descriptors as a per-atom array into current_desc.
        #Per-atom forces are expected for beta
        #Per-atom energy is expected for energy, need to do some testing if per-atom values can be reported back.
        if self.mode=="score":     
            elems, current_desc, beta, energy = args
            self.n_atoms = np.shape(current_desc)[0]

            #TODO Explain
            score = self.construct_loss(current_desc, self.target_desc)
            energy[:] = 0
            energy[0] = float(self.config.sections["SCORING"].strength_target)*score #Scaled score (energy) between current and target
            forces = self.grad_loss(current_desc, self.target_desc) #Forces between current and target
            beta[:,:]= 0
            beta[:,self.mask] = float(self.config.sections["SCORING"].strength_target)*forces #Scaled forces between current and target

            #TODO Explain
            score = self.construct_loss(current_desc, self.prior_desc)
#            energy[0] += float(self.config.sections["SCORING"].strength_prior)*score #Scaled score (energy) between current and prior
#            print("     Target, Prior Scores: ", energy[0], score)
            forces = self.grad_loss(current_desc, self.prior_desc) #Forces between current and prior structures
            beta[:,self.mask] += float(self.config.sections["SCORING"].strength_prior)*forces #Scaled forces between current and prior

        elif self.mode=="update":
            self.update(args)
            beta = self.grad_loss(self.target_desc, self.target_desc)


    def set_mode_update(self):
        self.mode="update"

    def set_mode_score(self):
        self.mode="score"

    def update(self,*args):
        pt, config, descriptors = args[0]
        self.target_desc = descriptors.get('target',None).copy()
        self.prior_desc = descriptors.get('prior',None).copy()
        self.n_descriptors = np.shape(self.target_desc)[1]
        self.mask = list(range(self.n_descriptors))

        if self.n_elements > 1:
            self.current_desc = self.current_desc.flatten()
            self.target_desc = self.target_desc.flatten()
            self.prior_desc = self.prior_desc.flatten()
        self.mode = "score"

    @partial(jit, static_argnums=(0,))
    def construct_loss(self, current_desc, target_desc):
        #This needs to be a dynamic call like is done in descriptor calcs 
        loss_ff = 0
        set_of_moments = []

        if (any(x == 'mean' for x in self.config.sections['SCORING'].moments)):
            first_mom = self.first_moment(current_desc, target_desc)
        else:
            first_mom = None
            
        if (any(x == 'stdev' for x in self.config.sections['SCORING'].moments)):
            second_mom = self.second_moment(current_desc, target_desc)
        else:
            second_mom = None

        if (any(x == 'skew' for x in self.config.sections['SCORING'].moments)):
            third_mom = self.third_moment(current_desc, target_desc)
        else:
            third_mom = None

        if (any(x == 'kurt' for x in self.config.sections['SCORING'].moments)):
            fourth_mom = self.fourth_moment(current_desc, target_desc)
        else:
            fourth_mom = None

        for item in [first_mom, second_mom, third_mom, fourth_mom]: 
            if item != None:
                    set_of_moments.append(item)

        for item in set_of_moments:
             loss_ff += item
        return loss_ff

    @partial(jit, static_argnums=(0,))
    def first_moment(self, current_desc, target_desc):
        current_avg = jnp.average(current_desc, axis=0)
        target_avg = jnp.average(target_desc, axis=0)
        tst_residual = jnp.sum(jnp.nan_to_num(jnp.abs(current_avg-target_avg)))
        is_zero = jnp.array(jnp.isclose(tst_residual,jnp.zeros(tst_residual.shape)),dtype=int)
        bonus = -jnp.sum(is_zero*(float(self.config.sections['SCORING'].moments_bonus[0])))
        tst_residual_final = tst_residual*float(self.config.sections['SCORING'].moments_coeff[0]) + bonus #MAE + bonus
        return tst_residual_final

    @partial(jit, static_argnums=(0,))
    def second_moment(self, current_desc, target_desc):
        current_std = jnp.std(current_desc, axis=0)
        target_std = jnp.std(target_desc, axis=0)
        tst_residual = jnp.sum(jnp.nan_to_num(jnp.abs(current_std-target_std)))
        is_zero = jnp.array(jnp.isclose(tst_residual,jnp.zeros(tst_residual.shape)),dtype=int)
        bonus = -jnp.sum(is_zero*float(self.config.sections['SCORING'].moments_bonus[1]))
        tst_residual_final = tst_residual*float(self.config.sections['SCORING'].moments_coeff[1]) + bonus #MAE + bonus
        return tst_residual_final

    @partial(jit, static_argnums=(0,))
    def third_moment(self, current_desc, target_desc):
        #Showing my work for Pearsons skew = (3(mean-median)/stdev))
        current_avg = jnp.average(current_desc, axis=0)
        target_avg = jnp.average(target_desc, axis=0)
        current_std = jnp.std(current_desc, axis=0)
        target_std = jnp.std(target_desc, axis=0)
        current_med = jnp.median(current_desc, axis=0)
        target_med = jnp.median(target_desc, axis=0)

        current_skew = 3.0*(current_avg-current_med)/current_std
        target_skew = 3.0*(target_avg-target_med)/target_std

        tst_residual = jnp.sum(jnp.nan_to_num(jnp.abs(current_skew-target_skew)))
        is_zero = jnp.array(jnp.isclose(tst_residual,jnp.zeros(tst_residual.shape)),dtype=int)
        bonus = -jnp.sum(is_zero*float(self.config.sections['SCORING'].moments_bonus[2]))
        tst_residual_final = tst_residual*float(self.config.sections['SCORING'].moments_coeff[2]) + bonus #MAE + bonus
        return tst_residual_final

    @partial(jit, static_argnums=(0,))
    def fourth_moment(self, current_desc, target_desc):
        #Showing my work for Kurtosis = Avg(z^4.0)-3 where z=(x-avg(x))/stdev(x)
        current_avg = jnp.average(current_desc, axis=0)
        target_avg = jnp.average(target_desc, axis=0)
        current_std = jnp.std(current_desc, axis=0)
        target_std = jnp.std(target_desc, axis=0)

        current_kurt = jnp.average(((current_desc-current_avg)/current_std)**4.0)-3.0 
        target_kurt = jnp.average(((target_desc-target_avg)/target_std)**4.0)-3.0 

        tst_residual = jnp.sum(jnp.nan_to_num(jnp.abs(current_kurt-target_kurt)))
        is_zero = jnp.array(jnp.isclose(tst_residual,jnp.zeros(tst_residual.shape)),dtype=int)
        bonus = -jnp.sum(is_zero*float(self.config.sections['SCORING'].moments_bonus[3]))
        tst_residual_final = tst_residual*float(self.config.sections['SCORING'].moments_coeff[3]) + bonus #MAE + bonus
        return tst_residual_final
