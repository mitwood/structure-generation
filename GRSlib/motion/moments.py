import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap, random
from functools import partial
import lammps, lammps.mliap
from lammps.mliap.loader import *

@partial(jit, static_argnums=(0))
def first_moment(self, current_desc, target_desc):
    current_avg = jnp.average(current_desc, axis=0)
    target_avg = jnp.average(target_desc, axis=0)
    tst_residual = jnp.sum(jnp.nan_to_num(jnp.abs(current_avg-target_avg)))
    is_zero = jnp.array(jnp.isclose(tst_residual,jnp.zeros(tst_residual.shape)),dtype=int)
    bonus=-jnp.sum(is_zero*config.sections['SCORING'].moment_bonus[0])
    tst_residual_final = tst_residual*config.sections['SCORING'].moments_coeff[0] + bonus #MAE + bonus
    return tst_residual_final

@partial(jit, static_argnums=(0))
def second_moment(self, current_desc, target_desc):
    current_std = jnp.std(current_desc, axis=0)
    target_std = jnp.std(target_desc, axis=0)
    tst_residual = jnp.sum(jnp.nan_to_num(jnp.abs(current_std-target_std)))
    is_zero = jnp.array(jnp.isclose(tst_residual,jnp.zeros(tst_residual.shape)),dtype=int)
    bonus=-jnp.sum(is_zero*config.sections['SCORING'].moment_bonus[1])
    tst_residual_final = tst_residual*config.sections['SCORING'].moments_coeff[1] + bonus #MAE + bonus
    return tst_residual_final

@partial(jit, static_argnums=(0))
def third_moment(self, current_desc, target_desc):
    #Showing my work for Pearsons skew = (3(mean-median)/stdev)
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
    bonus=-jnp.sum(is_zero*config.sections['SCORING'].moment_bonus[2])
    tst_residual_final = tst_residual*config.sections['SCORING'].moments_coeff[2] + bonus #MAE + bonus
    return tst_residual_final

@partial(jit, static_argnums=(0))
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
    bonus=-jnp.sum(is_zero*config.sections['SCORING'].moment_bonus[3])
    tst_residual_final = tst_residual*config.sections['SCORING'].moments_coeff[3] + bonus #MAE + bonus
    return tst_residual_final


