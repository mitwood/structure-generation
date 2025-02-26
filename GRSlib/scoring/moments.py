#placeholder for scoring a generated structure based on the moments of the descriptor distribution
def cost_func_1(l_pace,descriptors_target,atoms):
    descriptors_tst_decomp = l_pace[ : len(atoms), : -1]
    #descriptors_tst = np.sum(descriptors_tst_decomp,axis=0)
    descriptors_tst = np.average(descriptors_tst_decomp,axis=0)
    abs_diffs = [np.abs(ii - kk) for ii,kk in zip(descriptors_tst, descriptors_target)]
    abs_diffs = np.array(abs_diffs)
#    tst_residual = np.sum(abs_diffs)
    tst_residual = np.mean(abs_diffs)
    return tst_residual,abs_diffs

def cost_func_2(l_pace,descriptors_target,atoms):
    descriptors_tst_decomp = l_pace[ : len(atoms), : -1]
    descriptors_tst = np.var(descriptors_tst_decomp,axis=0)
    abs_diffs = [np.abs(ii - kk) for ii,kk in zip(descriptors_tst, descriptors_target)]
    abs_diffs = np.array(abs_diffs)
#    tst_residual = np.sum(abs_diffs)
    tst_residual = np.mean(abs_diffs)
    return tst_residual,abs_diffs
