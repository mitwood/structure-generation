from fitsnap3lib.lib.sym_ACE.yamlpace_tools.potential import *
from fitsnap3lib.lib.sym_ACE.pa_gen import *
from fitsnap3lib.lib.sym_ACE.tree_method import *
from fitsnap3lib.lib.sym_ACE.clebsch_couple import *
from fitsnap3lib.lib.sym_ACE.wigner_couple import *

# potential parameters. these should match the FitSNAP input
reference_ens = [0.,0.]
# bonds:
bonds=  [(0, 0), (0, 1), (1, 0), (1, 1)]
#  [(H, H), (H, N), (H, O), (N, N), (N, O), (O, O)]

rcutfaci = [4.6,4.5,4.5,4.9]
lmbdai = [0.046,0.045,0.045,0.049]
rcinneri = [0.0]*4
drcinneri = [0.01]*4

exhaust_bonds = bonds.copy()

rcutfac = []
lmbda = []
rcinner = []
drcinner = []

for bond in exhaust_bonds:
	srt_bnd = tuple(sorted(bond))
	idx = bonds.index(srt_bnd)
	rcutfac.append(rcutfaci[idx])
	lmbda.append(lmbdai[idx])
	rcinner.append(rcinneri[idx])
	drcinner.append(drcinneri[idx])
print('rcutfac =',' '.join(str(k) for k in rcutfac))
print('lambda =',' '.join(str(k) for k in lmbda))
print('rcinner =',' '.join(str(k) for k in rcinner))
print('drcinner =',' '.join(str(k) for k in drcinner))

elements=["Ca","Mg"]
ranks = [1, 2, 3, 4]
lmax =  [1, 2, 2, 1]
nmax = [4, 2, 1, 1]
lmin = [0,0,2,1]
L_R=0
M_R=0
nradbase=4

#coupling type for angular spherical harmonics (either is fine)
coupling_type ='cg'
#wigner couplings (instead of Clebsch-Gordan) above

coeffs = None
ldict = {ranki:li for ranki,li in zip(ranks,lmax)}
rankstrlst = ['%s']*len(ranks)
rankstr = ''.join(rankstrlst) % tuple(ranks)
lstrlst = ['%s']*len(ranks)
lstr = ''.join(lstrlst) % tuple(lmax)

#---------------------------------------------------------------------
#coupling type for angular spherical harmonics (either is fine)
coupling_type ='cg'
#wigner couplings (instead of Clebsch-Gordan) above

coeffs = None
ldict = {ranki:li for ranki,li in zip(ranks,lmax)}
rankstrlst = ['%s']*len(ranks)
rankstr = ''.join(rankstrlst) % tuple(ranks)
lstrlst = ['%s']*len(ranks)
lstr = ''.join(lstrlst) % tuple(lmax)



#load or generate generalized coupling coefficients
if coupling_type == 'cg':
    try:
        with open('cg_LR_%d_r%s_lmax%s.pickle' %(L_R,rankstr,lstr),'rb') as handle:
            ccs = pickle.load(handle)
    except FileNotFoundError:
        ccs = get_cg_coupling(ldict,L_R=L_R)
        #store them for later so they don't need to be recalculated
        store_generalized(ccs, coupling_type='cg',L_R=L_R)

elif coupling_type == 'wig':
    try:
        with open('wig_LR_%d_r%s_lmax%s.pickle' %(L_R,rankstr,lstr),'rb') as handle:
            ccs = pickle.load(handle)
    except FileNotFoundError:
        ccs = get_wig_coupling(ldict,L_R=L_R)
        #store them for later so they don't need to be recalculated
        store_generalized(ccs, coupling_type='wig',L_R=L_R)



Apot = AcePot(elements,reference_ens,ranks,nmax,lmax,nradbase,rcutfac,lmbda,rcinner,drcinner,lmin,**{'ccs':ccs[M_R]})
# read the potential file to get expansion coefficients
Apot.write_pot('coupling_coefficients')
