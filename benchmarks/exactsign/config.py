from models.manual_sign import WangLei6
from utils import packnbits_pm, sign_func_from_vec, analyse_exact
from problems import load_hamiltonian

powerlist_list = [
        None,
        ]
num_features_list = [
        [20], [20], [20], [20], [20],
        ]
nonlinear_list_list = [
        ['sinh','none','none'],
        ]

def modifyconfig_and_getnn(config, bench_id):
    is_unitary = False
    momentum = 0
    eta0 = 0.2
    eta1 = 0.2
    NP = 0
    NC = 1
    itype = 'complex128'
    poly_order = 10
    usesum = True
    powerlist = powerlist_list[bench_id]
    num_features = num_features_list[bench_id]
    nonlinear_list = nonlinear_list_list[bench_id]
    soften_gradient = False
    isym = False

    if bench_id==1 or bench_id==3:
        soften_gradient = True
    if bench_id == 5:
        isym =True

    hconfig = config['hamiltonian']
    rbm = WangLei6(sign_func = get_sign_func(hconfig['J2'], hconfig['size']), 
            input_shape=tuple(hconfig['size']), num_features=num_features,
            itype=itype,dtype0=itype, dtype1=itype, powerlist=powerlist,
            usesum=usesum, nonlinear_list=nonlinear_list, poly_order=poly_order, do_BN=False,
            momentum=momentum, eta0=eta0, eta1=eta1, NP=NP, NC=NC,is_unitary=is_unitary,
            soften_gradient = soften_gradient, isym=isym)
    return rbm

def get_sign_func(J2, size):
    # definition of a problem
    h = load_hamiltonian('J1J2', size=size, J2=J2)
    # Exact Results
    H, e0, v0, configs = analyse_exact(h, do_printsign=False)
    return sign_func_from_vec(configs, v0)
