from models.wanglei6 import WangLei6

powerlist_list = [
        [[1,1],[1,0,1]],
        [[1,1],[1,0,1]],
        [[1,1],[1,0,1]],
        ]

num_features_list = [
        [4],
        [4],
        [4, 4],
        ]

def modifyconfig_and_getnn(config, bench_id):
    nonlinear = 'x^3'
    do_BN = True
    NP = 0
    if bench_id==1:
        do_BN=False
    if bench_id==2:
        NP=1
        do_BN=False
    rbm = WangLei6(input_shape=tuple(config['hamiltonian']['size']), num_features=num_features_list[bench_id],
            dtype0='complex128', powerlist=powerlist_list[bench_id], stride=1,
            usesum=False, nonlinear=nonlinear, NP=NP, NC=1, do_BN=do_BN)
    return rbm
