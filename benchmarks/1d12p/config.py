from models.wanglei3 import WangLei3

powerlist_list = [
        [[1]],
        [[1,1]],
        [[1,0,1]],
        [[1,1,1]],
        [[1],[1,1]],

        [[1],[1,1],[1,0,1]],
        [[1],[1,1],[1,0,1],[1,1,1]],
        [[1],[1,0,1]],
        [[1],[1,1],[1,0,1]],
        [[1],[1,1],[1,0,1]],

        [[1],[1,1],[1,0,1]],
        [[1,1],[1,0,1]],
        [[1,1],[1,0,1]],
        ]

def modifyconfig_and_getnn(config, bench_id):
    nonlinear = 'x^3'
    if bench_id == 9: nonlinear = 'sinh'
    if bench_id == 12: nonlinear = 'polynomial'
    rbm = WangLei3(input_shape=tuple(config['hamiltonian']['size']), num_features=[4],
            version='conv', itype='complex128', powerlist=powerlist_list[bench_id], stride=1,
            usesum=False if bench_id<8 else True, nonlinear=nonlinear, 
            fixbias=bench_id==10)
    return rbm
