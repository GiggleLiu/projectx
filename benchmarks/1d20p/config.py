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
        [[1]]+[[1]+[0]*i+[1] for i in range(10)],
        [[1],[1,1],[1,0,1],[1,0,0,1],[1]+[0]*7+[1]],
        ]

def modifyconfig_and_getnn(config, bench_id):
    rbm = WangLei3(input_shape=tuple(config['hamiltonian']['size']), num_features=[4],
            version='conv', itype='complex128', powerlist=powerlist_list[bench_id], stride=1)
    return rbm
