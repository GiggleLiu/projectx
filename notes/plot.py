import sys, os
sys.path.insert(0,'../')
from plotlib import *

def show_bench44K():
    ntask = 4
    token = 'K'
    plt.ion()
    for show_err in [True, False]:
        plt.cla()
        show_el(datafiles = ['../benchmarks/wanglei4K/el-%i.dat'%i for i in range(ntask)],
                EG = -8.45792335,
                legends = ['K = %s'%(i+1) for i in range(ntask)],
                show_err=show_err,
                xlogscale=not show_err)
        pdb.set_trace()
        plt.savefig('img/%s-%s%s.png'%('errl' if show_err else 'el', token, ''))

def show_bench44dn():
    '''Deep network.'''
    token = 'dn'
    ntask = 5
    plt.ion()
    for show_err in [True, False]:
        plt.cla()
        show_el(datafiles = ['../benchmarks/wanglei4dn/el-%i.dat'%i for i in range(ntask)],
                legends = ['structure = %s'%(i+1) for i in range(ntask)],
                show_err=show_err,
                xlogscale=not show_err)
        pdb.set_trace()
        plt.savefig('img/%s-%s%s.png'%('errl' if show_err else 'el', token, ''))

def show_bench44dncr():
    '''Compare complex and real.'''
    token = 'dn'
    ids = [1,6,7]
    show_err = True
    plt.ion()
    from problems import load_config
    folder = '../benchmarks/wanglei4dn'
    config = load_config(folder+'/config-sample.ini')
    legends = []
    for id in ids:
        for version in ['c', 'r']:
            show_el(datafiles = [folder+'/el-%i%s.dat'%(id, 'c' if version=='c' else '')],
                    EG = config['hamiltonian']['EG'],
                    show_err=show_err,
                    xlogscale=not show_err, smooth_step=20)
            legends.append('%s-%s'%(version, id+1))
    plt.legend(legends)
    plt.ylim(1e-5,1)
    pdb.set_trace()
    plt.savefig('img/errcr-%s.png'%token)

def show_kernel44K(bench_id):
    configfile = '../benchmarks/wanglei4K/config-sample.ini'
    plt.ion()
    show_kernel_bench(configfile, bench_id = bench_id)
    plt.colorbar()
    pdb.set_trace()

def show_mpiacc(task='tianhe'):
    tdata = np.loadtxt('mpiacc-%s.tbl'%task)
    ncores, t_tots, t_fors, t_cnvs, t_mpis, accs = tdata.T
    tmpis = t_tots-t_fors
    plt.ion()
    plt.figure(figsize=(6,4))
    plt.title('Component Analysis')
    plt.ylabel('niter/sec')
    for data, color in zip(tdata[:,1:4].T, ['r','g','b']):
        plt.plot(ncores, 1./data, color=color)
        plt.plot(ncores, 1./data[0]*ncores, color=color, ls='--')
    plt.legend(['total', '%.4f'%(tdata[0,1]/tdata[-1,1]/ncores[-1]),'forward',
        '%.4f'%(tdata[0,2]/tdata[-1,2]/ncores[-1]),
        'conv','%.4f'%(tdata[0,3]/tdata[-1,3]/ncores[-1])])
    plt.xlim(0,24)
    plt.tight_layout()
    pdb.set_trace()
    plt.savefig('img/mpiacc-%s.png'%task)

def show_mpi_err():
    task='tianhe'
    tdata = np.loadtxt('mpiacc-%s.tbl'%task)
    ncores, t_tots, t_fors, t_cnvs, t_mpis, accs = tdata.T
    #plt.plot(ncores, 1./ncores/t_tots[:]*t_tots[0])
    plt.ion()
    plt.figure(figsize=(6,4))
    plt.plot(ncores, accs, '-o')
    plt.xlim(0,24)
    plt.title('Error in 100 steps')
    plt.xlabel('#  of cores')
    pdb.set_trace()
    plt.savefig('img/mpiacc-error-%s.png'%task)

def show_mpiparts():
    task='tianhe'
    tdata = np.loadtxt('mpiacc-%s.tbl'%task)
    ncores, t_tots, t_fors, t_cnvs, t_mpis, accs = tdata.T
    tcals = t_tots-t_mpis
    a, b = np.polyfit(ncores, tcals*ncores,1)  #resource = a*x+b
    label = 'parallel = %s, sequencial = %s'%(b, a)

    plt.ion()
    plt.figure(figsize=(6,4))
    #plt.plot(ncores, tcals, color='k')
    #plt.plot(ncores, b/ncores)
    #plt.plot(ncores, a*np.ones(len(ncores)))
    #plt.plot(ncores, t_mpis)
    plt.plot(ncores, tcals*ncores, color='k')
    plt.plot(ncores, b*np.ones(len(ncores)))
    plt.plot(ncores, a*ncores)
    plt.plot(ncores, t_mpis*ncores)

    plt.ylabel('Resources',fontsize=14)
    plt.legend(['Total','Parallel','Sequencial','Transimision'])
    plt.xlim(0,24)
    plt.tight_layout()

    pdb.set_trace()
    plt.savefig('img/mpiacc-parts-tianhe.png')

def show_bench6():
    tasks = [0,1,2,4]
    token = '6'
    plt.ion()
    for show_err in [True, False]:
        plt.cla()
        show_el(datafiles = ['../benchmarks/wanglei6/el-%i.dat'%i for i in tasks],
                EG = -0.503810*36,
                legends = ['%s'%i for i in tasks],
                show_err=show_err,
                xlogscale=not show_err)
        pdb.set_trace()
        plt.savefig('img/%s-%s%s.png'%('errl' if show_err else 'el', token, ''))

def show_bench8():
    tasks = [0,1,2]
    token = '8'
    plt.ion()
    for show_err in [False]:
        plt.cla()
        show_el(datafiles = ['../benchmarks/wanglei6/el-%i.dat'%i for i in tasks],
                EG = -0.5*64,
                legends = ['%s'%i for i in tasks],
                show_err=show_err,
                xlogscale=not show_err)
        pdb.set_trace()
        plt.savefig('img/%s-%s%s.png'%('errl' if show_err else 'el', token, ''))

def show_benchm1(J2):
    if J2==0.8:
        EG = -0.426340679527*8
    elif J2==0.0:
        EG = -0.456386676117*8
    elif J2==0.5:
        EG = -0.375*8
    else:
        raise
    #tasks = [0,1,2,3]
    #tasks = [0,1,4,5,6]
    tasks = range(7)
    token = '%dpJ2%s'%(8,J2)
    plt.ion()
    for show_err in [True, False]:
        plt.cla()
        show_el(datafiles = ['../benchmarks/mul1dp%d/el-%i.dat'%(J2*10,i) for i in tasks],
                EG = EG,
                legends = ['%s'%i for i in tasks],
                show_err=show_err,
                xlogscale=not show_err)
        pdb.set_trace()
        plt.savefig('img/%s-%s%s.png'%('errl' if show_err else 'el', token, ''))

def show_1d12p():
    J2 = 0.8
    EG = -0.42424305019*12
    #tasks = [0,1,2,3]
    #tasks = [0,1,4,5,6]
    tasks = range(11)
    token = '%dpJ2%s'%(12,J2)
    plt.ion()
    for show_err in [True, False]:
        plt.cla()
        show_el(datafiles = ['../benchmarks/1d%dp/el-%i.dat'%(12,i) for i in tasks],
                EG = EG,
                legends = ['%s'%i for i in tasks],
                show_err=show_err,
                xlogscale=not show_err)
        pdb.set_trace()
        plt.savefig('img/%s-%s%s.png'%('errl' if show_err else 'el', token, ''))

def show_1d16p(dataset='nonlinear'):
    J2 = 0.8
    EG = -6.78879425  # Odd parity
    if dataset=='nonlinear':
        tasks = [1,4,5,10,11,12,13,14,16]
        legends = ['sinh',r'$x^3$',r'$x^5$','polynomial','legendre', 'hermite', 'chebyshev', 'laguerre','hermiteE']
    elif dataset=='even':
        tasks = [6,7,8]
        legends = ['sinh',r'$x^5$','polynomial']
        EG = -6.6889395  # Even parity
    elif dataset=='optimizer':
        tasks = [10,15]
        legends = ['adam','rmsprop']
    elif dataset=='usesum':
        tasks = [0,1]
        legends = ['usesum-fixbias', 'not']
    elif dataset=='nfeature':
        tasks = [17,10,18,19]
        legends = [4,8,16,32]
    elif dataset=='rate':
        tasks = [17,20,21,22]
        legends = [1e-2,2e-2,4e-2,8e-2]
    else:
        raise
    token = '%dpJ2%s_%s'%(16,J2,dataset)
    plt.ion()
    for show_err in [True, False]:
        plt.cla()
        show_el(datafiles = ['../benchmarks/1d%dp8/el-%i.dat'%(16,i) for i in tasks],
                EG = EG,
                legends = legends,
                show_err=show_err,
                xlogscale=not show_err)
        pdb.set_trace()
        plt.savefig('img/%s-%s%s.png'%('errl' if show_err else 'el', token, ''))

def show_1d20p(dataset=''):
    J2 = 0.8
    EG = -0.423063620451*20
    if dataset=='nonlinear':
        tasks = [1]
        legends = ['polynomial']
    else:
        raise
    token = '%dpJ2%s_%s'%(20,J2,dataset)
    plt.ion()
    for show_err in [True, False]:
        plt.cla()
        show_el(datafiles = ['../benchmarks/1d%dp/el-%i.dat'%(20,i) for i in tasks],
                EG = EG,
                legends = legends,
                show_err=show_err,
                xlogscale=not show_err)
        pdb.set_trace()
        plt.savefig('img/%s-%s%s.png'%('errl' if show_err else 'el', token, ''))

if __name__ == '__main__':
    #show_bench44K()
    #show_bench44dn()
    #show_kernel44K(1)
    #show_bench44dncr()
    #show_mpiacc('delta')
    #show_mpiparts()
    #show_mpi_err()
    #show_bench6()
    #show_bench8()
    #show_benchm1(0.8)
    #show_1d12p()
    #show_1d16p('nonlinear')
    #show_1d16p('even')
    #show_1d16p('optimizer')
    #show_1d16p('usesum')
    #show_1d16p('nfeature')
    #show_1d16p('rate')
    #show_1d20p('nonlinear')
