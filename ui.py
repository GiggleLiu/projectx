#!/usr/bin/env python

import fire, os, sys, pdb
import numpy as np

from utils import analyse_exact
from problems import load_hamiltonian
from plotlib import plt, DShow
from views import get_opt_prob

class UI(object):
    ###################  BENTCHMARK  ######################
    def benchmark(self, configfile, bench_id, interactive=False):
        from problems import load_config
        config = load_config(configfile)
        folder = os.path.dirname(configfile)

        # modification to parameters
        sys.path.insert(0,folder)
        from config import modifyconfig_and_getnn
        rbm = modifyconfig_and_getnn(config, bench_id)

        n = config['mpi']['num_core']

        logfile = '%s/log-%s.log'%(folder,bench_id)
        options = ' '.join(['%s'%item for item in [configfile, bench_id]])
        exec_code = 'python benchmark.py %s'%(options,)
        if not interactive:
            exec_code = 'nohup mpirun -n %s %s > %s &'%(n,exec_code,logfile)
        print('Excuting: %s'%exec_code)
        os.system(exec_code)

    def b(self, subfolder, bench_id, interactive=False):
        '''
        Quick benchmark.

        Args:
            subfolder (str): subfolder under benchmarks.
            bench_id (int): the integer used for benchmark.
        '''
        self.benchmark('benchmarks/%s/config-sample.ini'%subfolder, bench_id, interactive=interactive)

    def bdn(self, bench_id, interactive=False):
        '''shortcut for benchmark depth of wanglei4 model.'''
        self.benchmark(configfile='benchmarks/wanglei4dn/config-sample.ini', bench_id=bench_id, interactive=interactive)

    def bK(self, bench_id, interactive=False):
        '''shortcut for benchmark filter size of wanglei4 model.'''
        self.benchmark(configfile='benchmarks/wanglei4K/config-sample.ini', bench_id=bench_id, interactive=interactive)

    def b6(self, bench_id, interactive=False):
        self.benchmark(configfile='benchmarks/wanglei6/config-sample.ini', bench_id=bench_id, interactive=interactive)

    def bm18(self, bench_id, interactive=False):
        self.benchmark(configfile='benchmarks/mul1dp8/config-sample.ini', bench_id=bench_id, interactive=interactive)

    def bm15(self, bench_id, interactive=False):
        self.benchmark(configfile='benchmarks/mul1dp5/config-sample.ini', bench_id=bench_id, interactive=interactive)

    def bm10(self, bench_id, interactive=False):
        self.benchmark(configfile='benchmarks/mul1dp0/config-sample.ini', bench_id=bench_id, interactive=interactive)

    def b12p(self, bench_id, interactive=False):
        self.benchmark(configfile='benchmarks/1d12p/config-sample.ini', bench_id=bench_id, interactive=interactive)

    def b12pbn(self, bench_id, interactive=False):
        self.benchmark(configfile='benchmarks/1d12p_bn/config-sample.ini', bench_id=bench_id, interactive=interactive)

    def b12pu(self, bench_id, interactive=False):
        self.benchmark(configfile='benchmarks/1d12pu/config-sample.ini', bench_id=bench_id, interactive=interactive)

    def b16p(self, bench_id, interactive=False):
        self.benchmark(configfile='benchmarks/1d16p/config-sample.ini', bench_id=bench_id, interactive=interactive)

    def b16p8(self, bench_id, interactive=False):
        self.benchmark(configfile='benchmarks/1d16p8/config-sample.ini', bench_id=bench_id, interactive=interactive)

    def b16pd(self, bench_id, interactive=False):
        self.benchmark(configfile='benchmarks/1d16pd/config-sample.ini', bench_id=bench_id, interactive=interactive)

    def b20p(self, bench_id, interactive=False):
        self.benchmark(configfile='benchmarks/1d20p/config-sample.ini', bench_id=bench_id, interactive=interactive)

    def b20p8(self, bench_id, interactive=False):
        self.benchmark(configfile='benchmarks/1d20p8/config-sample.ini', bench_id=bench_id, interactive=interactive)

    def b20pu(self, bench_id, interactive=False):
        self.benchmark(configfile='benchmarks/1d20pu/config-sample.ini', bench_id=bench_id, interactive=interactive)

    def b40pu(self, bench_id, interactive=False):
        self.benchmark(configfile='benchmarks/1d40pu/config-sample.ini', bench_id=bench_id, interactive=interactive)

    def bw5(self, bench_id, interactive=False):
        self.benchmark(configfile='benchmarks/wanglei5/config-sample.ini', bench_id=bench_id, interactive=interactive)

    def b40p(self, bench_id, mode='', interactive=False):
        if mode=='':
            self.benchmark(configfile='benchmarks/1d40p/config-sample.ini', bench_id=bench_id, interactive=interactive)
        elif mode=='sr':
            self.benchmark(configfile='benchmarks/1d40p_sr/config-sample.ini', bench_id=bench_id, interactive=interactive)

    def bmpi(self, bench_id, interactive=False):
        self.benchmark(configfile='benchmarks/mpiacc/config-sample.ini', bench_id=bench_id, interactive=interactive)

    def bvmodel(self, subfolder, bench_id):
        from poornn import viznn
        optimizer, problem = get_opt_prob(subfolder, bench_id)
        filename = 'benchmarks/%s/%s-%s.png'%(subfolder,problem.rbm.__class__.__name__,bench_id)
        viznn(problem.rbm, filename=filename)
        os.system('xdg-open %s'%filename)

    def bestep(self, subfolder, bench_id, istep):
        '''show energy at specific step for a benchmark instance.'''
        optimizer, problem = get_opt_prob(subfolder, bench_id)
        varfile = 'benchmarks/%s/variables-%s%s.npy'%(subfolder,bench_id,istep)
        problem.rbm.set_variables(np.load(varfile))
        print('Energy at step %s = %s'%(istep, problem.get_energy()))
     
    def bve(self, subfolder, bench_id, extension='png'):
        '''show energy function.'''
        # get configuration and foler
        from problems import load_config, pconfig
        from plotlib import show_el
        configfile = 'benchmarks/%s/config-sample.ini'%subfolder
        folder = os.path.dirname(configfile)
        config = load_config(configfile)
        with DShow((5,3.5),filename="notes/data/EL-%s-%s.%s"%(subfolder,bench_id,extension)) as ds:
            show_el(datafiles = ['%s/el-%i.dat'%(folder,bench_id)],
                    nsite = np.prod(config['hamiltonian']['size']),
                    EG = config['hamiltonian']['EG'],
                    legends = ['id = %s'%bench_id],
                    show_err=False,
                    xlogscale=True)

    def bverr(self, subfolder, bench_id, extension='png'):
        '''show energy error function.'''
        # get configuration and foler
        from problems import load_config, pconfig
        from plotlib import show_el
        configfile = 'benchmarks/%s/config-sample.ini'%subfolder
        folder = os.path.dirname(configfile)
        config = load_config(configfile)
        with DShow((5,3.5),filename="notes/data/EL-%s-%s.%s"%(subfolder,bench_id,extension)) as ds:
            show_el(datafiles = ['%s/el-%i.dat'%(folder,bench_id)],
                    nsite = np.prod(config['hamiltonian']['size']),
                    EG = config['hamiltonian']['EG'],
                    legends = ['id = %s'%bench_id],
                    show_err=True,
                    xlogscale=False)

    ######################  EXACT  #########################

    def ed(self, J2, size, *tasks):
        h = load_hamiltonian(model='J1J2',J1=1, J2=J2, size=size)
        H, e, v, configs = analyse_exact(h, do_printsign=False, num_eng=1)
        if 'v' in tasks: print('v = %s'%v)
        if 'H' in tasks: print('H = %s'%H)
        if 'sign' in tasks:
            np.set_printoptions(threshold='nan')
            print('sign = %s'%np.sign(v))
            np.set_printoptions(threshold=1000)
    

if __name__ == '__main__':
    fire.Fire(UI)
