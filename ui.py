#!/usr/bin/env python

import fire, os, sys, pdb
import numpy as np

from utils import analyse_exact
from problems import load_hamiltonian
from plotlib import plt

class UI(object):
    ###################  BENTCHMARK  ######################
    def benchmark(self, configfile, id, interactive=False):
        from problems import load_config
        config = load_config(configfile)
        folder = os.path.dirname(configfile)

        # modification to parameters
        sys.path.insert(0,folder)
        from config import modifyconfig_and_getnn
        rbm = modifyconfig_and_getnn(config, id)

        n = config['mpi']['num_core']

        logfile = '%s/log-%s.log'%(folder,id)
        options = ' '.join(['%s'%item for item in [configfile, id]])
        exec_code = 'python benchmark.py %s'%(options,)
        if not interactive:
            exec_code = 'nohup mpirun -n %s %s > %s &'%(n,exec_code,logfile)
        print('Excuting: %s'%exec_code)
        os.system(exec_code)

    def bbb(self, subfolder, id):
        '''Quick benchmark, not using mpi or nohup.'''
        self.benchmark('benchmarks/%s/config-sample.ini'%subfolder, id, interactive=True)

    def bdn(self, id, interactive=False):
        '''shortcut for benchmark depth of wanglei4 model.'''
        self.benchmark(configfile='benchmarks/wanglei4dn/config-sample.ini', id=id, interactive=interactive)

    def bK(self, id, interactive=False):
        '''shortcut for benchmark filter size of wanglei4 model.'''
        self.benchmark(configfile='benchmarks/wanglei4K/config-sample.ini', id=id, interactive=interactive)

    def b6(self, id, interactive=False):
        self.benchmark(configfile='benchmarks/wanglei6/config-sample.ini', id=id, interactive=interactive)

    def bm18(self, id, interactive=False):
        self.benchmark(configfile='benchmarks/mul1dp8/config-sample.ini', id=id, interactive=interactive)

    def bm15(self, id, interactive=False):
        self.benchmark(configfile='benchmarks/mul1dp5/config-sample.ini', id=id, interactive=interactive)

    def bm10(self, id, interactive=False):
        self.benchmark(configfile='benchmarks/mul1dp0/config-sample.ini', id=id, interactive=interactive)

    def b12p(self, id, interactive=False):
        self.benchmark(configfile='benchmarks/1d12p/config-sample.ini', id=id, interactive=interactive)

    def b12pbn(self, id, interactive=False):
        self.benchmark(configfile='benchmarks/1d12p_bn/config-sample.ini', id=id, interactive=interactive)

    def b16p(self, id, interactive=False):
        self.benchmark(configfile='benchmarks/1d16p/config-sample.ini', id=id, interactive=interactive)

    def b16p8(self, id, interactive=False):
        self.benchmark(configfile='benchmarks/1d16p8/config-sample.ini', id=id, interactive=interactive)

    def b20p(self, id, interactive=False):
        self.benchmark(configfile='benchmarks/1d20p/config-sample.ini', id=id, interactive=interactive)

    def b20p8(self, id, interactive=False):
        self.benchmark(configfile='benchmarks/1d20p8/config-sample.ini', id=id, interactive=interactive)

    def bw5(self, id, interactive=False):
        self.benchmark(configfile='benchmarks/wanglei5/config-sample.ini', id=id, interactive=interactive)

    def b40p(self, id, mode='', interactive=False):
        if mode=='':
            self.benchmark(configfile='benchmarks/1d40p/config-sample.ini', id=id, interactive=interactive)
        elif mode=='sr':
            self.benchmark(configfile='benchmarks/1d40p_sr/config-sample.ini', id=id, interactive=interactive)

    def bmpi(self, id, interactive=False):
        self.benchmark(configfile='benchmarks/mpiacc/config-sample.ini', id=id, interactive=interactive)

    def test(self, arg1, arg2):
        print('GET: arg1 = %s, arg2 = %s'%(arg1, arg2))

    def bestep(self, subfolder, id, istep):
        '''show energy at specific step for a benchmark instance.'''
        # get configuration and foler
        from problems import load_config, pconfig
        configfile = 'benchmarks/%s/config-sample.ini'%subfolder
        config = load_config(configfile)

        # modification to parameters
        folder = os.path.dirname(configfile)
        sys.path.insert(0,folder)
        from config import modifyconfig_and_getnn
        rbm = modifyconfig_and_getnn(config, id)
        varfile = os.path.join(folder,'variables-%s%s.npy'%(id,istep))
        rbm.set_variables(np.load(varfile))
        optimizer, problem = pconfig(config, rbm)
        print('Energy at step %s = %s'%(istep, problem.get_energy()))
     
    def bshowe(self, subfolder, id):
        '''show energy function.'''
        # get configuration and foler
        from problems import load_config, pconfig
        from plotlib import show_el
        configfile = 'benchmarks/%s/config-sample.ini'%subfolder
        folder = os.path.dirname(configfile)
        config = load_config(configfile)
        plt.ion()
        show_err = False
        show_el(datafiles = ['%s/el-%i.dat'%(folder,id)],
                EG = config['hamiltonian']['EG'],
                legends = ['id = %s'%id],
                show_err=show_err,
                xlogscale=not show_err)
        pdb.set_trace()

    def bshowerr(self, subfolder, id):
        '''show energy function.'''
        # get configuration and foler
        from problems import load_config, pconfig
        from plotlib import show_el
        configfile = 'benchmarks/%s/config-sample.ini'%subfolder
        folder = os.path.dirname(configfile)
        config = load_config(configfile)
        plt.ion()
        show_err = True
        show_el(datafiles = ['%s/el-%i.dat'%(folder,id)],
                EG = config['hamiltonian']['EG'],
                legends = ['id = %s'%id],
                show_err=show_err,
                xlogscale=not show_err)
        pdb.set_trace()

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
    

def main():
    fire.Fire(UI)

if __name__ == '__main__':
    main()
