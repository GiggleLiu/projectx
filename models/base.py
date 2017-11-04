'''
Basic State Ansatz
'''
from poornn nets, linears

class StateNN(nets.ANN):
    def forward_acc(self, locs, dx, data_cache=None):
        ys = []
        acc_tag = True
        for layer in self.layers:
            if acc_tag:
                x = layer.forward_acc(x, data_cache=data_cache, **kwargs)
                if isinstance(layer, linears.LinearBase):
                    acc_tag = False
            else:
                x = layer.forward(x, data_cache=data_cache, **kwargs)
            ys.append(x)
            if isinstance(x, list):
                x = x[-1]
        if data_cache is not None:
            data_cache[self.uuid] = ys
        return x

    def __call__(self, config):
        return self.get_weight(config)

    @property
    def nsite(self):
        return np.prod(self.input_shape)

    def tovec(self, mag=None):
        '''
        Get the state vector.

        \Psi(s,W)=\sum_{\{hi\}} e^{\sum_j a_j\sigma_j^z+\sum_i b_ih_i +\sum_{ij}W_{ij}h_i\sigma_j}
        '''
        configs=unpacknbits_pm(np.arange(2**self.nsite), self.nsite)
        if mag is not None:
            configs = configs[configs.sum(axis=1)==mag]
        return np.array([self.get_weight(config_i) for config_i in configs])

    def get_weight(self,config,theta=None):
        '''
        Get the weight for specific configuration.

        Parameters:
            :config: 1darray,
            :theta: 1darray/None, table of hidden layer output: b+v.dot(W), intended to boost operation.

        Return:
            number,
        '''
        return self.forward(config)*self.get_sign(config)

    def get_sign(self, config, **kwargs):
        return 1
