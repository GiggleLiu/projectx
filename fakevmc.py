'''Several Toy Models.'''

from numpy import *
from numpy.linalg import norm
import scipy.sparse as sps
import pdb
#from scipy.sparse import csr_matrix,kron,eye
#from scipy.sparse.linalg import eigsh

from utils import sx,sy,sz
from spaceconfig import SpinSpaceConfig
from utils import logfh_prime
from models import *

__all__=['ExactVMC']

class ExactVMC(object):
    '''The Fake but exact VMC program'''
    def __init__(self,h):
        self.h=h
        self.scfg=scfg=SpinSpaceConfig([h.nsite,2])

    def get_H(self, sparse=False):
        '''Get the target Hamiltonian Matrix.'''
        nsite,periodic=self.h.nsite,self.h.periodic
        scfg=SpinSpaceConfig([nsite,2])
        if sparse:
            kron_ = sps.kron
            sx_ = sps.csr_matrix(sx)
            sy_ = sps.csr_matrix(sy)
            sz_ = sps.csr_matrix(sz)
            eye_ = sps.eye
        else:
            kron_ = kron
            sx_ = sx
            sy_ = sy
            sz_ = sz
            eye_ = eye
        if isinstance(self.h,TFI):
            Jz,h=self.h.Jz,self.h.h
            h2=Jz/4.*kron_(sz_,sz_)
            H=0
            for i in range(nsite):
                if i!=nsite-1:
                    H=H+kron_(kron_(eye_(2**i),h2),eye_(2**(nsite-2-i)))
                elif periodic: #periodic boundary
                    H=H+Jz/4.*kron_(kron_(sz_,eye_(2**(nsite-2))),sz_)
                H=H+h/2.*kron_(kron_(eye_(2**i),sx_),eye_(2**(nsite-i-1)))
            return H
        elif isinstance(self.h,HeisenbergH):
            H=0
            for INB,(J,Jz) in enumerate(zip(self.h.Js,self.h.Jzs)):
                h2=J/4.*(kron_(sx_,kron_(eye_(2**INB),sx_))+kron_(sy_,kron_(eye_(2**INB),sy_)))+Jz/4.*kron_(sz_,kron_(eye_(2**INB),sz_))
                for i in range(nsite-INB-1):
                    H=H+kron_(kron_(eye_(2**i),h2),eye_(2**(nsite-2-i-INB)))

                #impose periodic boundary
                if periodic:
                    if INB==1:
                        H=H+J/4.*(kron_(kron_(kron_(eye_(2),sx_),eye_(2**(nsite-3))),sx_)+kron_(kron_(kron_(eye_(2),sy_),eye_(2**(nsite-3))),sy_))\
                        +Jz/4.*(kron_(kron_(kron_(eye_(2),sz_),eye_(2**(nsite-3))),sz_))
                        H=H+J/4.*(kron_(kron_(sx_,eye_(2**(nsite-3))),kron_(sx_,eye_(2)))+kron_(kron_(sy_,eye_(2**(nsite-3))),kron_(sy_,eye_(2))))\
                        +Jz/4.*(kron_(kron_(sz_,eye_(2**(nsite-3))),kron_(sz_,eye_(2))))
                    else:
                        H=H+J/4.*(kron_(kron_(sx_,eye_(2**(nsite-2))),sx_)+kron_(kron_(sy_,eye_(2**(nsite-2))),sy_))+Jz/4.*(kron_(kron_(sz_,eye_(2**(nsite-2))),sz_))
            return H
        elif isinstance(self.h,HeisenbergH2D):
            J,Jz=self.h.J,self.h.Jz
            atom0s=arange(nsite).reshape((self.h.N1,self.h.N2))
            atomys=roll(atom0s,1,axis=1)
            atomxs=roll(atom0s,1,axis=0)
            atom2a=roll(roll(atom0s,1,axis=1),1,axis=0)
            atom2b=roll(roll(atom0s,1,axis=0),-1,axis=1)
            H=0
            if False:
                sx0=csr_matrix(sx_)
                sy0=csr_matrix(sy_)
                sz0=csr_matrix(sz_)
            sx0,sy0,sz0=sx_,sy_,sz_
            for atom1,atom2 in zip(atom0s.ravel(),atomxs.ravel(),atom2a.ravel())+zip(atom0s.ravel(),atomys.ravel(),atom2b.ravel()):
                if atom1>atom2: atom1,atom2=atom2,atom1
                for ss,Ji in zip([sx0,sy0,sz0],[J,J,Jz]):
                    H=H+Ji/4.*kron_(kron_(kron_(kron_(eye_(2**atom1),ss),eye_(2**(atom2-atom1-1))),ss),eye_(2**(nsite-atom2-1)))
            return H

    def project_vec(self,vec,m=0):
        '''Project vector to good quantum number'''
        scfg=self.scfg
        configs=1-2*scfg.ind2config(arange(scfg.hndim))
        vec[sum(configs,axis=1)!=0]=0
        return vec

    def subspace_mask(self,m=0):
        '''Project vector to good quantum number'''
        scfg=self.scfg
        configs=1-2*scfg.ind2config(arange(scfg.hndim))
        return sum(configs,axis=1)==m


    def measure(self,op,state,initial_config=None,**kwargs):
        '''Measure an operator through detailed calculation.'''
        nsite=state.nsite
        H=self.get_H()
        scfg=self.scfg
        #prepair state
        v=state.tovec(scfg)
        if isinstance(self.h,(HeisenbergH,HeisenbergH2D)): v=self.project_vec(v,0)
        v/=norm(v)

        if isinstance(op,(HeisenbergH,TFI,HeisenbergH2D)):
            return v.conj().dot(H).dot(v)
        elif isinstance(op,PartialW):
            configs=1-2*scfg.ind2config(arange(scfg.hndim))
            pS=[]
            pS.append(configs)
            theta=state.forward(configs)
            pS.append(logfh_prime(theta).reshape([len(configs),state.group.ng,len(state.b)]).sum(axis=1))
            configs_g=state.group.apply_all(configs).swapaxes(0,1)
            pS.append(sum(configs_g[:,:,:,newaxis]*logfh_prime(theta).reshape([scfg.hndim,state.group.ng,1,state.W.shape[1]]),axis=1).reshape([configs.shape[0],-1]))
            pS=concatenate(pS,axis=-1)
            return sum((v.conj()*v)[:,newaxis]*pS,axis=0)
        elif isinstance(op,OpQueue):
            #get H
            OH=v.conj().dot(H).dot(v)

            #get W
            configs=1-2*scfg.ind2config(arange(scfg.hndim))
            pS=[]
            pS.append(configs)
            theta=state.forward(configs)
            pS.append(logfh_prime(theta).reshape([len(configs),state.group.ng,len(state.b)]).sum(axis=1))
            configs_g=state.group.apply_all(configs).swapaxes(0,1)
            pS.append(sum(configs_g[:,:,:,newaxis]*logfh_prime(theta).reshape([scfg.hndim,state.group.ng,1,state.W.shape[1]]),axis=1).reshape([configs.shape[0],-1]))
            pS=concatenate(pS,axis=-1)
            OPW=sum((v.conj()*v)[:,newaxis]*pS,axis=0)

            OPW2=sum((v.conj()*v)[:,newaxis,newaxis]*(pS.conj()[:,:,newaxis]*pS[:,newaxis]),axis=0)
            #OPWH=(v.conj()[:,newaxis,newaxis]*(pS.conj()[:,newaxis,:]*OH[:,:,newaxis])*v[newaxis,:,newaxis]).sum(axis=(0,1))
            Hloc=zeros(v.shape,dtype='complex128')
            Hloc[v!=0]=H.dot(v)[v!=0]/v[v!=0]
            #Hloc=H.dot(v)/v
            OPWH=(Hloc[:,newaxis]*pS.conj()*(v.conj()*v)[:,newaxis]).sum(axis=0)
            if op.nop==4:
                return OPW,OH,OPW2,OPWH
            else:
                return OPW,OH,OPWH
        else:
            raise TypeError()
