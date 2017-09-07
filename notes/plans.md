Plans
============================

# To Do List

## JgLiu:

- [x] $N = 20, J_2 = 0, 0.2$
- [x] $N = 10$, show $\psi$ and $\psi_{exact}$
- [ ] prove optimization is convex if signs are known
- [ ] calculate AFM structure factor.
- [x] optimize SA solver using sparsity
- [x] skip sampling
- [x] formula: gradient descent, the $r-\theta$ version
- [ ] formula: stochastic reconfiguration, the $r-\theta$ version
- [x] VMC with 5% chance flip a configuration
- [x] train $\theta$ with amplitude fixed, plot it.
- [x] translational invariance
- [x] test $2D $ $J_1-J_2$ model
- [ ] representation power of $\theta$ network
- [x] visualization of networks.
- [ ] bentchmark 40 sites, $J_2= 0.8, E_G =-17.???$, $J_2=0, E_G=$,$J_2=0.25, E_G=-16.???$
- [ ] bentchmark depth of network, using 10000 iteration.
- [ ] plot filters in first layer.
- [ ] show why product network works

## Bentch Mark Guidelines on WangLei4 model.

Important Hyper-Parameters

- [ ] depth of network
  * As the network become deep, the training convergence become slow.
- [x] filter size
  * Error systemetically decrease as $K$ increses, $K=4$ for $size = (4,4)$ is suggested.

- [ ] initialization parameters $\eta$
- [ ] filter data type "complex128"/"float64"
- [x] different non-linear functions
  * $x^3$ better than ReLU and Exp, Sin.
  * $x^2, Cos$ do not work, symmetry reason?

- [ ] learning strategy "gd"/"adam"/"rmsprop"/"sr"
- [ ] learning rate

Marginal Hyper-Parameters
- [ ] vmc sample size

Roger:

- [ ] theta net
- [ ] exp net
- [ ] no theta

# To Read List
- [ ] brief read phase transition and combinatorial optimization, stoquastic problem

- [ ] read helmut's paper on spin glass

- [ ] spin glass server - as a bentch mark

- [ ] read blogs of Alex Selby

- [ ] paper on training sign, complex network (Aizenberg)

- [x] Violation of MSR

      > https://arxiv.org/pdf/cond-mat/0003207.pdf
      >
      > http://iopscience.iop.org/article/10.1209/0295-5075/25/7/012/pdf

# DMRG Result
J1 = 1, J2 = 0.0, N = 20, E = -8.9044
J1 = 1, J2 = 0.2, N = 20, E = -8.2029
J1 = 1, J2 = 0.5, N = 20, E = -7.5
