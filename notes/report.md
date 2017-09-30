#Day 15 Aug

## Wave function

### parameters

* A: J2 = 0.2, N = 10

Lowest 10 energies = [-4.15096326 -3.76842215 -3.70954039 -3.39455178 -3.39455178 -3.06084076
 -3.06084076 -2.96478367 -2.96478367 -2.83993641]

* B: J2 = 0.5, N = 10

Lowest 10 energies = [-3.75       -3.75       -3.39978825 -3.27928965 -3.27928965 -3.13824484
 -3.13824484 -3.13464932 -2.98733684 -2.98733684]

Exists degeneracy, not a good example for benchmarks.

* C: J2 = 0.2, N = 4

### results

    1. Comparing to correct sign, error in sign can be much larger than in energy. They have smaller sample size or cancel each other?
    2. Given correct signs, NN can be trained more efficiently. Can we prove the optimization is convex when the "Hamiltonian" matrix is negative.
    3. At J2 = 0.5, we can not compare signs due to degeneracy (~ 50% correct).
    4. Only ~ 250/30240 different configurations appear in 10-site samples.

### parameter A

Lowest 10 energies = [-4.15096326 -3.76842215 -3.70954039 -3.39455178 -3.39455178 -3.06084076
 -3.06084076 -2.96478367 -2.96478367 -2.83993641]

Running 0-th Iteration.
2000       Accept rate: 0.856
4000       Accept rate: 0.861
6000       Accept rate: 0.854
8000       Accept rate: 0.857
10000      Accept rate: 0.851
Before Training Sign, Energy = 2.97322923137
RESULT (252) = sa: -2.89471167857, relax: -2.73639249802
After training sign, Energy = -2.89471167857
Number of Configs Signed Incorrectly/All: 10/252 (3.9683%)
Error Ratio of Matrix Elements 1.2053%
Number of Samples Signed Incorrectly/All 147/10000 (1.4700%)
E/site = -0.289471167857 (-0.41509632551), Error = 17.8301%
diff rate = 0.0(norm=0.719404208193)

Running 1-th Iteration.
2000       Accept rate: 0.844
4000       Accept rate: 0.847
6000       Accept rate: 0.850
8000       Accept rate: 0.826
10000      Accept rate: 0.856
Before Training Sign, Energy = 2.95764451453
RESULT (252) = sa: -2.94907947751, relax: -2.79000973701
After training sign, Energy = -2.94907947751
Number of Configs Signed Incorrectly/All: 10/252 (3.9683%)
Error Ratio of Matrix Elements 1.0652%
Number of Samples Signed Incorrectly/All 128/10000 (1.2800%)
E/site = -0.294907947751 (-0.41509632551), Error = 16.9278%
diff rate = 0.0227777484859(norm=0.719404208193)

Running 2-th Iteration.
2000       Accept rate: 0.845
4000       Accept rate: 0.825
6000       Accept rate: 0.842
8000       Accept rate: 0.841
10000      Accept rate: 0.815
Before Training Sign, Energy = 2.96254019743
RESULT (252) = sa: -2.96155749439, relax: -2.77015161514
After training sign, Energy = -2.96155749439
Number of Configs Signed Incorrectly/All: 10/252 (3.9683%)
Error Ratio of Matrix Elements 1.0597%
Number of Samples Signed Incorrectly/All 129/10000 (1.2900%)
E/site = -0.296155749439 (-0.41509632551), Error = 16.7227%
diff rate = 0.0225978649609(norm=0.717653697288)

Running 3-th Iteration.
2000       Accept rate: 0.820
4000       Accept rate: 0.833
6000       Accept rate: 0.821
8000       Accept rate: 0.822
10000      Accept rate: 0.848
Before Training Sign, Energy = 2.95691943991
RESULT (252) = sa: -2.99696654149, relax: -2.74660515785
After training sign, Energy = -2.99696654149
Number of Configs Signed Incorrectly/All: 10/252 (3.9683%)
Error Ratio of Matrix Elements 0.9664%
Number of Samples Signed Incorrectly/All 118/10000 (1.1800%)
E/site = -0.299696654149 (-0.41509632551), Error = 16.1445%
diff rate = 0.022264571165(norm=0.716727473647)

... ...

Running 300-th Iteration.
2000       Accept rate: 0.281
4000       Accept rate: 0.282
6000       Accept rate: 0.281
8000       Accept rate: 0.310
10000      Accept rate: 0.271
Before Training Sign, Energy = 1.89933931239
RESULT (245) = sa: -4.15024465644, relax: -3.78778600693
After training sign, Energy = -4.15024465644
Number of Configs Signed Incorrectly/All: 29/245 (11.8367%)
Error Ratio of Matrix Elements 0.0117%
Number of Samples Signed Incorrectly/All 1524/10000 (15.2400%)
E/site = -0.415024465644 (-0.41509632551), Error = 0.0087%
diff rate = 0.000668497206872(norm=1.3827257929)

Running 301-th Iteration.
2000       Accept rate: 0.308
4000       Accept rate: 0.297
6000       Accept rate: 0.278
8000       Accept rate: 0.273
10000      Accept rate: 0.293
Before Training Sign, Energy = 1.96156032406
RESULT (243) = sa: -4.15009002005, relax: -3.86941695213
After training sign, Energy = -4.15009002005
Number of Configs Signed Incorrectly/All: 26/243 (10.6996%)
Error Ratio of Matrix Elements 0.0078%
Number of Samples Signed Incorrectly/All 1751/10000 (17.5100%)
E/site = -0.415009002005 (-0.41509632551), Error = 0.0105%
diff rate = 0.000833871120181(norm=1.382845482)

### wave function

$J_2= 0.2$, $|\langle \Psi|\Psi_0\rangle|^2 = 0.999771263796$

![WF_J20.2_N10](img/WF_J20.2_N10.png)

Day 15 Aug
==============================
## Taking subspace gradient to train is not a valid approach.

## VMC will work in the following two cases

1. $\{x\}$ sampled adequately, so that observables can be represented accuratly.
2. most elements in $\{x\}$ are unique, but observable are continuous with respect to neighborhood.

In case 2, signs differ a lot between different Monte Carlo sampling process, not consistantly changed.
Sign change

## New Ideas
1. multiple featured RBM + Linear, with some features negative definite.
2. contrastive divergence to generate samples.

Day 17 Aug
==============================
## $r-\theta$ neural network
Ansatz $\psi(\alpha,\beta,x) = r(\alpha,x)e^{i\theta(\beta,x)}, st. r,\theta,\alpha,\beta\in \mathcal{R}$.

### Variation

In the following we try to get $\frac{\partial E}{\partial \alpha}$ and $\frac{\partial E}{\partial \beta}$.

-----------------------

$$E(\alpha,\beta)=\frac{\int_{x,x'} H_{x,x'}r(\alpha,x)r(\alpha,x')e^{i(\theta(\beta,x')-\theta(\beta,x))}}{\int_x r(\alpha,x)^2}$$

$$\frac{\partial E}{\partial \alpha}=\frac{\int_{x,x'} H_{x,x'}(\theta) r(\alpha,x)\frac{\partial r(\alpha,x')}{\partial\alpha}+H_{x,x'}(\theta) r(\alpha,x')\frac{\partial r(\alpha,x)}{\partial\alpha}-E(\alpha,\beta)\int_{x}2r(\alpha,x)\frac{\partial r(\alpha,x)}{\partial \alpha}}{\int_xr(\alpha,x)^2}$$

with $H_{x,x'}=H_{xx'}e^{i(\theta(\beta,x')-\theta(\beta,x))}$.

Let $p(\alpha,x)\equiv \frac{r(\alpha,x)^2}{\int_xr(\alpha,x)^2}$,

$$\frac{\partial E}{\partial \alpha}=\int_{x}p(\alpha,x)\frac{\int_{x'}\left[H_{xx'}(\theta)+H_{x'x}(\theta)\right]r(\alpha,x')\partial r(\alpha,x)/\partial \alpha}{r(\alpha,x)^2}-2E(\alpha,\beta)\int_{x}p(\alpha,x)\frac{\partial r(\alpha,x)/\partial \alpha}{r(\alpha,x)}$$

$$\frac{\partial E}{\partial \alpha}\sim\langle\Re[E_{loc}]\Delta^{\alpha}_{loc}\rangle-E\langle\Delta^\alpha_{loc}\rangle$$

with $E_{loc}\equiv \int_{x'} H_{xx'}(\theta)\frac{r(\alpha,x')}{r(\alpha,x)}$ and $\Delta_{loc}^\alpha(x)\equiv\frac{\partial r(\alpha,x)/\partial \alpha}{r(\alpha,x)}$

------------------------------

$$\frac{\partial E}{\partial \beta}=\frac{\int_{x,x'} H_{x,x'}(\theta) r(\alpha,x)r(\alpha,x')i[\frac{\partial\theta(\beta,x')}{\partial \beta}-\frac{\partial\theta(\beta,x)}{\partial \beta}]}{\int_xr(\alpha,x)^2}$$

$$\frac{\partial E}{\partial \beta}=\int_{x,x'} \left[H_{x,x'}(\theta)-H_{x'x}(\theta)\right] p(\alpha,x)\frac{r(\alpha,x')}{r(\alpha,x)}(-i)\frac{\partial\theta(\beta,x)}{\partial \beta}$$

$$\frac{\partial E}{\partial \beta}\sim\langle\Im[E_{loc}]\Delta^{\beta}_{loc}\rangle$$

with $\Delta^\beta_{loc}\equiv \partial\theta(\beta,x)/\partial\beta$

Apparently, gradients are all real.

Day 20 Aug
===========================
## Do translational invariance allow a change of sign?

Yes, for example, 6 site version Marshall Sign Rule (MSR), we can have 3 up spins in the ground state, 
e.g. 1, 2 up spins in A, B sub-lattices respectively.
When we shift the spin configuration for 1-site, the sign changes according to MSR.

However, for real Hamiltonian, $T_1|\Psi\rangle=\pm|\Psi\rangle$. In Heisenberg model, $-1$ corresponds to $2\times odd$ number of spins.

## r-theta version Stochastic Reconfiguration
To get the imaginary time evolution of parameters, we start from the Lagrangian

$$L=\left[\frac{i}{2}(\langle\dot{\psi(\alpha)}|\psi(\alpha)\rangle-\langle\psi(\alpha)|\dot{\psi(\alpha)}\rangle)-\langle\psi(\alpha)|H|\psi(\alpha)\rangle\right]/\langle\psi(\alpha)|\psi(\alpha)\rangle$$

Here, $\alpha$ is a set of real parameters.

To achieve minimum action, we apply Euler-Lagrangian formula $\frac{\partial L}{\partial \alpha}-\frac{d}{dt}(\frac{\partial L}{\partial \dot{\alpha}})=0$.

$$\begin{align}\frac{\partial L}{\partial\alpha_i}=&\left[\frac{i}{2}(\langle\psi(\alpha)|\frac{\partial}{\partial\alpha_i}^\dagger\sum_j\dot{\alpha_j}\frac{\partial}{\partial\alpha_j}^\dagger+\sum_j\dot{\alpha_j}\frac{\partial}{\partial\alpha_j}^\dagger\frac{\partial}{\partial\alpha_i}|\psi(\alpha)\rangle-\langle\psi(\alpha)|\frac{\partial}{\alpha_i}^\dagger\sum_j\dot{a_j}\frac{\partial}{\partial a_j}+\sum_j\dot{a_j}\frac{\partial}{\partial a_j}\frac{\partial}{\partial\alpha_i}|\psi(\alpha)\rangle)\\-\langle\psi(\alpha)|\frac{\partial}{\partial \alpha_i}^\dagger H+H\frac{\partial}{\partial \alpha_i}|\psi(\alpha)\rangle\right]/N\\&-\left[\frac{i}{2}(\langle\psi(\alpha)|\sum_j\dot{\alpha_j}\frac{\partial}{\partial\alpha_j}^\dagger|\psi(\alpha)\rangle-\langle\psi(\alpha)|\sum_j\dot{\alpha_j}\frac{\partial}{\partial\alpha_j}|\psi(\alpha)\rangle)-\langle\psi(\alpha)|H|\psi(\alpha)\rangle\right]\langle\psi(\alpha)|\frac{\partial}{\partial \alpha_i}+\frac{\partial}{\partial \alpha_i}^\dagger|\psi(\alpha)\rangle/N^2\end{align}$$

Here, $N=\langle\psi(\alpha)|\psi(\alpha)\rangle$.

Notice $\partial_i,\partial_j$ (shorthand for $\frac{\partial}{\partial \alpha_{i,j}}$) are diagonal in $x$ basis ( ? ), they commute with each other.

$$\begin{align}\frac{\partial L}{\partial \alpha_i}=&\langle(\partial_i+\partial_i^\dagger)\left[-\frac{i}{2}\sum_j\dot{\alpha_j}(\partial_j-\partial_j^\dagger)\right]\rangle-\langle\partial_i^\dagger H+H\partial_i\rangle+\left[\frac{i}{2}\langle\sum_j\dot{\alpha_j}(\partial_j-\partial_j^\dagger)\rangle+\langle H\rangle\right]\langle\partial_i+\partial_i^\dagger\rangle\\=&2\langle\Re(\partial_i)\sum_j\dot{\alpha_j}\Im(\partial_j)\rangle-2\sum_j\dot{\alpha_j}\langle\Im(\partial_j)\rangle\langle\Re(\partial_i)\rangle-2\Re(\langle\partial_i^\dagger H\rangle-\langle\partial^\dagger\rangle\langle H\rangle )\end{align}$$

On the other side,

$$\begin{align}\frac{d}{dt}(\frac{\partial L}{\partial \dot{\alpha_i}})=&-\frac{i}{2}\left[\langle\psi(\alpha)|\sum_j\dot{\alpha_j}\partial_j^\dagger(\partial_i-\partial_i^\dagger)+(\partial_i-\partial_i^\dagger)\sum_j\dot{\alpha_j}\partial_j|\psi(\alpha)\rangle\right]/N\\&-\langle\psi(\alpha)|-\frac{i}{2}(\partial_i-\partial_i^\dagger)|\psi(\alpha)\rangle\langle\psi(\alpha)|\sum_j\dot{\alpha_j}(\partial_j+\partial_j^\dagger)|\psi(\alpha)\rangle/N^2\\=&2\sum_j\dot{\alpha_j}\langle\Re(\partial_j)\Im(\partial_i)\rangle-2\sum_j\dot{\alpha_j}\langle\Re(\partial_j)\rangle\langle\Im(\partial_i)\rangle\end{align}$$

To make it clear, we define matrix $S_{i,j}=\Im(\langle:\partial_i^\dagger\partial_j:\rangle$ and vectors $F_i=\Re(\langle:\partial_i^\dagger H:\rangle)$,  $g_i=\dot{\alpha_i}$. When we take the extreme, we will have

$$S\cdot g-F=0$$

obviously $g=S^{-1}F$ is real during real time evolution.

What about imaginary time evolution?

We have our new Lagrangian

$$L=\left[\frac{1}{2}(\langle\psi(\alpha)|\frac{d}{d\tau}^\dagger+\frac{d}{d\tau}|\psi(\alpha)\rangle-\langle\psi(\alpha)|H|\psi(\alpha)\rangle\right]/\langle\psi(\alpha)|\psi(\alpha)\rangle$$

Then derivatives canceled ??????????????????????????????????????????

## Naive version

$$S(\alpha,\alpha')\equiv\langle\Delta^{\alpha\dagger}_{loc}\Delta^{\alpha'}_{loc}\rangle-\langle\Delta_{loc}^{\alpha\dagger}\rangle\langle\Delta_{loc}^{\alpha'}\rangle$$

$F(\alpha)\equiv\langle\Delta_{loc}^{\alpha\dagger} E_{loc}\rangle-\langle\Delta_{loc}^{\alpha\dagger}\rangle\langle E_{loc}\rangle$

and the gradient

$$G=S^{-1}F$$

------------------------------------

In out case,

$$\Delta_{loc}^\alpha=\frac{\partial\psi(\alpha,\beta,x)/\partial\alpha}{\psi(\alpha,\beta,x)}=\frac{\partial r(\alpha,x)/\partial\alpha} {r(\alpha,x)}$$

$$\Delta_{loc}^\beta=\frac{\partial\psi(\alpha,\beta,x)}{\partial\beta}=i\frac{\partial \theta(\beta,x)}{\partial\beta}$$

* must $G$ be complex in this imaginary time evolution?
* $S$ is block diagonal?

$S$ is block diagonal means $S(\alpha,\beta)\equiv\langle\Delta^{\alpha\dagger}_{loc}\Delta^{\beta}_{loc}\rangle-\langle\Delta_{loc}^{\alpha\dagger}\rangle\langle\Delta_{loc}^{\beta}\rangle=0$

or it will be block diagonal layer wise? as Roger Luo commented.

Day 23 Aug
======================

Violation of Marshall Sign Rule
---------------------------

![MSR-J1J2_N12](img/MSR-J1J2_N12.png)

$J_1,J_2$ model, 12 sites.

Here, $E_0$ is evaluated using exact ground state wave function $v_0$.

$E$ is evaluated using ${\rm MSR}(|v|)$. Here, $\rm MSR$ is Marshall Sign Rule function.

To make it clear,

![MSR-J1J2_N12_log](img/MSR-J1J2_N12_log.png)

we see when $J_2\leq0.4$, MSR is only slightly violated.

To run this example, 
```
    from controllers import scale_ed_msr
    scale_ed_msr(J2MIN=0, J2MAX=1, NJ2=51, size=(12,), yscale='log')
```

For $2D$  $4\times4$ model, we have the similar behavior

![MSR-J1J22D_N16_log](img/MSR-J1J22D_N16_log.png)

## Training large $J_2$
For large $J_2$, MSR is strongly violated, as a result, neural networks are extremely hard to train. For example $J_2=0.8$.
* given sign, 1-layer RBM, after 300 steps of SR($\delta=10^{-4}$ regulation) training,  $\|\langle\Psi_0|\Psi\rangle\|_2\simeq0.999$, Error in energy is about $0.05\%$.

  ![Exact-Sign-J20.8-RBM](img/Exact-Sign-J20.8-RBM.png)

* not given sign, complex 1 Conv-Layer + 2 Linear-Layer (with 16, 8 features), $\|\langle\Psi_0|\Psi\rangle\|_2\simeq0.57$, Error in energy is about $2.5\%$.

  ![WL_16](img/WL_16.png)

  ## Given Amplitude to train sign

  $J_2=0.8, N=10, E_G=-0.433768079327$.

  For vectors, ![WL_16](img/SIGN-N10-J20.8-2.png)

For 1-layer/2-layer network, $E=-0.145755657726, overlap = 0.09$, they do not converge!

![WL_16](img/SIGN-N10-J20.8-1l.png)

without TI, even worse.

How can gradient descent work in sign networks?



# Day 29 Aug

Representation power of a linear $\theta$-network.

$L=4, J_2=0.8,S_z=0$

Bases of Hilbert space are $\sigma_{1-6}\in\{|++--\rangle,|+-+-\rangle,|+--+\rangle,|-++-\rangle,|-+-+\rangle,|--++\rangle\}$.

Here, $+$ represents spin $\uparrow$ or $+1$.

The sign structure of ground state is $\{-,0,+,+,0,-\}$, our task is to construct a network of function $\theta$ that $\theta(\sigma_{1,3,4,6})\%2\pi=\pi,0,0,\pi$, like XOR gates.

## Why our network fails

Our $\theta$ Network

$$x\rightarrow {\rm Conv(stride=2, num\_feature=4)}$$

​	$$\color{red}\rightarrow{\rm Sum~over~convolution~ axes}$$

​	$$\rightarrow{\rm Linear(4\times1)\rightarrow {\rm output~ as~} \theta}$$

Summation after convolution is what matters! Because $W\sigma_i =-W\sigma_{i+2}$, outputs will cancel each other if added directly!

Instead of putting a non-linear function layer after convolution, or training XOR gates, can we use product of neighbor bits as inputs? like $x_1x_2, x_2x_3,\ldots,x_{n-1}x_n,x_nx_1$. Which is supposed easier to train in large $J_2$ limit.



# Day 1 Sep

Our New Network that works extremely fine in$N=8,12, J_2=0.8$ network.

![wanglei3](img/WangLei3.png)

Our new network that works for $4\times4, J_2=0.8$ networks.

![wanglei4](img/WangLei4.png)

Result, Energy as a function of steps, with $x$ axis in log scale.
![el44](img/ENG44-J20.8.png)

The dashed line is the exact energy. Zoom in, we see it is still going down
![el44](img/ENG44[-10.1,-8]-J20.8.png)


# Day 5 Sep
Bentchmark on filter size $K$ and network depth $D$.
Model parameter $J_2=0.5, size = (4, 4)$, exact energy is -8.45792335.

## Bentchmark for different kernel size

### structures
For $K=1,2,3,4$ (filter size $(K\times K)$), using number of features [8, 32]

|                   K=1                    |                   K=2                    |                   K=3                    |                   K=4                    |
| :--------------------------------------: | :--------------------------------------: | :--------------------------------------: | :--------------------------------------: |
| ![](../benchmarks/wanglei4K/WangLei4-0.png) | ![](../benchmarks/wanglei4K/WangLei4-1.png) | ![](../benchmarks/wanglei4K/WangLei4-2.png) | ![](../benchmarks/wanglei4K/WangLei4-3.png) |

### Error as a function of step
![](img/errl-K.png)

## Bentchmark for different network depth

### structures
For $K=4$ (filter size $(K\times K)$), using number of features as shown in the following tables

|                  8, 32                   |               16, 128, 32                |             16, 128, 64, 32              |           16, 128, 64, 32, 16            |            32, 256, 64, 16, 8            |
| :--------------------------------------: | :--------------------------------------: | :--------------------------------------: | :--------------------------------------: | :--------------------------------------: |
| ![](../benchmarks/wanglei4dn/WangLei4-0.png) | ![](../benchmarks/wanglei4dn/WangLei4-1.png) | ![](../benchmarks/wanglei4dn/WangLei4-2.png) | ![](../benchmarks/wanglei4dn/WangLei4-3.png) | ![](../benchmarks/wanglei4dn/WangLei4-4.png) |
Table: Configuration 1-5


|              1, 16, 256, 16              |            32, 64, 512, 64, 8            |              4, 16, 256, 16              |
| :--------------------------------------: | :--------------------------------------: | :--------------------------------------: |
| ![](../benchmarks/wanglei4dn/WangLei4-5.png) | ![](../benchmarks/wanglei4dn/WangLei4-6.png) | ![](../benchmarks/wanglei4dn/WangLei4-7.png) |
Table: Configuration 6-8

### Error as a function of step
![](img/errl-dn.png)

# Day 7 Sep

10000 step benchmark, lines are averaged over 20 steps (otherwise too noisy).

### Networks

![](img/errcr-dn.png)

Fig: In above legends r for real product layer, c for complex. 2, 7, 8 are target networks.

### Observations

* complex product layers are stabler than real ones.
* network - 7 (32, 64, 512, 64, 8), performance of deep and wide network is similar to shallow network - 2 (16, 128, 32), all with error $<0.1\%$, but converge slower.
* number of features in product network is important.

# Day 11 Sep
## MPI Acceleration Test
Number of flops on TianHe and Delta servers as a function of number of cores.
|           TianHe           |         Delta101          |          Delta102          |
| :------------------------: | :-----------------------: | :------------------------: |
| ![](img/mpiacc-tianhe.png) | ![](img/mpiacc-delta.png) | ![](img/mpiacc-delta2.png) |
The forward method of convolusion layer take >50% resources, good news.

Resources spent on Parrallel/ Sequencial/ Transimission (Tian He as an example)
![](img/mpiacc-parts-tianhe.png)

Resources spent on Parrallel/ Sequencial/ Transimission (Tian He as an example)
Error in 100 step didn't blow up.
![](img/mpiacc-error-tianhe.png)

# Day 19 Sep
Benchmark on different product input,

* 0: raw data,
* 1: product of 1st NN,
* 2: product of 2nd NN,
* 3: product of consequtive 3 sites,
* 4: 0 + 1
* 5: 0 + 1 + 2
* 6: 0 + 1 + 2 +3

## 8 sites, different $J_2$
### $J_2=0.0$
![](img/errl-8pJ20.0.png)

### $J_2=0.5$

![](img/errl-8pJ20.5.png)

### $J_2=0.8$

![](img/errl-8pJ20.8.png)

## $12$ sites, $J_2=0.8$

![](img/errl-12pJ20.8.png)

Curve for energy

![](img/el-12pJ20.8.png)

# Day 21 Sep
Ground state symmetry analysis, these ground states are obtained using ED. 

columns are different sizes for chain varying from $2$ to $20$, and rows are different $J_2$ varying from $0.0$ to $1.0$.

Data element '-++'  means system changes sign for translate 1 site operation ($T_1$), keeps sign for spin flip ($F$) and space inversion symmetry $I$.

![](img/symmetry_table.png)

## Summary

* $I,F$ changes sign of wave function when and only when $L/2$ is odd.
* For $J_2<0.5$, $T_1, F, I$ are all positive for even $L/2$ and negative for odd $L/2$.
* For $L=16$, $T_1$ have negative eigenvalue! This must be why I fail to get ground state for $N=16$. Is there a phase transition? I suspect even representation power of our Ansaz is enough to describe signn structures of $J_2=0.5$ and $J_2=0.0$, it can fail to describe sign structures for $J_2>0.5$.

# Day 23 Sep

## 16 Sites, $J_2=0.8$

## Network

![](../benchmarks/1d16p8/WangLei3-17.png)

### Description

* Poly layer has kernels attribute, which represents different types of polynomial expansions.
* Filter filter out Fourier amplitude with specific momentum, equal to `Mean` when $momentum=0$.

## Error vs different polynomial expansions

![](img/errl-16pJ20.8_nonlinear.png)

## Error vs number of feature.
![](img/errl-16pJ20.8_nfeature.png)

## Target State with $k=\pi$
![](img/errl-16pJ20.8_even.png)

## Weights of polynomials (Real Part)

![](img/polyweight.png)

* from wave function overlap, we see performance of pure polynomial is worst (however, using Rmsprop instead of Adam, we have ~0.996 overlap for pure polynomial expansion).

*  positive-negative oscillating structure is shown in all these cases. 

* The real part of these functions for $x\in[-2,2]$ are very different

  ![](img/polycurve.png)

# kData flow

![](img/dataflow.png)

This is an example, I show that after polynomial operation, we have data amplitude blowing up.



## 20 Sites, $J_2=0.8$

![](img/errl-20pJ20.8_nonlinear.png)

$20$ features in convolution layer, polynomial kernel, optimize using Adam (0.01). 

# Day 25 Sep
Complex activation function,
* Sigmoid is singular at $z=(2n+1)i\pi$, its adaption is $f(x)=\frac{1}{1+e^{-\Re z}}+\frac{1}{1+e^{-\Im z}}$ (Birx 1992, Benvenuto 1992).
* Tanh is singular at $z=(n+1/2)i\pi$, its adaption for CVNN is $f(z) = \tanh(\Re z)+i\tanh(\Im z)$ (Kechriotis 1994, Kinouchi 1995), $f(z)=\tanh(|z|)\exp(i\arg(z))$ (Hirose 1994).
* $f(x) = \frac{z}{|z|}$
* $f(x) = \frac{z}{c+\frac{1}{r}|z|}$ (Noest 1998)
* $f(x) = \frac{|z|}{c+\frac{1}{r}|z|}\exp\left[i\{\arg z-\frac{1}{2^n}\sin(2^n \arg z)\}\right]$ (Kuroe 2005).

Comparison can be found in Kuroe 2009.

* using conformal mapping $f(z) = \frac{(\cos\theta+i\sin\theta)(z-\alpha)}{1-\bar{\alpha}z},|\alpha|<1$,
  This function is the general conformal mapping that transform unit disk in the complex plane onto itself and also a M¨obius transformation.
  First used by Mandic 2000 in RVNN.

# Day 27 Sep
Using real networks, 16 site $J_2=0.8$ will converge to an error $\simeq 1\%$.
