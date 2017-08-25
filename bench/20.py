import numpy as np

from controller import *

np.random.seed(2)

N=10
# depth benchmark
run_rtheta_mlp(J2=0.8, nsite=N, mlp_shape=[128])
run_rtheta_mlp(J2=0.8, nsite=N, mlp_shape=[128, 64])
run_rtheta_mlp(J2=0.8, nsite=N, mlp_shape=[128, 64, 32])
run_rtheta_mlp(J2=0.8, nsite=N, mlp_shape=[128, 64, 32, 16])
run_rtheta_mlp(J2=0.8, nsite=N, mlp_shape=[128, 64, 32, 16, 8])
run_rtheta_mlp(J2=0.8, nsite=N, mlp_shape=[128, 64, 32, 32, 16, 8])
run_rtheta_mlp(J2=0.8, nsite=N, mlp_shape=[128, 256, 200, 64, 32, 32, 32, 32, 32, 32, 32, 32, 16, 16, 8])

# shape benchmark
run_rtheta_mlp(J2=0.8, nsite=N, mlp_shape=[128, 32, 64])
run_rtheta_mlp(J2=0.8, nsite=N, mlp_shape=[128, 32, 16, 32])
run_rtheta_mlp(J2=0.8, nsite=N, mlp_shape=[128, 64, 32, 32, 64])
#run_ed_msr(J2=0.8, nsite=16)
#scale_ed_msr(NJ2=51, nsite=16)
