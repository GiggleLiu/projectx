import numpy as np
import fire
from controller import *

np.random.seed(2)

if __name__ == '__main__':
    fire.Fire()

# # depth benchmark
# run_rtheta_mlp(J2=0.8, nsite=10, mlp_shape=[128])
# run_rtheta_mlp(J2=0.8, nsite=10, mlp_shape=[128, 64])
# run_rtheta_mlp(J2=0.8, nsite=10, mlp_shape=[128, 64, 32])
# run_rtheta_mlp(J2=0.8, nsite=10, mlp_shape=[128, 64, 32, 16])
# run_rtheta_mlp(J2=0.8, nsite=10, mlp_shape=[128, 64, 32, 16, 8])
# run_rtheta_mlp(J2=0.8, nsite=10, mlp_shape=[128, 64, 32, 32, 16, 8])
# # run_rtheta_mlp(J2=0.8, nsite=10, mlp_shape=[128, 256, 200, 64, 32, 32, 32, 32, 32, 32, 32, 32, 16, 16, 8])

# # shape benchmark
# run_rtheta_mlp(J2=0.8, nsite=10, mlp_shape=[128, 32, 64])
# run_rtheta_mlp(J2=0.8, nsite=10, mlp_shape=[128, 32, 16, 32])
# run_rtheta_mlp(J2=0.8, nsite=10, mlp_shape=[128, 64, 32, 32, 64])
# #run_ed_msr(J2=0.8, nsite=16)
# #scale_ed_msr(NJ2=51, nsite=16)
