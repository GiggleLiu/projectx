# for shape in '[128]' '[128, 64]'
# do
#     echo $shape
# done
python run_bench.py run_rtheta_mlp --J2=0.8 --nsite=10 --mlp_shape=[128]
python run_bench.py run_rtheta_mlp --J2=0.8 --nsite=10 --mlp_shape=[128, 64]
python run_bench.py run_rtheta_mlp --J2=0.8 --nsite=10 --mlp_shape=[128, 64, 32]
python run_bench.py run_rtheta_mlp --J2=0.8 --nsite=10 --mlp_shape=[128, 64, 32, 16]
python run_bench.py run_rtheta_mlp --J2=0.8 --nsite=10 --mlp_shape=[128, 64, 32, 16, 8]
python run_bench.py run_rtheta_mlp --J2=0.8 --nsite=10 --mlp_shape=[128, 64, 32, 32, 16, 8]