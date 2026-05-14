cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_CUDA_FLAGS='-lineinfo'
cmake --build build

nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  -o ./results/nsys1/nsys_mg_v_2d_4095 \
  ./build/poisson_cuda \
  --dtype double --case sine --solver mg \
  --cycle v --mg-coarse exact \
  --grid-size 4095 --tol 1e-9 --max-iter 10 --nu 3 --omega 1.25

nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  -o ./results/nsys1/nsys_mg_w_2d_4095 \
  ./build/poisson_cuda \
  --dtype double --case sine --solver mg \
  --cycle w --mg-coarse exact \
  --grid-size 4095 --tol 1e-9 --max-iter 10 --nu 3 --omega 1.25

nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  -o ./results/nsys1/nsys_mg_v_3d_383 \
  ./build/poisson_cuda \
  --dtype double --case sine --solver mg \
  --dim 3 --grid-size 383 \
  --cycle v --mg-coarse exact \
  --tol 1e-9 --max-iter 10 --nu 3 --omega 1.25

nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  -o ./results/nsys1/nsys_mg_w_3d_383 \
  ./build/poisson_cuda \
  --dtype double --case sine --solver mg \
  --dim 3 --grid-size 383 \
  --cycle w --mg-coarse exact \
  --tol 1e-9 --max-iter 10 --nu 3 --omega 1.25

nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  -o ./results/nsys1/nsys_mg_v_sor_2d_4095 \
  ./build/poisson_cuda \
  --dtype double --case sine --solver mg \
  --cycle v --mg-coarse sor \
  --grid-size 4095 --tol 1e-9 --max-iter 10 --nu 3 --omega 1.25

nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  -o ./results/nsys1/nsys_mg_w_sor_2d_4095 \
  ./build/poisson_cuda \
  --dtype double --case sine --solver mg \
  --cycle w --mg-coarse sor \
  --grid-size 4095 --tol 1e-9 --max-iter 10 --nu 3 --omega 1.25

nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  -o ./results/nsys1/nsys_mg_v_sor_3d_383 \
  ./build/poisson_cuda \
  --dtype double --case sine --solver mg \
  --dim 3 --grid-size 383 \
  --cycle v --mg-coarse sor \
  --tol 1e-9 --max-iter 10 --nu 3 --omega 1.25

nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  -o ./results/nsys1/nsys_mg_w_sor_3d_383 \
  ./build/poisson_cuda \
  --dtype double --case sine --solver mg \
  --dim 3 --grid-size 383 \
  --cycle w --mg-coarse sor \
  --tol 1e-9 --max-iter 10 --nu 3 --omega 1.25
