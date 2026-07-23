#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
__global__ void saxpy(int n, float a, const float* x, float* y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}
int main() {
  int n = 1 << 20; size_t sz = n * sizeof(float);
  float *x = (float*)malloc(sz), *y = (float*)malloc(sz), *dx, *dy;
  for (int i = 0; i < n; i++) { x[i] = 1.0f; y[i] = 2.0f; }
  cudaMalloc(&dx, sz); cudaMalloc(&dy, sz);
  cudaMemcpy(dx, x, sz, cudaMemcpyHostToDevice);
  cudaMemcpy(dy, y, sz, cudaMemcpyHostToDevice);
  saxpy<<<(n + 255) / 256, 256>>>(n, 3.0f, dx, dy);
  cudaError_t e = cudaDeviceSynchronize();
  cudaMemcpy(y, dy, sz, cudaMemcpyDeviceToHost);
  float maxerr = 0; for (int i = 0; i < n; i++) maxerr = fmaxf(maxerr, fabsf(y[i] - 5.0f));
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
  printf("device=%s sync=%s max_error=%f\n", p.name, cudaGetErrorString(e), maxerr);
  return maxerr == 0.0f ? 0 : 1;
}
