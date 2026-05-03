#include <curand_kernel.h>

__device__ float curand_laplace_standard(curandState* state) {
    float u = curand_uniform(state);
    float half_minus = u - 0.5f;
    float sign_val = (half_minus >= 0.0f) ? 1.0f : -1.0f;
    float abs_val = fabsf(half_minus);
    float inner = fmaxf(1.0f - 2.0f * abs_val, 1e-10f);
    return -sign_val * logf(inner);  // Laplace(0, 1)
}

extern "C" {

__global__ void generate_sample_kernel(
        int dim,
        unsigned long long shared_seed,
        unsigned long long idx,
        float* sample_out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState state;
        curand_init(shared_seed, 0, idx * dim, &state);
        for (int i = 0; i < dim; i++) {
            sample_out[i] = curand_laplace_standard(&state);
        }
    }
}

__global__ void reverse_channel_encode_kernel(
    const float* mu_q,
    int dim,
    unsigned long long K,
    unsigned long long shared_seed,
    float* log_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K) return;

    curandState state;
    curand_init(shared_seed, 0, idx * dim, &state);

    float log_w_value = 0.0f;
    for (int i = 0; i < dim; i++) {
        float z = curand_laplace_standard(&state);
        log_w_value += fabsf(z) - fabsf(z - mu_q[i]);
    }

    log_w[idx] = log_w_value;
}

} // extern "C"