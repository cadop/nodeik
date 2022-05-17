
#include "../native/builtin.h"

// avoid namespacing of float type for casting to float type, this is to avoid wp::float(x), which is not valid in C++
#define float(x) cast_float(x)
#define adj_float(x, adj_x, adj_ret) adj_cast_float(x, adj_x, adj_ret)

#define int(x) cast_int(x)
#define adj_int(x, adj_x, adj_ret) adj_cast_int(x, adj_x, adj_ret)


using namespace wp;



extern "C" __global__ void compute_loss_cuda_kernel_forward(launch_bounds_t dim,
	array_t<transform> var_body_q,
	int32 var_body_index,
	array_t<float32> var_loss)
{
    int _idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (_idx >= dim.size) 
        return;

    set_launch_bounds(dim);

        //---------
    // primal vars
    transform var_0;
    vec3 var_1;
    const vec3 var_2 = {0.44999998807907104, 0.0, 0.5};
    vec3 var_3;
    float32 var_4;
    const int32 var_5 = 0;
    //---------
    // forward
        var_0 = wp::load(var_body_q, var_body_index);
        var_1 = wp::transform_get_translation(var_0);
        var_3 = wp::sub(var_1, var_2);
        var_4 = wp::dot(var_3, var_3);
        wp::store(var_loss, var_5, var_4);

}


extern "C" __global__ void compute_loss_cuda_kernel_backward(launch_bounds_t dim,
	array_t<transform> var_body_q,
	int32 var_body_index,
	array_t<float32> var_loss,
	array_t<transform> adj_body_q,
	int32 adj_body_index,
	array_t<float32> adj_loss)
{
    int _idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (_idx >= dim.size) 
        return;

    set_launch_bounds(dim);

        //---------
    // primal vars
    transform var_0;
    vec3 var_1;
    const vec3 var_2 = {0.44999998807907104, 0.0, 0.5};
    vec3 var_3;
    float32 var_4;
    const int32 var_5 = 0;
    //---------
    // dual vars
    transform adj_0 = 0;
    vec3 adj_1 = 0;
    vec3 adj_2 = 0;
    vec3 adj_3 = 0;
    float32 adj_4 = 0;
    int32 adj_5 = 0;
        //---------
        // forward
        var_0 = wp::load(var_body_q, var_body_index);
        var_1 = wp::transform_get_translation(var_0);
        var_3 = wp::sub(var_1, var_2);
        var_4 = wp::dot(var_3, var_3);
        //wp::store(var_loss, var_5, var_4);
        //---------
        // reverse
        wp::adj_store(var_loss, var_5, var_4, adj_loss, adj_5, adj_4);
        wp::adj_dot(var_3, var_3, adj_3, adj_3, adj_4);
        wp::adj_sub(var_1, var_2, adj_1, adj_2, adj_3);
        wp::adj_transform_get_translation(var_0, adj_0, adj_1);
        wp::adj_load(var_body_q, var_body_index, adj_body_q, adj_body_index, adj_0);
        return;

}



extern "C" {

// Python entry points
WP_API void compute_loss_cuda_forward(void* stream, launch_bounds_t dim,
	array_t<transform> var_body_q,
	int32 var_body_index,
	array_t<float32> var_loss)
{
    compute_loss_cuda_kernel_forward<<<(dim.size + 256 - 1) / 256, 256, 0, (cudaStream_t)stream>>>(dim,
			var_body_q,
			var_body_index,
			var_loss);
}

WP_API void compute_loss_cuda_backward(void* stream, launch_bounds_t dim,
	array_t<transform> var_body_q,
	int32 var_body_index,
	array_t<float32> var_loss,
	array_t<transform> adj_body_q,
	int32 adj_body_index,
	array_t<float32> adj_loss)
{
    compute_loss_cuda_kernel_backward<<<(dim.size + 256 - 1) / 256, 256, 0, (cudaStream_t)stream>>>(dim,
			var_body_q,
			var_body_index,
			var_loss,
			adj_body_q,
			adj_body_index,
			adj_loss);
}

} // extern C



extern "C" __global__ void step_kernel_cuda_kernel_forward(launch_bounds_t dim,
	array_t<float32> var_x,
	array_t<float32> var_grad,
	float32 var_alpha)
{
    int _idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (_idx >= dim.size) 
        return;

    set_launch_bounds(dim);

        //---------
    // primal vars
    int32 var_0;
    float32 var_1;
    float32 var_2;
    float32 var_3;
    float32 var_4;
    //---------
    // forward
        var_0 = wp::tid();
        var_1 = wp::load(var_x, var_0);
        var_2 = wp::load(var_grad, var_0);
        var_3 = wp::mul(var_2, var_alpha);
        var_4 = wp::sub(var_1, var_3);
        wp::store(var_x, var_0, var_4);

}


extern "C" __global__ void step_kernel_cuda_kernel_backward(launch_bounds_t dim,
	array_t<float32> var_x,
	array_t<float32> var_grad,
	float32 var_alpha,
	array_t<float32> adj_x,
	array_t<float32> adj_grad,
	float32 adj_alpha)
{
    int _idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (_idx >= dim.size) 
        return;

    set_launch_bounds(dim);

        //---------
    // primal vars
    int32 var_0;
    float32 var_1;
    float32 var_2;
    float32 var_3;
    float32 var_4;
    //---------
    // dual vars
    int32 adj_0 = 0;
    float32 adj_1 = 0;
    float32 adj_2 = 0;
    float32 adj_3 = 0;
    float32 adj_4 = 0;
        //---------
        // forward
        var_0 = wp::tid();
        var_1 = wp::load(var_x, var_0);
        var_2 = wp::load(var_grad, var_0);
        var_3 = wp::mul(var_2, var_alpha);
        var_4 = wp::sub(var_1, var_3);
        //wp::store(var_x, var_0, var_4);
        //---------
        // reverse
        wp::adj_store(var_x, var_0, var_4, adj_x, adj_0, adj_4);
        wp::adj_sub(var_1, var_3, adj_1, adj_3, adj_4);
        wp::adj_mul(var_2, var_alpha, adj_2, adj_alpha, adj_3);
        wp::adj_load(var_grad, var_0, adj_grad, adj_0, adj_2);
        wp::adj_load(var_x, var_0, adj_x, adj_0, adj_1);
        return;

}



extern "C" {

// Python entry points
WP_API void step_kernel_cuda_forward(void* stream, launch_bounds_t dim,
	array_t<float32> var_x,
	array_t<float32> var_grad,
	float32 var_alpha)
{
    step_kernel_cuda_kernel_forward<<<(dim.size + 256 - 1) / 256, 256, 0, (cudaStream_t)stream>>>(dim,
			var_x,
			var_grad,
			var_alpha);
}

WP_API void step_kernel_cuda_backward(void* stream, launch_bounds_t dim,
	array_t<float32> var_x,
	array_t<float32> var_grad,
	float32 var_alpha,
	array_t<float32> adj_x,
	array_t<float32> adj_grad,
	float32 adj_alpha)
{
    step_kernel_cuda_kernel_backward<<<(dim.size + 256 - 1) / 256, 256, 0, (cudaStream_t)stream>>>(dim,
			var_x,
			var_grad,
			var_alpha,
			adj_x,
			adj_grad,
			adj_alpha);
}

} // extern C

