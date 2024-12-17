

#include<iostream>
#include "cuda_runtime.h"
#include<string>
#include<vector>
#include<fstream>
#include <cuda.h>
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>
#include <device_functions.h>


# define PI  3.14159265358979323846

static void HandleError(cudaError_t err, const char* file, int line) {
	if (err != cudaSuccess) {
		std::cout << cudaGetErrorString(err) << "in" << file << "at line" << line;
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ )) 



__global__ void Eout_Ein_calculation(float* Eo_gpu, float* Ei_gpu, float* fo_gpu, float* fi_gpu, float* input_img_gpu, int img_w, int img_h, int img_l, int sigma) {


	size_t i = blockDim.y * blockIdx.y + threadIdx.y;	// calculate row index, point to the output  //width 
	size_t j = blockDim.x * blockIdx.x + threadIdx.x;	// calculate column index, point to the output //height
	size_t p = blockDim.z * blockIdx.z + threadIdx.z;
	if (i >= img_h || j >= img_w || p >= img_l) return;

	float s1 = 0;
	float s2 = 0;

	for (int u = -sigma; u <= sigma; u++)
	{
		for (int v = -sigma; v <= sigma; v++)
		{
			for (int w = -sigma; w <= sigma; w++)
			{
				if (((i + u) >= 0) && ((i + u) < img_h) && ((j + v) >= 0) && ((j + v) < img_w) && ((p + w) >= 0) && ((p + w) < img_l))
				{
					s1 = (s1 + ((1 / pow(((2 * sigma) + 1), 2)) * (input_img_gpu[((p + w) * img_w * img_h) + ((i + u) * img_w) + (j + v)] - fo_gpu[(p * img_w * img_h) + i * img_w + j]) * (input_img_gpu[((p + w) * img_w * img_h) + ((i + u) * img_w) + (j + v)] - fo_gpu[(p * img_w * img_h) + i * img_w + j])));
					s2 = (s2 + ((1 / pow(((2 * sigma) + 1), 2)) * (input_img_gpu[((p + w) * img_w * img_h) + ((i + u) * img_w) + (j + v)] - fi_gpu[(p * img_w * img_h) + i * img_w + j]) * (input_img_gpu[((p + w) * img_w * img_h) + ((i + u) * img_w) + (j + v)] - fi_gpu[(p * img_w * img_h) + i * img_w + j])));
				}
			}
		}
	}
	Eo_gpu[(p * img_w * img_h) + i * img_w + j] = s1;
	Ei_gpu[(p * img_w * img_h) + i * img_w + j] = s2;



}

__global__ void gradient_dx_cuda(float* gradient_dx, float* output, int img_w, int img_h, int img_l) {
	size_t i = blockDim.y * blockIdx.y + threadIdx.y;
	size_t j = blockDim.x * blockIdx.x + threadIdx.x;
	size_t p = blockDim.z * blockIdx.z + threadIdx.z;

	if (i >= img_h || j >= img_w || p >= img_l) return;

	size_t j_left = j - 1;
	if (j_left < 0) {
		j_left = 0;
	}
	size_t j_right = j + 1;
	if (j_right >= img_w) {
		j_right = img_w - 1;
	}

	double dist_grad_left = gradient_dx[(p * img_h * img_w) + (i * img_w) + j_left];
	double dist_grad_right = gradient_dx[(p * img_h * img_w) + (i * img_w) + j_right];

	double dist_grad = (dist_grad_right - dist_grad_left) / 2.0;

	output[(p * img_h * img_w) + (i * img_w) + j] = dist_grad;
}


__global__ void gradient_dy_cuda(float* gradient_dx, float* output, int img_w, int img_h, int img_l) {
	// Calculate gradient along dy (3D)
	int i = blockDim.y * blockIdx.y + threadIdx.y; // Calculate row index (height)
	int j = blockDim.x * blockIdx.x + threadIdx.x; // Calculate column index (width)
	int p = blockDim.z * blockIdx.z + threadIdx.z;

	if (i >= img_h || j >= img_w || p >= img_l) return;

	int i_left = i - 1;
	int i_right = i + 1;

	if (i_left < 0) {
		i_left = 0;
		i_right = 1;
	}
	else if (i_right >= img_h) {
		i_right = img_h - 1;
		i_left = i_right - 1;
	}

	double dist_grad = (gradient_dx[(p * img_h * img_w) + (i_right * img_w) + j] - gradient_dx[(p * img_h * img_w) + (i_left * img_w) + j]) / 2.0f;

	output[(p * img_h * img_w) + (i * img_w) + j] = dist_grad;


}

__global__ void gradient_dz_cuda(float* gradient_dx, float* output, int img_w, int img_h, int img_l) {
	size_t i = blockDim.y * blockIdx.y + threadIdx.y;
	size_t j = blockDim.x * blockIdx.x + threadIdx.x;
	size_t p = blockDim.z * blockIdx.z + threadIdx.z;

	if (i >= img_h || j >= img_w || p >= img_l) return;

	size_t p_left = (p > 0) ? p - 1 : 0;
	size_t p_right = (p < img_l - 1) ? p + 1 : img_l - 1;

	double dist_grad_left = gradient_dx[(p_left * img_h * img_w) + (i * img_w) + j];
	double dist_grad_right = gradient_dx[(p_right * img_h * img_w) + (i * img_w) + j];

	double dist_grad = (dist_grad_right - dist_grad_left) / 2.0;

	output[(p * img_h * img_w) + (i * img_w) + j] = dist_grad;
}

void gradient_cal(float* input, float* output_x, float* output_y, float* output_z, int img_w, int img_h, int img_l) {


	cudaDeviceProp props;
	HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));


	float* input_gpu;
	float* output_x_gpu;
	float* output_y_gpu;
	float* output_z_gpu;

	size_t bytes = (img_w * img_h * img_l) * sizeof(float);
	HANDLE_ERROR(cudaMalloc(&input_gpu, bytes));  							    //allocate memory on device
	HANDLE_ERROR(cudaMalloc(&output_x_gpu, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&output_y_gpu, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&output_z_gpu, bytes));  							//allocate memory on device


	HANDLE_ERROR(cudaMemcpy(input_gpu, input, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device


	size_t blockDim = sqrt(props.maxThreadsPerBlock);
	//dim3 threads(blockDim, blockDim);
	dim3 threads(16, 16, 4);
	dim3 blocks(img_w / threads.x + 1, img_h / threads.y + 1, img_l / threads.z + 1);

	gradient_dx_cuda << < blocks, threads >> > (input_gpu, output_x_gpu, img_w, img_h, img_l);
	gradient_dy_cuda << < blocks, threads >> > (input_gpu, output_y_gpu, img_w, img_h, img_l);
	gradient_dz_cuda << < blocks, threads >> > (input_gpu, output_z_gpu, img_w, img_h, img_l);



	HANDLE_ERROR(cudaMemcpy(output_x, output_x_gpu, bytes, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(output_y, output_y_gpu, bytes, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(output_z, output_z_gpu, bytes, cudaMemcpyDeviceToHost));


	cudaFree(input_gpu);
	cudaFree(output_x_gpu);
	cudaFree(output_y_gpu);
	cudaFree(output_z_gpu);


}

__global__ void division_grad(float* output, float* input_1, float* input_2, int img_w, int img_h, int img_l) {


	size_t i = blockDim.y * blockIdx.y + threadIdx.y;	// calculate row index, point to the output  //width 
	size_t j = blockDim.x * blockIdx.x + threadIdx.x;	// calculate column index, point to the output //height
	size_t p = blockDim.z * blockIdx.z + threadIdx.z;
	if (i >= img_h || j >= img_w || p >= img_l) return;

	size_t index = i * img_w * img_l + j * img_l + p;
	float gradient_x = input_1[index];
	float gradient_y = input_2[index];

	float final = gradient_x / (gradient_y + 1e-6);
	output[index] = final;


}

__global__ void division(float* output, float* input_1, float* input_2, int img_w, int img_h, int img_l) {


	size_t i = blockDim.y * blockIdx.y + threadIdx.y;	// calculate row index, point to the output  //width 
	size_t j = blockDim.x * blockIdx.x + threadIdx.x;	// calculate column index, point to the output //height
	size_t p = blockDim.z * blockIdx.z + threadIdx.z;
	if (i >= img_h || j >= img_w || p >= img_l) return;

	size_t index = i * img_w * img_l + j * img_l + p;
	float gradient_x = input_1[index];
	float gradient_y = input_2[index];

	float final = gradient_x / (gradient_y);
	output[index] = final;


}


__global__ void multiplication(float* output, float* input_1, float* input_2, int img_w, int img_h, int img_l) {


	size_t i = blockDim.y * blockIdx.y + threadIdx.y;	// calculate row index, point to the output  //width 
	size_t j = blockDim.x * blockIdx.x + threadIdx.x;	// calculate column index, point to the output //height
	size_t p = blockDim.z * blockIdx.z + threadIdx.z;
	if (i >= img_h || j >= img_w || p >= img_l) return;

	size_t index = i * img_w * img_l + j * img_l + p;
	float gradient_x = input_1[index];
	float gradient_y = input_2[index];

	float final = gradient_x * (gradient_y);
	output[index] = final;


}

__global__ void plus(float* output, float* input_1, float* input_2, float* input_3, int img_w, int img_h, int img_l) {



	size_t i = blockDim.y * blockIdx.y + threadIdx.y;	// calculate row index, point to the output  //width 
	size_t j = blockDim.x * blockIdx.x + threadIdx.x;	// calculate column index, point to the output //height
	size_t p = blockDim.z * blockIdx.z + threadIdx.z;
	if (i >= img_h || j >= img_w || p >= img_l) return;

	size_t index = i * img_w * img_l + j * img_l + p;
	float gradient_x = input_1[index];
	float gradient_y = input_2[index];
	float gradient_z = input_3[index];

	float final = gradient_x + gradient_y + gradient_z;
	output[index] = final;

}


__global__ void heaviside(float* output, float* input, int img_w, int img_h, int img_l) {



	size_t i = blockDim.y * blockIdx.y + threadIdx.y;	// calculate row index, point to the output  //width 
	size_t j = blockDim.x * blockIdx.x + threadIdx.x;	// calculate column index, point to the output //height
	size_t p = blockDim.z * blockIdx.z + threadIdx.z;
	if (i >= img_h || j >= img_w || p >= img_l) return;

	size_t index = i * img_w * img_l + j * img_l + p;
	float epsilon = 0.2;
	float heaviside = (0.5 * (1 + (2 / 3.14159265358979323846) * atan(input[index] / epsilon)));

	output[index] = heaviside;

}

__global__ void deri_heaviside(float* output, float* input, int img_w, int img_h, int img_l) {



	size_t i = blockDim.y * blockIdx.y + threadIdx.y;	// calculate row index, point to the output  //width 
	size_t j = blockDim.x * blockIdx.x + threadIdx.x;	// calculate column index, point to the output //height
	size_t p = blockDim.z * blockIdx.z + threadIdx.z;
	if (i >= img_h || j >= img_w || p >= img_l) return;

	size_t index = i * img_w * img_l + j * img_l + p;
	float epsilon = 0.2;
	float deri_heaviside = (1 / 3.14159265358979323846) * (epsilon / ((epsilon * epsilon) + (input[index] * input[index])));

	output[index] = deri_heaviside;

}

__global__ void inverse(float* output, float* input, int img_w, int img_h, int img_l) {



	size_t i = blockDim.y * blockIdx.y + threadIdx.y;	
	size_t j = blockDim.x * blockIdx.x + threadIdx.x;	
	size_t p = blockDim.z * blockIdx.z + threadIdx.z;
	if (i >= img_h || j >= img_w || p >= img_l) return;

	size_t index = i * img_w * img_l + j * img_l + p;
	float gradient_x = input[index];


	float final = 1 - gradient_x;
	output[index] = final;

}

__global__ void pad_border(float* output, const float* input, int width, int height, int depth, int pad_w, int padded_width, int padded_height, int padded_depth) {
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	int i = blockDim.y * blockIdx.y + threadIdx.y;
	int k = blockDim.z * blockIdx.z + threadIdx.z;

	if (j >= padded_width || i >= padded_height || k >= padded_depth) return;

	// Calculate the output index in the padded volume
	int out_index = (k * padded_width * padded_height) + (i * padded_width) + j;

	// if the current position is within the padding border
	if (j < pad_w || j >= width + pad_w || i < pad_w || i >= height + pad_w || k < pad_w || k >= depth + pad_w) {
		// within the border
		output[out_index] = 30;
	}
	else {
		// corresponding input index
		int in_j = j - pad_w;
		int in_i = i - pad_w;
		int in_k = k - pad_w;
		int in_index = (in_k * width * height) + (in_i * width) + in_j;

		// Copy data 
		output[out_index] = input[in_index];
	}
}





void pad_border_cal(float* output, float* input, int img_w, int img_h, int img_l, int pad_w) {

	cudaDeviceProp props;
	HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));

	int padded_w = img_w + 2 * pad_w;
	int padded_h = img_h + 2 * pad_w;
	int padded_l = img_l + 2 * pad_w;

	float* output_gpu;
	float* input_gpu;
	size_t input_bytes = (img_w * img_h * img_l) * sizeof(float);
	size_t output_bytes = padded_w * padded_h * padded_l * sizeof(float);

	HANDLE_ERROR(cudaMalloc(&output_gpu, output_bytes));
	HANDLE_ERROR(cudaMalloc(&input_gpu, input_bytes));

	HANDLE_ERROR(cudaMemcpy(input_gpu, input, input_bytes, cudaMemcpyHostToDevice));
	size_t blockDim = sqrt(props.maxThreadsPerBlock);

	dim3 threads(16, 16, 4);


	dim3 blocks(padded_w / threads.x + 1, padded_h / threads.y + 1, padded_l / threads.z + 1);

	pad_border << <blocks, threads >> > (output_gpu, input_gpu, img_w, img_h, img_l, pad_w, padded_w, padded_h, padded_l);

	HANDLE_ERROR(cudaMemcpy(output, output_gpu, output_bytes, cudaMemcpyDeviceToHost));

	cudaFree(output_gpu);
	cudaFree(input_gpu);
}


__global__ void Final_phi(float* output, float* phi, float* deri_heav, float* Eout, float* Ein, float* normalized_DIV, float*DIV, float e, float dt, float v, float meu, int img_w, int img_h, int img_l) {



	size_t i = blockDim.y * blockIdx.y + threadIdx.y;
	size_t j = blockDim.x * blockIdx.x + threadIdx.x;
	size_t p = blockDim.z * blockIdx.z + threadIdx.z;
	if (i >= img_h || j >= img_w || p >= img_l) return;

	size_t index = i * img_w * img_l + j * img_l + p;
	float deri_heavi = deri_heav[index];
	float E_out = Eout[index];
	float E_in = Ein[index];
	float norm_div = normalized_DIV[index];
	float div = DIV[index];
	float prev_phi = phi[index];



	float final_phi = prev_phi - (deri_heavi *(e * (E_out - E_in)) * dt) + (v * dt * deri_heavi *norm_div) + meu * (dt * (div - norm_div)) ;
	output[index] = final_phi;

}


void Final_PHI_cal(float* output, float* phi, float* deri_heav, float* Eout, float* Ein, float* normalized_DIV, float* DIV, float e, float dt, float v, float meu, int img_w, int img_h, int img_l) {


	cudaDeviceProp props;
	HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));


	float* output_gpu;
	float* phi_gpu;
	float* deri_heav_gpu;
	float* DIV_gpu;
	float* normalized_DIV_gpu;
	float* Eout_gpu;
	float* Ein_gpu;





	size_t bytes = (img_w * img_h * img_l) * sizeof(float);
	HANDLE_ERROR(cudaMalloc(&output_gpu, bytes));  							    //allocate memory on 
	HANDLE_ERROR(cudaMalloc(&phi_gpu, bytes));  							    //allocate memory on device
	HANDLE_ERROR(cudaMalloc(&deri_heav_gpu, bytes));  							    //allocate memory on device
	HANDLE_ERROR(cudaMalloc(&DIV_gpu, bytes));  							    //allocate memory on device
	HANDLE_ERROR(cudaMalloc(&normalized_DIV_gpu, bytes));  							    //allocate memory on device
	HANDLE_ERROR(cudaMalloc(&Eout_gpu, bytes));  							    //allocate memory on device
	HANDLE_ERROR(cudaMalloc(&Ein_gpu, bytes));  							    //allocate memory on device


	HANDLE_ERROR(cudaMemcpy(phi_gpu, phi, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device
	HANDLE_ERROR(cudaMemcpy(deri_heav_gpu, deri_heav, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device
	HANDLE_ERROR(cudaMemcpy(DIV_gpu, DIV, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device
	HANDLE_ERROR(cudaMemcpy(normalized_DIV_gpu, normalized_DIV, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to 
	HANDLE_ERROR(cudaMemcpy(Eout_gpu, Eout, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device
	HANDLE_ERROR(cudaMemcpy(Ein_gpu, Ein, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device




	size_t blockDim = sqrt(props.maxThreadsPerBlock);
	//dim3 threads(blockDim, blockDim);
	dim3 threads(16, 16, 4);
	dim3 blocks(img_w / threads.x + 1, img_h / threads.y + 1, img_l / threads.z + 1);

	Final_phi << < blocks, threads >> > (output_gpu, phi_gpu, deri_heav_gpu, Eout_gpu, Ein_gpu, normalized_DIV_gpu, DIV_gpu, e, dt, v, meu, img_w, img_h, img_l);

	HANDLE_ERROR(cudaMemcpy(output, output_gpu, bytes, cudaMemcpyDeviceToHost));

	cudaFree(output_gpu);
	cudaFree(phi_gpu);
	cudaFree(deri_heav_gpu);
	cudaFree(DIV_gpu);
	cudaFree(normalized_DIV_gpu);
	cudaFree(Eout_gpu);
	cudaFree(Ein_gpu);






}

void division_cal(float* output, float* input_1, float* input_2, int img_w, int img_h, int img_l) {


	cudaDeviceProp props;
	HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));


	float* output_gpu;
	float* input_1_gpu;
	float* input_2_gpu;



	size_t bytes = (img_w * img_h * img_l) * sizeof(float);
	HANDLE_ERROR(cudaMalloc(&output_gpu, bytes));  							    //allocate memory on 
	HANDLE_ERROR(cudaMalloc(&input_1_gpu, bytes));  							    //allocate memory on device
	HANDLE_ERROR(cudaMalloc(&input_2_gpu, bytes));  							    //allocate memory on device


	HANDLE_ERROR(cudaMemcpy(input_1_gpu, input_1, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device
	HANDLE_ERROR(cudaMemcpy(input_2_gpu, input_2, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device


	size_t blockDim = sqrt(props.maxThreadsPerBlock);
	//dim3 threads(blockDim, blockDim);
	dim3 threads(16, 16, 4);
	dim3 blocks(img_w / threads.x + 1, img_h / threads.y + 1, img_l / threads.z + 1);

	division << < blocks, threads >> > (output_gpu, input_1_gpu, input_2_gpu, img_w, img_h, img_l);

	HANDLE_ERROR(cudaMemcpy(output, output_gpu, bytes, cudaMemcpyDeviceToHost));

	cudaFree(output_gpu);
	cudaFree(input_1_gpu);
	cudaFree(input_2_gpu);



}


void heaviside_cal(float* output, float* input, int img_w, int img_h, int img_l) {


	cudaDeviceProp props;
	HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));


	float* output_gpu;
	float* input_gpu;


	size_t bytes = (img_w * img_h * img_l) * sizeof(float);
	HANDLE_ERROR(cudaMalloc(&output_gpu, bytes));  							    //allocate memory on 
	HANDLE_ERROR(cudaMalloc(&input_gpu, bytes));  							    //allocate memory on device

	HANDLE_ERROR(cudaMemcpy(input_gpu, input, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device

	size_t blockDim = sqrt(props.maxThreadsPerBlock);
	//dim3 threads(blockDim, blockDim);
	dim3 threads(16, 16, 4);
	dim3 blocks(img_w / threads.x + 1, img_h / threads.y + 1, img_l / threads.z + 1);

	heaviside << < blocks, threads >> > (output_gpu, input_gpu, img_w, img_h, img_l);

	HANDLE_ERROR(cudaMemcpy(output, output_gpu, bytes, cudaMemcpyDeviceToHost));

	cudaFree(output_gpu);
	cudaFree(input_gpu);


}

void inverse_cal(float* output, float* input, int img_w, int img_h, int img_l) {


	cudaDeviceProp props;
	HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));


	float* output_gpu;
	float* input_gpu;


	size_t bytes = (img_w * img_h * img_l) * sizeof(float);
	HANDLE_ERROR(cudaMalloc(&output_gpu, bytes));  							    //allocate memory on 
	HANDLE_ERROR(cudaMalloc(&input_gpu, bytes));  							    //allocate memory on device

	HANDLE_ERROR(cudaMemcpy(input_gpu, input, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device

	size_t blockDim = sqrt(props.maxThreadsPerBlock);
	//dim3 threads(blockDim, blockDim);
	dim3 threads(16, 16, 4);
	dim3 blocks(img_w / threads.x + 1, img_h / threads.y + 1, img_l / threads.z + 1);

	inverse << < blocks, threads >> > (output_gpu, input_gpu, img_w, img_h, img_l);

	HANDLE_ERROR(cudaMemcpy(output, output_gpu, bytes, cudaMemcpyDeviceToHost));

	cudaFree(output_gpu);
	cudaFree(input_gpu);


}

void deri_heaviside_cal(float* output, float* input, int img_w, int img_h, int img_l) {


	cudaDeviceProp props;
	HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));


	float* output_gpu;
	float* input_gpu;


	size_t bytes = (img_w * img_h * img_l) * sizeof(float);
	HANDLE_ERROR(cudaMalloc(&output_gpu, bytes));  							    //allocate memory on 
	HANDLE_ERROR(cudaMalloc(&input_gpu, bytes));  							    //allocate memory on device

	HANDLE_ERROR(cudaMemcpy(input_gpu, input, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device

	size_t blockDim = sqrt(props.maxThreadsPerBlock);
	//dim3 threads(blockDim, blockDim);
	dim3 threads(16, 16, 4);
	dim3 blocks(img_w / threads.x + 1, img_h / threads.y + 1, img_l / threads.z + 1);

	deri_heaviside << < blocks, threads >> > (output_gpu, input_gpu, img_w, img_h, img_l);

	HANDLE_ERROR(cudaMemcpy(output, output_gpu, bytes, cudaMemcpyDeviceToHost));

	cudaFree(output_gpu);
	cudaFree(input_gpu);


}

void multiplication_cal(float* output, float* input_1, float* input_2, int img_w, int img_h, int img_l) {


	cudaDeviceProp props;
	HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));


	float* output_gpu;
	float* input_1_gpu;
	float* input_2_gpu;



	size_t bytes = (img_w * img_h * img_l) * sizeof(float);
	HANDLE_ERROR(cudaMalloc(&output_gpu, bytes));  							    //allocate memory on 
	HANDLE_ERROR(cudaMalloc(&input_1_gpu, bytes));  							    //allocate memory on device
	HANDLE_ERROR(cudaMalloc(&input_2_gpu, bytes));  							    //allocate memory on device


	HANDLE_ERROR(cudaMemcpy(input_1_gpu, input_1, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device
	HANDLE_ERROR(cudaMemcpy(input_2_gpu, input_2, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device


	size_t blockDim = sqrt(props.maxThreadsPerBlock);
	//dim3 threads(blockDim, blockDim);
	dim3 threads(16, 16, 4);
	dim3 blocks(img_w / threads.x + 1, img_h / threads.y + 1, img_l / threads.z + 1);

	multiplication << < blocks, threads >> > (output_gpu, input_1_gpu, input_2_gpu, img_w, img_h, img_l);

	HANDLE_ERROR(cudaMemcpy(output, output_gpu, bytes, cudaMemcpyDeviceToHost));

	cudaFree(output_gpu);
	cudaFree(input_1_gpu);
	cudaFree(input_2_gpu);



}

void DIV_cal(float* DIV, float* input_x, float* input_y, float* input_z, int img_w, int img_h, int img_l) {


	cudaDeviceProp props;
	HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));


	float* input_x_gpu;
	float* input_y_gpu;
	float* input_z_gpu;
	float* output_x_gpu;
	float* output_y_gpu;
	float* output_z_gpu;
	float* DIV_gpu;

	size_t bytes = (img_w * img_h * img_l) * sizeof(float);
	HANDLE_ERROR(cudaMalloc(&input_x_gpu, bytes));  							    //allocate memory on 
	HANDLE_ERROR(cudaMalloc(&input_y_gpu, bytes));  							    //allocate memory on device
	HANDLE_ERROR(cudaMalloc(&input_z_gpu, bytes));  							    //allocate memory on device
	HANDLE_ERROR(cudaMalloc(&output_x_gpu, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&output_y_gpu, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&output_z_gpu, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&DIV_gpu, bytes));  							//allocate memory on device


	HANDLE_ERROR(cudaMemcpy(input_x_gpu, input_x, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to 
	HANDLE_ERROR(cudaMemcpy(input_y_gpu, input_y, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device
	HANDLE_ERROR(cudaMemcpy(input_z_gpu, input_z, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device


	size_t blockDim = sqrt(props.maxThreadsPerBlock);
	//dim3 threads(blockDim, blockDim);
	dim3 threads(16, 16, 4);
	dim3 blocks(img_w / threads.x + 1, img_h / threads.y + 1, img_l / threads.z + 1);

	gradient_dx_cuda << < blocks, threads >> > (input_x_gpu, output_x_gpu, img_w, img_h, img_l);
	gradient_dy_cuda << < blocks, threads >> > (input_y_gpu, output_y_gpu, img_w, img_h, img_l);
	gradient_dz_cuda << < blocks, threads >> > (input_z_gpu, output_z_gpu, img_w, img_h, img_l);
	plus << < blocks, threads >> > (DIV_gpu, output_x_gpu, output_y_gpu, output_z_gpu, img_w, img_h, img_l);


	HANDLE_ERROR(cudaMemcpy(DIV, DIV_gpu, bytes, cudaMemcpyDeviceToHost));


	cudaFree(DIV_gpu);
	cudaFree(output_x_gpu);
	cudaFree(output_y_gpu);
	cudaFree(output_z_gpu);
	cudaFree(input_x_gpu);
	cudaFree(input_y_gpu);
	cudaFree(input_z_gpu);



}


void division_nDIV(float* input_x, float* input_y, float* input_z, float* gradientmagnitude, float* output_x, float* output_y, float* output_z, int img_w, int img_h, int img_l) {


	cudaDeviceProp props;
	HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));


	float* input_x_gpu;
	float* input_y_gpu;
	float* input_z_gpu;
	float* output_x_gpu;
	float* output_y_gpu;
	float* output_z_gpu;
	float* gradientmagnitude_gpu;


	size_t bytes = (img_w * img_h * img_l) * sizeof(float);
	HANDLE_ERROR(cudaMalloc(&input_x_gpu, bytes));  							    //allocate memory on 
	HANDLE_ERROR(cudaMalloc(&input_y_gpu, bytes));  							    //allocate memory on device
	HANDLE_ERROR(cudaMalloc(&input_z_gpu, bytes));  							    //allocate memory on device
	HANDLE_ERROR(cudaMalloc(&output_x_gpu, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&output_y_gpu, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&output_z_gpu, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&gradientmagnitude_gpu, bytes));  							//allocate memory on device


	HANDLE_ERROR(cudaMemcpy(input_x_gpu, input_x, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to 
	HANDLE_ERROR(cudaMemcpy(input_y_gpu, input_y, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device
	HANDLE_ERROR(cudaMemcpy(input_z_gpu, input_z, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device
	HANDLE_ERROR(cudaMemcpy(gradientmagnitude_gpu, gradientmagnitude, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device


	size_t blockDim = sqrt(props.maxThreadsPerBlock);
	//dim3 threads(blockDim, blockDim);
	dim3 threads(16, 16, 4);
	dim3 blocks(img_w / threads.x + 1, img_h / threads.y + 1, img_l / threads.z + 1);

	division_grad << < blocks, threads >> > (output_x_gpu, input_x_gpu, gradientmagnitude_gpu, img_w, img_h, img_l);
	division_grad << < blocks, threads >> > (output_y_gpu, input_y_gpu, gradientmagnitude_gpu, img_w, img_h, img_l);
	division_grad << < blocks, threads >> > (output_z_gpu, input_z_gpu, gradientmagnitude_gpu, img_w, img_h, img_l);

	HANDLE_ERROR(cudaMemcpy(output_x, output_x_gpu, bytes, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(output_y, output_y_gpu, bytes, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(output_z, output_z_gpu, bytes, cudaMemcpyDeviceToHost));

	cudaFree(output_x_gpu);
	cudaFree(output_y_gpu);
	cudaFree(output_z_gpu);
	cudaFree(input_x_gpu);
	cudaFree(input_y_gpu);
	cudaFree(input_z_gpu);
	cudaFree(gradientmagnitude_gpu);



}

void normalized_div_plus(float* normalize_DIV, float* input_x, float* input_y, float* input_z, int img_w, int img_h, int img_l) {


	cudaDeviceProp props;
	HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));


	float* input_x_gpu;
	float* input_y_gpu;
	float* input_z_gpu;
	float* normalize_DIV_gpu;

	size_t bytes = (img_w * img_h * img_l) * sizeof(float);
	HANDLE_ERROR(cudaMalloc(&input_x_gpu, bytes));  							    //allocate memory on 
	HANDLE_ERROR(cudaMalloc(&input_y_gpu, bytes));  							    //allocate memory on device
	HANDLE_ERROR(cudaMalloc(&input_z_gpu, bytes));  							    //allocate memory on device
	HANDLE_ERROR(cudaMalloc(&normalize_DIV_gpu, bytes));  							//allocate memory on device


	HANDLE_ERROR(cudaMemcpy(input_x_gpu, input_x, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to 
	HANDLE_ERROR(cudaMemcpy(input_y_gpu, input_y, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device
	HANDLE_ERROR(cudaMemcpy(input_z_gpu, input_z, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device


	size_t blockDim = sqrt(props.maxThreadsPerBlock);
	//dim3 threads(blockDim, blockDim);
	dim3 threads(16, 16, 4);
	dim3 blocks(img_w / threads.x + 1, img_h / threads.y + 1, img_l / threads.z + 1);

	plus << < blocks, threads >> > (normalize_DIV_gpu, input_x_gpu, input_y_gpu, input_z_gpu, img_w, img_h, img_l);

	HANDLE_ERROR(cudaMemcpy(normalize_DIV, normalize_DIV_gpu, bytes, cudaMemcpyDeviceToHost));


	cudaFree(input_x_gpu);
	cudaFree(input_y_gpu);
	cudaFree(input_z_gpu);
	cudaFree(normalize_DIV_gpu);



}

void normalized_DIV_cal_cross(float* normalize_DIV, float* gradientmagnitude, float* input_x, float* input_y, float* input_z, int img_w, int img_h, int img_l) {


	cudaDeviceProp props;
	HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));


	float* input_x_gpu;
	float* input_y_gpu;
	float* input_z_gpu;
	float* output_x_gpu;
	float* output_y_gpu;
	float* output_z_gpu;
	float* gradientmagnitude_gpu;
	float* output_x_gpu_final;
	float* output_y_gpu_final;
	float* output_z_gpu_final;
	float* normalize_DIV_gpu;

	size_t bytes = (img_w * img_h * img_l) * sizeof(float);
	HANDLE_ERROR(cudaMalloc(&input_x_gpu, bytes));  							    //allocate memory on 
	HANDLE_ERROR(cudaMalloc(&input_y_gpu, bytes));  							    //allocate memory on device
	HANDLE_ERROR(cudaMalloc(&input_z_gpu, bytes));  							    //allocate memory on device
	HANDLE_ERROR(cudaMalloc(&output_x_gpu, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&output_y_gpu, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&output_z_gpu, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&gradientmagnitude_gpu, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&output_x_gpu_final, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&output_y_gpu_final, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&output_z_gpu_final, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&normalize_DIV_gpu, bytes));  							//allocate memory on device


	HANDLE_ERROR(cudaMemcpy(input_x_gpu, input_x, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to 
	HANDLE_ERROR(cudaMemcpy(input_y_gpu, input_y, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device
	HANDLE_ERROR(cudaMemcpy(input_z_gpu, input_z, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device
	HANDLE_ERROR(cudaMemcpy(gradientmagnitude_gpu, gradientmagnitude, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device


	size_t blockDim = sqrt(props.maxThreadsPerBlock);
	dim3 threads(blockDim, blockDim);
	dim3 blocks(img_w / threads.x + 1, img_h / threads.y + 1, img_l / threads.z + 1);

	//gradientmagnitude << < blocks, threads >> > (gradientmagnitude_gpu, input_x_gpu, input_y_gpu, input_z_gpu, img_w, img_h, img_l);
	division_grad << < blocks, threads >> > (output_x_gpu, input_x_gpu, gradientmagnitude_gpu, img_w, img_h, img_l);
	division_grad << < blocks, threads >> > (output_y_gpu, input_y_gpu, gradientmagnitude_gpu, img_w, img_h, img_l);
	division_grad << < blocks, threads >> > (output_z_gpu, input_z_gpu, gradientmagnitude_gpu, img_w, img_h, img_l);
	gradient_dx_cuda << < blocks, threads >> > (output_x_gpu, output_x_gpu_final, img_w, img_h, img_l);
	gradient_dy_cuda << < blocks, threads >> > (output_y_gpu, output_y_gpu_final, img_w, img_h, img_l);
	gradient_dz_cuda << < blocks, threads >> > (output_z_gpu, output_z_gpu_final, img_w, img_h, img_l);
	plus << < blocks, threads >> > (normalize_DIV_gpu, output_x_gpu_final, output_y_gpu_final, output_z_gpu_final, img_w, img_h, img_l);


	HANDLE_ERROR(cudaMemcpy(normalize_DIV, normalize_DIV_gpu, bytes, cudaMemcpyDeviceToHost));

	cudaFree(normalize_DIV_gpu);
	cudaFree(gradientmagnitude_gpu);
	cudaFree(output_z_gpu_final);
	cudaFree(output_y_gpu_final);
	cudaFree(output_x_gpu_final);
	cudaFree(output_z_gpu);
	cudaFree(output_y_gpu);
	cudaFree(output_x_gpu);
	cudaFree(input_z_gpu);
	cudaFree(input_y_gpu);
	cudaFree(input_x_gpu);



}

void gradient_cal_3_input(float* input_x, float* input_y, float* input_z, float* output_x, float* output_y, float* output_z, int img_w, int img_h, int img_l) {


	cudaDeviceProp props;
	HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));


	float* input_x_gpu;
	float* input_y_gpu;
	float* input_z_gpu;
	float* output_x_gpu;
	float* output_y_gpu;
	float* output_z_gpu;

	size_t bytes = (img_w * img_h * img_l) * sizeof(float);
	HANDLE_ERROR(cudaMalloc(&input_x_gpu, bytes));  							    //allocate memory on 
	HANDLE_ERROR(cudaMalloc(&input_y_gpu, bytes));  							    //allocate memory on device
	HANDLE_ERROR(cudaMalloc(&input_z_gpu, bytes));  							    //allocate memory on device
	HANDLE_ERROR(cudaMalloc(&output_x_gpu, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&output_y_gpu, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&output_z_gpu, bytes));  							//allocate memory on device


	HANDLE_ERROR(cudaMemcpy(input_x_gpu, input_x, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to 
	HANDLE_ERROR(cudaMemcpy(input_y_gpu, input_y, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device
	HANDLE_ERROR(cudaMemcpy(input_z_gpu, input_z, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device


	size_t blockDim = sqrt(props.maxThreadsPerBlock);
	dim3 threads(blockDim, blockDim);
	dim3 blocks(img_w / threads.x + 1, img_h / threads.y + 1, img_l / threads.z + 1);

	gradient_dx_cuda << < blocks, threads >> > (input_x_gpu, output_x_gpu, img_w, img_h, img_l);
	gradient_dy_cuda << < blocks, threads >> > (input_y_gpu, output_y_gpu, img_w, img_h, img_l);
	gradient_dz_cuda << < blocks, threads >> > (input_z_gpu, output_z_gpu, img_w, img_h, img_l);



	HANDLE_ERROR(cudaMemcpy(output_x, output_x_gpu, bytes, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(output_y, output_y_gpu, bytes, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(output_z, output_z_gpu, bytes, cudaMemcpyDeviceToHost));


	cudaFree(input_x_gpu);
	cudaFree(input_y_gpu);
	cudaFree(input_z_gpu);
	cudaFree(output_x_gpu);
	cudaFree(output_y_gpu);
	cudaFree(output_z_gpu);


}




__global__ void gradientmagnitude(float* input, float* output_x, float* output_y, float* output_z, int img_w, int img_h, int img_l) {


	size_t i = blockDim.y * blockIdx.y + threadIdx.y;  // Row index
	size_t j = blockDim.x * blockIdx.x + threadIdx.x;  // Column index
	size_t p = blockDim.z * blockIdx.z + threadIdx.z;  // Depth index

	if (i < img_h && j < img_w && p < img_l) {
		size_t index = i * img_w * img_l + j * img_l + p;

		float gradient_x = output_x[index];
		float gradient_y = output_y[index];
		float gradient_z = output_z[index];

		float gradient_magnitude = sqrt(gradient_x * gradient_x + gradient_y * gradient_y + gradient_z * gradient_z);

		input[index] = gradient_magnitude;
	}

}

void gradientmagnitude_cal(float* output, float* input_x, float* input_y, float* input_z, int img_w, int img_h, int img_l) {


	cudaDeviceProp props;
	HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));


	float* output_gpu;
	float* input_x_gpu;
	float* input_y_gpu;
	float* input_z_gpu;

	size_t bytes = (img_w * img_h * img_l) * sizeof(float);
	HANDLE_ERROR(cudaMalloc(&output_gpu, bytes));  							    //allocate memory on device
	HANDLE_ERROR(cudaMalloc(&input_x_gpu, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&input_y_gpu, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&input_z_gpu, bytes));  							//allocate memory on device



	HANDLE_ERROR(cudaMemcpy(input_x_gpu, input_x, bytes, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(input_y_gpu, input_y, bytes, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(input_z_gpu, input_z, bytes, cudaMemcpyHostToDevice));


	size_t blockDim = sqrt(props.maxThreadsPerBlock);
	//dim3 threads(blockDim, blockDim);
	dim3 threads(16, 16, 4);
	dim3 blocks(img_w / threads.x + 1, img_h / threads.y + 1, img_l / threads.z + 1);

	gradientmagnitude << < blocks, threads >> > (output_gpu, input_x_gpu, input_y_gpu, input_z_gpu, img_w, img_h, img_l);


	HANDLE_ERROR(cudaMemcpy(output, output_gpu, bytes, cudaMemcpyDeviceToHost));     //copy the array from main memory to device

	cudaFree(output_gpu);
	cudaFree(input_x_gpu);
	cudaFree(input_y_gpu);
	cudaFree(input_z_gpu);


}



// convolution on device
__global__ void Convolution__on_device(float* out, float* img, float* kernel, int img_w, int img_l, int img_h, int out_w, int out_h, int out_l, int K) {
	size_t i = blockDim.y * blockIdx.y + threadIdx.y;
	size_t j = blockDim.x * blockIdx.x + threadIdx.x;
	size_t p = blockDim.z * blockIdx.z + threadIdx.z;

	// i and j being smaller than output's width and height, manage the edges perfectly
	if (i >= out_h || j >= out_w || p >= out_l) return;

	float conv = 0;
	for (int ki = 0; ki < K; ki++)
		for (int kj = 0; kj < K; kj++)
			for (int kk = 0; kk < K; kk++)
				conv += img[(i + ki) * img_w * img_h + (j + kj) * img_w + (p + kk)] * kernel[ki * K * K + kj * K + kk];

	out[i * out_w * out_h + j * out_w + p] = conv;

}

// convolution on device x
__global__ void Convolution__on_X(float* out, float* img, float* kernel, int img_w, int img_l, int img_h, int out_w, int out_h, int out_l, unsigned int K) {
	size_t i = blockDim.y * blockIdx.y + threadIdx.y;
	size_t j = blockDim.x * blockIdx.x + threadIdx.x;
	size_t p = blockDim.z * blockIdx.z + threadIdx.z;

	// i and j being smaller than output's width and height, manage the edges perfectly
	if (i >= out_h || j >= out_w || p >= out_l) return;

	float conv = 0;
	for (int ki = 0; ki < K; ki++)
		conv += (float)img[(p * img_h + i) * img_w + (j + ki)] * kernel[ki];

	out[(p * out_h + i) * out_w + j] = (float)conv;

}

// convolution on device y
__global__ void Convolution__on_Y(float* out, float* img, float* kernel, int img_w, int img_l, int img_h, int out_w, int out_h, int out_l, unsigned int K) {
	size_t i = blockDim.y * blockIdx.y + threadIdx.y;
	size_t j = blockDim.x * blockIdx.x + threadIdx.x;
	size_t p = blockDim.z * blockIdx.z + threadIdx.z;

	// i and j being smaller than output's width and height, manage the edges perfectly
	if (i >= out_h || j >= out_w || p >= out_l) return;

	float conv = 0;
	for (int ki = 0; ki < K; ki++)
		conv += (float)img[(p * img_h + (i + ki)) * img_w + j] * kernel[ki];

	out[(p * out_h + i) * out_w + j] = (float)conv;

}


// convolution on device z
__global__ void Convolution__on_Z(float* out, float* img, float* kernel, int img_w, int img_l, int img_h, int out_w, int out_h, int out_l, unsigned int K) {
	size_t i = blockDim.y * blockIdx.y + threadIdx.y;
	size_t j = blockDim.x * blockIdx.x + threadIdx.x;
	size_t p = blockDim.z * blockIdx.z + threadIdx.z;

	// i and j being smaller than output's width and height, manage the edges perfectly
	if (i >= out_h || j >= out_w || p >= out_l) return;

	float conv = 0;
	for (int ki = 0; ki < K; ki++)
		conv += img[((p + ki) * img_h + i) * img_w + j] * kernel[ki];

	out[(p * out_h + i) * out_w + j] = conv;

}

void adddevice_convolution_seperable(float* output, float* in_img, int img_w, int img_h, int img_l, float sigma, float* gkernel, unsigned int k_size) {


	cudaDeviceProp props;
	HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));

	int x_height = img_h;
	int x_width = img_w - k_size + 1;
	int x_length = img_l;
	int x_size = x_height * x_width * x_length;

	int y_height = img_h - k_size + 1;
	int y_width = img_w - k_size + 1;
	int y_length = img_l;
	int y_size = y_height * y_width * y_length;

	int z_height = img_h - k_size + 1;
	int z_width = img_w - k_size + 1;
	int z_length = img_l - k_size + 1;
	int z_size = z_height * z_width * z_length;
	//y_output = (float*)malloc(y_size * sizeof(float));


	float* gkernel_gpu;
	float* input_img_gpu;
	float* gpu_output_x;
	float* gpu_output_y;
	float* gpu_output_z;
	size_t bytes = (img_w * img_h * img_l) * sizeof(float);


	HANDLE_ERROR(cudaMalloc(&gkernel_gpu, k_size * sizeof(float)));
	HANDLE_ERROR(cudaMalloc(&input_img_gpu, bytes));  							    //allocate memory on device
	HANDLE_ERROR(cudaMalloc(&gpu_output_x, x_size * sizeof(float)));  				//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&gpu_output_y, y_size * sizeof(float)));  				//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&gpu_output_z, z_size * sizeof(float)));  				//allocate memory on device

	HANDLE_ERROR(cudaMemcpy(input_img_gpu, in_img, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device
	HANDLE_ERROR(cudaMemcpy(gkernel_gpu, gkernel, k_size * sizeof(float), cudaMemcpyHostToDevice));     //copy the array from main memory to device


	//// convolving along x
	//Convolution_x_on_device << < blocks, threads >> > (gpu_output_x, input_img_gpu, gkernel_gpu, img_w, x_height, x_width, k);
	//// convolving along y
	//Convolution__on_device(float* out, float* img, float* kernel, int img_w, int img_l, int out_w, int out_h, int out_l, int K)
	size_t blockDim = sqrt(props.maxThreadsPerBlock);
	//dim3 threads(blockDim, blockDim);
	dim3 threads(16, 16, 4);
	dim3 blocks(img_w / threads.x + 1, img_h / threads.y + 1, img_l / threads.z + 1);
	//dim3 blocks((x_width + blockDim - 1) / blockDim, (x_height + blockDim - 1) / blockDim, (x_length + blockDim - 1) / blockDim);


	Convolution__on_X << < blocks, threads >> > (gpu_output_x, input_img_gpu, gkernel_gpu, img_w, img_l, img_h, x_width, x_height, x_length, k_size);
	Convolution__on_Y << < blocks, threads >> > (gpu_output_y, gpu_output_x, gkernel_gpu, x_width, x_length, x_height, y_width, y_height, y_length, k_size);
	Convolution__on_Z << < blocks, threads >> > (gpu_output_z, gpu_output_y, gkernel_gpu, y_width, y_length, y_height, z_width, z_height, z_length, k_size);
	//float* out, float* img, float* kernel, int img_w, int out_h, int out_w, int K
	// copy convolved outputs from Device to main memory
	//HANDLE_ERROR(cudaMemcpy(x_output, gpu_output_x, x_size * sizeof(float), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(output, gpu_output_z, z_size * sizeof(float), cudaMemcpyDeviceToHost));




	cudaFree(gkernel_gpu);
	cudaFree(input_img_gpu);
	cudaFree(gpu_output_x);
	cudaFree(gpu_output_y);
	cudaFree(gpu_output_z);


}


void adddevice_convolution(float* y_output, float* in_img, int img_w, int img_h, int img_l, float sigma, float* gkernel, unsigned int k_size) {


	cudaDeviceProp props;
	HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));



	int y_height = img_h - k_size + 1;
	int y_width = img_w - k_size + 1;
	int y_length = img_l - k_size + 1;
	int y_size = y_height * y_width * y_length;
	//y_output = (float*)malloc(y_size * sizeof(float));


	float* gkernel_gpu;
	float* input_img_gpu;
	float* gpu_output_y;
	size_t bytes = (img_w * img_h * img_l) * sizeof(float);


	HANDLE_ERROR(cudaMalloc(&gkernel_gpu, k_size * k_size * k_size * sizeof(float)));
	HANDLE_ERROR(cudaMalloc(&input_img_gpu, bytes));  							    //allocate memory on device
	HANDLE_ERROR(cudaMalloc(&gpu_output_y, y_size * sizeof(float)));  				//allocate memory on device


	HANDLE_ERROR(cudaMemcpy(input_img_gpu, in_img, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device
	HANDLE_ERROR(cudaMemcpy(gkernel_gpu, gkernel, k_size * k_size * k_size * sizeof(float), cudaMemcpyHostToDevice));     //copy the array from main memory to device


	//// convolving along x
	//Convolution_x_on_device << < blocks, threads >> > (gpu_output_x, input_img_gpu, gkernel_gpu, img_w, x_height, x_width, k);
	//// convolving along y
	//Convolution__on_device(float* out, float* img, float* kernel, int img_w, int img_l, int out_w, int out_h, int out_l, int K)
	size_t blockDim = sqrt(props.maxThreadsPerBlock);
	//dim3 threads(blockDim, blockDim);
	dim3 threads(16, 16, 4);
	dim3 blocks(img_w / threads.x, img_h / threads.y, img_l / threads.z);

	Convolution__on_device << < blocks, threads >> > (gpu_output_y, input_img_gpu, gkernel_gpu, img_w, img_l, img_h, y_width, y_height, y_length, k_size);
	//float* out, float* img, float* kernel, int img_w, int out_h, int out_w, int K
	// copy convolved outputs from Device to main memory
	//HANDLE_ERROR(cudaMemcpy(x_output, gpu_output_x, x_size * sizeof(float), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(y_output, gpu_output_y, y_size * sizeof(float), cudaMemcpyDeviceToHost));



	cudaFree(gpu_output_y);
	cudaFree(gkernel_gpu);
	cudaFree(input_img_gpu);


}

void adddevice(float* input, float* fout, float* fin, float* Eo, float* Ei, int img_w, int img_h, int img_l, float sigma) {


	cudaDeviceProp props;
	HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));


	/*size_t blockDim = sqrt(props.maxThreadsPerBlock);
	dim3 threads(blockDim, blockDim);
	dim3 blocks(img_w / threads.x   , img_h / threads.y , img_l / threads.z );*/


	float* Eo_gpu;
	float* Ei_gpu;
	float* fo_gpu;
	float* fi_gpu;
	float* input_img_gpu;
	size_t bytes = (img_w * img_h * img_l) * sizeof(float);
	HANDLE_ERROR(cudaMalloc(&input_img_gpu, bytes));  							    //allocate memory on device
	HANDLE_ERROR(cudaMalloc(&fo_gpu, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&fi_gpu, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&Eo_gpu, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&Ei_gpu, bytes));  							//allocate memory on device

	HANDLE_ERROR(cudaMemcpy(input_img_gpu, input, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device
	HANDLE_ERROR(cudaMemcpy(fo_gpu, fout, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device
	HANDLE_ERROR(cudaMemcpy(fi_gpu, fin, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device

	size_t blockDim = sqrt(props.maxThreadsPerBlock);
	//dim3 threads(blockDim, blockDim);
	dim3 threads(16, 16, 4);
	dim3 blocks(img_w / threads.x + 1, img_h / threads.y + 1, img_l / threads.z + 1);

	Eout_Ein_calculation << < blocks, threads >> > (Eo_gpu, Ei_gpu, fo_gpu, fi_gpu, input_img_gpu, img_w, img_h, img_l, sigma);
	// float* Eo_gpu, float* Ei_gpu, float* fo_gpu, float* fi_gpu, float* input_img_gpu, float* input, float* fout, float* fin, int img_w, int img_h, int img_l, int sigma

	HANDLE_ERROR(cudaMemcpy(Eo, Eo_gpu, bytes, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(Ei, Ei_gpu, bytes, cudaMemcpyDeviceToHost));

	/*for (int i = 0; i < 10; i++)
	{
		std::cout << Eo[i] << std::endl;
	}

	std::cout << "adddevice\n";*/
	//cudaDeviceSynchronize();
	/*free(input);
	free(fout);
	free(fin);
	free(Eo);
	free(Ei);*/
	cudaFree(Eo_gpu);
	cudaFree(Ei_gpu);
	cudaFree(fo_gpu);
	cudaFree(fi_gpu);
	cudaFree(input_img_gpu);


}





void All(float* output, float* phi, float* input_image, int img_w, int img_h, int img_l, float sigma, float* gkernel, float e, float dt, float v, float meu ) {


	cudaDeviceProp props;
	HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));


	float* HD_out_gpu;
	float* phi_gpu;
	float* input_image_gpu;
	float* I_out_gpu;
	float* HD_in_gpu;
	float* I_in_gpu;
	float* dPHI_dx_gpu;
	float* dPHI_dy_gpu;
	float* dPHI_dz_gpu;
	float* d2PHI_dx2_1_gpu;
	float* d2PHI_dy2_1_gpu;
	float* d2PHI_dz2_1_gpu;
	float* Gradientmagnitude_phi_gpu;
	float* d2PHI_dx2_gpu;
	float* d2PHI_dy2_gpu;
	float* d2PHI_dz2_gpu;
	float* normalize_DIV_gpu;
	float* DIV_gpu;
	float* I_out_blurred_gpu;
	float* HD_out_blurred_gpu;
	float* I_in_blurred_gpu;
	float* HD_in_blurred_gpu;
	float* I_ob_gpu;
	float* HD_ob_gpu;
	float* I_ib_gpu;
	float* HD_ib_gpu;
	float* fout_x_gpu;
	float* fin_x_gpu;
	float* Eout_gpu;
	float* Ein_gpu;
	float* derivative_heaviside_gpu;
	float* gkernel_gpu;
	float* gpu_output_x;
	float* gpu_output_y;
	float* gpu_output_z;
	float* output_gpu;

	int k_size = 7 * sigma;

	int pad_w  = (k_size - 1) / 2;

	int x_height = img_h;
	int x_width = img_w - k_size + 1;
	int x_length = img_l;
	int x_size = x_height * x_width * x_length;

	int y_height = img_h - k_size + 1;
	int y_width = img_w - k_size + 1;
	int y_length = img_l;
	int y_size = y_height * y_width * y_length;

	int z_height = img_h - k_size + 1;
	int z_width = img_w - k_size + 1;
	int z_length = img_l - k_size + 1;
	int z_size = z_height * z_width * z_length;

	int padded_w = img_w + 2 * pad_w;
	int padded_h = img_h + 2 * pad_w;
	int padded_l = img_l + 2 * pad_w;

	size_t bytes = (img_w * img_h * img_l) * sizeof(float);
	HANDLE_ERROR(cudaMalloc(&HD_out_gpu, bytes));  							     
	HANDLE_ERROR(cudaMalloc(&phi_gpu, bytes));  							    //allocate memory on device
	HANDLE_ERROR(cudaMalloc(&input_image_gpu, bytes));
	HANDLE_ERROR(cudaMalloc(&I_out_gpu, bytes));
	HANDLE_ERROR(cudaMalloc(&HD_in_gpu, bytes));
	HANDLE_ERROR(cudaMalloc(&I_in_gpu, bytes));
	HANDLE_ERROR(cudaMalloc(&dPHI_dx_gpu, bytes));
	HANDLE_ERROR(cudaMalloc(&dPHI_dy_gpu, bytes));
	HANDLE_ERROR(cudaMalloc(&dPHI_dz_gpu, bytes));
	HANDLE_ERROR(cudaMalloc(&Gradientmagnitude_phi_gpu, bytes));
	HANDLE_ERROR(cudaMalloc(&d2PHI_dx2_1_gpu, bytes));
	HANDLE_ERROR(cudaMalloc(&d2PHI_dy2_1_gpu, bytes));
	HANDLE_ERROR(cudaMalloc(&d2PHI_dz2_1_gpu, bytes));
	HANDLE_ERROR(cudaMalloc(&d2PHI_dx2_gpu, bytes));
	HANDLE_ERROR(cudaMalloc(&d2PHI_dy2_gpu, bytes));
	HANDLE_ERROR(cudaMalloc(&d2PHI_dz2_gpu, bytes));
	HANDLE_ERROR(cudaMalloc(&normalize_DIV_gpu, bytes));
	HANDLE_ERROR(cudaMalloc(&DIV_gpu, bytes));
	HANDLE_ERROR(cudaMalloc(&I_out_blurred_gpu, bytes));
	HANDLE_ERROR(cudaMalloc(&I_in_blurred_gpu, bytes));
	HANDLE_ERROR(cudaMalloc(&HD_out_blurred_gpu, bytes));
	HANDLE_ERROR(cudaMalloc(&HD_in_blurred_gpu, bytes));
	HANDLE_ERROR(cudaMalloc(&I_ob_gpu, bytes));
	HANDLE_ERROR(cudaMalloc(&I_ib_gpu, bytes));
	HANDLE_ERROR(cudaMalloc(&HD_ob_gpu, bytes));
	HANDLE_ERROR(cudaMalloc(&HD_ib_gpu, bytes));
	HANDLE_ERROR(cudaMalloc(&fout_x_gpu, bytes));
	HANDLE_ERROR(cudaMalloc(&fin_x_gpu, bytes));
	HANDLE_ERROR(cudaMalloc(&Eout_gpu, bytes));
	HANDLE_ERROR(cudaMalloc(&Ein_gpu, bytes));
	HANDLE_ERROR(cudaMalloc(&derivative_heaviside_gpu, bytes));
	HANDLE_ERROR(cudaMalloc(&gkernel_gpu, k_size * sizeof(float)));
	HANDLE_ERROR(cudaMalloc(&gpu_output_x, x_size * sizeof(float)));  				
	HANDLE_ERROR(cudaMalloc(&gpu_output_y, y_size * sizeof(float)));  				
	HANDLE_ERROR(cudaMalloc(&gpu_output_z, z_size * sizeof(float)));

	HANDLE_ERROR(cudaMemcpy(phi_gpu, phi, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device
	HANDLE_ERROR(cudaMemcpy(input_image_gpu, input_image, bytes, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(gkernel_gpu, gkernel, k_size * sizeof(float), cudaMemcpyHostToDevice));

	size_t blockDim = sqrt(props.maxThreadsPerBlock);
	dim3 threads(16, 16, 4);
	dim3 blocks(img_w / threads.x + 1, img_h / threads.y + 1, img_l / threads.z + 1);




	heaviside << < blocks, threads >> > (HD_out_gpu, phi_gpu, img_w, img_h, img_l);
	multiplication << < blocks, threads >> > (I_out_gpu, input_image_gpu, HD_out_gpu, img_w, img_h, img_l);
	inverse << <blocks, threads >> > (HD_in_gpu, HD_out_gpu, img_w, img_h, img_l);
	multiplication << < blocks, threads >> > (I_in_gpu, input_image_gpu, HD_in_gpu, img_w, img_h, img_l);
	gradient_dx_cuda << < blocks, threads >> > (phi_gpu, dPHI_dx_gpu, img_w, img_h, img_l);
	gradient_dy_cuda << < blocks, threads >> > (phi_gpu, dPHI_dy_gpu, img_w, img_h, img_l);
	gradient_dz_cuda << < blocks, threads >> > (phi_gpu, dPHI_dz_gpu, img_w, img_h, img_l);
	gradientmagnitude << < blocks, threads >> > (Gradientmagnitude_phi_gpu, dPHI_dx_gpu, dPHI_dy_gpu, dPHI_dz_gpu, img_w, img_h, img_l);
	division_grad << < blocks, threads >> > (d2PHI_dx2_1_gpu, dPHI_dx_gpu, Gradientmagnitude_phi_gpu, img_w, img_h, img_l);
	division_grad << < blocks, threads >> > (d2PHI_dy2_1_gpu, dPHI_dy_gpu, Gradientmagnitude_phi_gpu, img_w, img_h, img_l);
	division_grad << < blocks, threads >> > (d2PHI_dz2_1_gpu, dPHI_dz_gpu, Gradientmagnitude_phi_gpu, img_w, img_h, img_l);
	gradient_dx_cuda << < blocks, threads >> > (d2PHI_dx2_1_gpu, d2PHI_dx2_gpu, img_w, img_h, img_l);
	gradient_dy_cuda << < blocks, threads >> > (d2PHI_dy2_1_gpu, d2PHI_dy2_gpu, img_w, img_h, img_l);
	gradient_dz_cuda << < blocks, threads >> > (d2PHI_dz2_1_gpu, d2PHI_dz2_gpu, img_w, img_h, img_l);
	plus << < blocks, threads >> > (normalize_DIV_gpu, d2PHI_dx2_gpu, d2PHI_dy2_gpu, d2PHI_dz2_gpu, img_w, img_h, img_l);
	plus << < blocks, threads >> > (DIV_gpu, dPHI_dx_gpu, dPHI_dy_gpu, dPHI_dz_gpu, img_w, img_h, img_l);
	Convolution__on_X << < blocks, threads >> > (gpu_output_x, I_out_gpu, gkernel_gpu, img_w, img_l, img_h, x_width, x_height, x_length, k_size);
	Convolution__on_Y << < blocks, threads >> > (gpu_output_y, gpu_output_x, gkernel_gpu, x_width, x_length, x_height, y_width, y_height, y_length, k_size);
	Convolution__on_Z << < blocks, threads >> > (I_out_blurred_gpu, gpu_output_y, gkernel_gpu, y_width, y_length, y_height, z_width, z_height, z_length, k_size);
	Convolution__on_X << < blocks, threads >> > (gpu_output_x, HD_out_gpu, gkernel_gpu, img_w, img_l, img_h, x_width, x_height, x_length, k_size);
	Convolution__on_Y << < blocks, threads >> > (gpu_output_y, gpu_output_x, gkernel_gpu, x_width, x_length, x_height, y_width, y_height, y_length, k_size);
	Convolution__on_Z << < blocks, threads >> > (HD_out_blurred_gpu, gpu_output_y, gkernel_gpu, y_width, y_length, y_height, z_width, z_height, z_length, k_size);
	Convolution__on_X << < blocks, threads >> > (gpu_output_x, I_in_gpu, gkernel_gpu, img_w, img_l, img_h, x_width, x_height, x_length, k_size);
	Convolution__on_Y << < blocks, threads >> > (gpu_output_y, gpu_output_x, gkernel_gpu, x_width, x_length, x_height, y_width, y_height, y_length, k_size);
	Convolution__on_Z << < blocks, threads >> > (I_in_blurred_gpu, gpu_output_y, gkernel_gpu, y_width, y_length, y_height, z_width, z_height, z_length, k_size);
	Convolution__on_X << < blocks, threads >> > (gpu_output_x, HD_in_gpu, gkernel_gpu, img_w, img_l, img_h, x_width, x_height, x_length, k_size);
	Convolution__on_Y << < blocks, threads >> > (gpu_output_y, gpu_output_x, gkernel_gpu, x_width, x_length, x_height, y_width, y_height, y_length, k_size);
	Convolution__on_Z << < blocks, threads >> > (HD_in_blurred_gpu, gpu_output_y, gkernel_gpu, y_width, y_length, y_height, z_width, z_height, z_length, k_size);
	pad_border << <blocks, threads >> > (I_ob_gpu, I_out_blurred_gpu, img_w, img_h, img_l, pad_w, padded_w, padded_h, padded_l);
	pad_border << <blocks, threads >> > (HD_ob_gpu, HD_out_blurred_gpu, img_w, img_h, img_l, pad_w, padded_w, padded_h, padded_l);
	pad_border << <blocks, threads >> > (I_in_gpu, I_in_blurred_gpu, img_w, img_h, img_l, pad_w, padded_w, padded_h, padded_l);
	pad_border << <blocks, threads >> > (HD_in_gpu, HD_in_blurred_gpu, img_w, img_h, img_l, pad_w, padded_w, padded_h, padded_l);
	division << < blocks, threads >> > (fout_x_gpu, I_ob_gpu, HD_ob_gpu, img_w, img_h, img_l);
	division << < blocks, threads >> > (fin_x_gpu, I_ib_gpu, HD_ib_gpu, img_w, img_h, img_l);
	Eout_Ein_calculation << < blocks, threads >> > (Eout_gpu, Ein_gpu, fout_x_gpu, fin_x_gpu, input_image_gpu, img_w, img_h, img_l, sigma);
	deri_heaviside << < blocks, threads >> > (derivative_heaviside_gpu, phi_gpu, img_w, img_h, img_l);
	Final_phi << < blocks, threads >> > (output_gpu, phi_gpu, derivative_heaviside_gpu, Eout_gpu, Ein_gpu, normalize_DIV_gpu, DIV_gpu, e, dt, v, meu, img_w, img_h, img_l);


	HANDLE_ERROR(cudaMemcpy(output, output_gpu, bytes, cudaMemcpyDeviceToHost));

	cudaFree(HD_out_gpu);
	cudaFree(phi_gpu);
	cudaFree(input_image_gpu);
	cudaFree(I_out_gpu);
	cudaFree(HD_in_gpu);
	cudaFree(I_in_gpu);
	cudaFree(dPHI_dx_gpu);
	cudaFree(dPHI_dy_gpu);
	cudaFree(dPHI_dz_gpu);
	cudaFree(d2PHI_dx2_1_gpu);
	cudaFree(d2PHI_dy2_1_gpu);
	cudaFree(d2PHI_dz2_1_gpu);
	cudaFree(Gradientmagnitude_phi_gpu);
	cudaFree(d2PHI_dx2_gpu);
	cudaFree(d2PHI_dy2_gpu);
	cudaFree(d2PHI_dz2_gpu);
	cudaFree(normalize_DIV_gpu);
	cudaFree(DIV_gpu);
	cudaFree(I_out_blurred_gpu);
	cudaFree(HD_out_blurred_gpu);
	cudaFree(I_in_blurred_gpu);
	cudaFree(HD_in_blurred_gpu);
	cudaFree(I_ob_gpu);
	cudaFree(HD_ob_gpu);
	cudaFree(I_ib_gpu);
	cudaFree(HD_ib_gpu);
	cudaFree(fout_x_gpu);
	cudaFree(fin_x_gpu);
	cudaFree(Eout_gpu);
	cudaFree(Ein_gpu);
	cudaFree(derivative_heaviside_gpu);
	cudaFree(gkernel_gpu);
	cudaFree(gpu_output_x);
	cudaFree(gpu_output_y);
	cudaFree(gpu_output_z);
	cudaFree(output_gpu);

}
