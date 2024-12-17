#include <vector>
#include <array>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include<string>
#include <cstdlib>
#include<stdio.h>
#include "tira/image.h"
#include "tira/volume.h"
#include "tira/image/colormap.h"
#include<complex>
#include<cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h> 
#include<Windows.h>
#include<boost/program_options.hpp>
#include <stack>
#include<chrono>
#include"tira/volume.h"


using namespace std;


float normaldistance(float x, float sigma)
{
	float y = 1.0f / (sigma * sqrt(2 * 3.14159));
	float ex = -(x * x) / (2 * sigma * sigma);
	return y * exp(ex);
}

inline float normaldist(float x, float sigma) {
	float scale = 1.0f / (sigma * sqrt(2 * 3.14159));
	float ex = -(x * x) / (2 * sigma * sigma);
	return (scale * exp(ex));
}



float sigma;
int T;
float dt;
float meu;
float v;
float N = 255 * 255;
float v1 = N * v;
float e;
float sigma2;
std::string centerline;
std::string input_volume;
std::string output_filename;
bool debug = false;
void adddevice_convolution(float* y_output, float* in_img, int img_w, int img_h, int img_l, float sigma, float* gkernel, unsigned int k_size);
void adddevice(float* input, float* fout, float* fin, float* Eo, float* Ei, int img_w, int img_h, int img_l, float sigma);
void adddevice_convolution_seperable(float* output, float* in_img, int img_w, int img_h, int img_l, float sigma, float* gkernel, unsigned int k_size);
void gradient_cal(float* input, float* output_x, float* output_y, float* output_z, int img_w, int img_h, int img_l);
void gradient_cal_3_input(float* input_x, float* input_y, float* input_z, float* output_x, float* output_y, float* output_z, int img_w, int img_h, int img_l);
void gradientmagnitude_cal(float* output, float* input_x, float* input_y, float* input_z, int img_w, int img_h, int img_l);
void DIV_cal(float* DIV, float* input_x, float* input_y, float* input_z, int img_w, int img_h, int img_l);
void division_nDIV(float* input_x, float* input_y, float* input_z, float* gradientmagnitude, float* output_x, float* output_y, float* output_z, int img_w, int img_h, int img_l);
void normalized_div_plus(float* normalize_DIV, float* input_x, float* input_y, float* input_z, int img_w, int img_h, int img_l);
void division_cal(float* output, float* input_1, float* input_2, int img_w, int img_h, int img_l);
void multiplication_cal(float* output, float* input_1, float* input_2, int img_w, int img_h, int img_l);
void heaviside_cal(float* output, float* input, int img_w, int img_h, int img_l);
void deri_heaviside_cal(float* output, float* input, int img_w, int img_h, int img_l);
void inverse_cal(float* output, float* input, int img_w, int img_h, int img_l);
void Final_PHI_cal(float* output, float* phi, float* deri_heav, float* Eout, float* Ein, float* normalized_DIV, float* DIV, float e, float dt, float v, float meu, int img_w, int img_h, int img_l);
void pad_border_cal(float* output, float* input, int img_w, int img_h, int img_l, int pad_w);

int main(int argc, char** argv) {

	//MEHER: set up command line arguments
	// What arguments do you need?
	// input image (numpy array)
	// binary seed image (numpy array)
	// output phi (numpy array)
	// 
	// Come up with a list of all other arguments here, along with default values.
	// Make these arguments global variables with the prefix "in_". If you have arguments like "time_step, sigma"
	// then make the variables that store these values "in_timestep, in_sigma"
	// 
	//creating gaussian kernel

	// Declare the supported options.
	boost::program_options::options_description desc("Allowed options");
	desc.add_options()
		("help", "produce help message")
		("sigma", boost::program_options::value<float >(&sigma)->default_value(5.0), "blur kernel for the fitting term (in pixels)")
		("t", boost::program_options::value<int>(&T)->default_value(30), "total number of time steps")
		("dt", boost::program_options::value<float >(&dt)->default_value(0.1), "time step size")
		("meu", boost::program_options::value<float >(&meu)->default_value(1.0), "weight for the regularization term")
		("e", boost::program_options::value<float >(&e)->default_value(0.1), "weight for the fitting term")
		("v", boost::program_options::value<float >(&v)->default_value(0.0001), "weight for the smoothing term")
		("sigma2", boost::program_options::value<float >(&sigma2)->default_value(0.00), "weight for sigma2")
		("binary", boost::program_options::value<std::string>(&centerline), "binary seed surface (.npy)")
		("image", boost::program_options::value<std::string>(&input_volume), "input image filename (.npy)")
		("ouput", boost::program_options::value<std::string>(&output_filename)->default_value("out.npy"), "output level set (.npy)")
		("debug", "output all debugging details")
		;
	boost::program_options::variables_map vm;
	boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).style(
		boost::program_options::command_line_style::unix_style ^ boost::program_options::command_line_style::allow_short
	).run(), vm);
	boost::program_options::notify(vm);

	if (vm.count("help")) {
		std::cout << desc << std::endl;
		return 1;
	}

	if (vm.count("debug")) {
		debug = true;
	}


	//Creating 3D Gaussian Kernel

	unsigned int size = (sigma * 7);
	float dx = 1.0f;
	float start = -(float)(size - 1) / 2.0f;

	tira::volume<float>K(size, size, size);
	for (size_t yi = 0; yi < size; yi++)
	{
		float gy = normaldist(start + dx * yi, sigma);

		for (size_t xi = 0; xi < size; xi++)
		{
			float gx = normaldist(start + dx * xi, sigma);
			for (size_t zi = 0; zi < size; zi++)
			{
				float gz = normaldist(start + dx * zi, sigma);
				K(xi, yi, zi) = gx * gy * gz;
			}
		}

	}


	// load initial phi
	tira::volume<float> initial_PHI;
	initial_PHI.load_npy(centerline);

	// load input image
	tira::volume<float> input_image;
	input_image.load_npy(input_volume);

	//Calculating kernel size
	unsigned int p = 7 * sigma;

	if (p % 2 == 0) p++;
	float miu = p / 2;




	//Evaluating the Gaussian kernel by defining a float pointer
	float* gkernel = (float*)malloc(p * sizeof(float));

	for (int xi = 0; xi < p; xi++) {
		int u = 2 * sigma * sigma;
		gkernel[xi] = 1 / sqrt(u * (float)3.14159265358979323846) * exp(-(xi - miu) * (xi - miu) / u);
	}

	int width = input_image.X();
	int height = input_image.Y();
	int length = input_image.Z();



	for (int yi = 0; yi < height; yi++) {
		for (int xi = 0; xi < width; xi++) {
			for (int zi = 0; zi < length; zi++) {
				initial_PHI(xi, yi, zi) = initial_PHI(xi, yi, zi) - 13.0;
			}
		}
	}

	//Create Signed Distance Field from initial phi
	tira::volume<float>phi = initial_PHI.sdf();


	// intializing volumes
	tira::volume<float> HD_out(width, height, length);
	tira::volume<float> I_out(width, height, length);
	tira::volume<float> HD_in(width, height, length);
	tira::volume<float> I_in(width, height, length);
	tira::volume<float> I_out_blurred(width - (K.X() - 1), height - (K.Y() - 1), length - (K.Z() - 1));
	tira::volume<float> HD_out_blurred(width - (K.X() - 1), height - (K.Y() - 1), length - (K.Z() - 1));
	tira::volume<float> I_in_blurred(width - (K.X() - 1), height - (K.Y() - 1), length - (K.Z() - 1));
	tira::volume<float> HD_in_blurred(width - (K.X() - 1), height - (K.Y() - 1), length - (K.Z() - 1));
	tira::volume<float> fout_x(width, height, length);
	tira::volume<float> fin_x(width, height, length);
	tira::volume<float> Eout(width, height, length);
	tira::volume<float> Ein(width, height, length);
	tira::volume<float> dPHI_dx(width, height, length);
	tira::volume<float> dPHI_dy(width, height, length);
	tira::volume<float> dPHI_dz(width, height, length);
	tira::volume<float>Gradientmagnitude_phi(width, height, length);
	tira::volume<float> d2PHI_dx2_1(width, height, length);
	tira::volume<float> d2PHI_dy2_1(width, height, length);
	tira::volume<float> d2PHI_dz2_1(width, height, length);
	tira::volume<float> normalized_DIV(width, height, length);
	tira::volume<float> DIV(width, height, length);
	tira::volume<float> d2PHI_dx2(width, height, length);
	tira::volume<float> d2PHI_dy2(width, height, length);
	tira::volume<float> d2PHI_dz2(width, height, length);
	tira::volume<float> derivative_heaviside(width, height, length);
	tira::volume<float>I_ob(width, height, length);
        tira::volume<float>HD_ob(width, height, length);
        tira::volume<float>I_ib(width, height, length);
        tira::volume<float>HD_ib(width, height, length);



	
	//starting the main for loop

	for (int t = 0; t < T; t++) {

		// creating timer for GPU
		cudaEvent_t c_start;
		cudaEvent_t c_stop;
		cudaEventCreate(&c_start);
		cudaEventCreate(&c_stop);
		cudaEventRecord(c_start, NULL);



		//applying heaviside function on initial phi
		heaviside_cal(HD_out.data(), phi.data(), width, height, length);

		// multiplying  heaviside_image with input_image

		multiplication_cal(I_out.data(), input_image.data(), HD_out.data(), width, height, length);

		//inverse of heaviside image

		inverse_cal(HD_in.data(), HD_out.data(), width, height, length);

		// multiplying  inverse of heaviside_image with input_image

		multiplication_cal(I_in.data(), input_image.data(), HD_in.data(), width, height, length);

		// calculating divergence

		gradient_cal(phi.data(), dPHI_dx.data(), dPHI_dy.data(), dPHI_dz.data(), width, height, length);

		gradientmagnitude_cal(Gradientmagnitude_phi.data(), dPHI_dx.data(), dPHI_dy.data(), dPHI_dz.data(), width, height, length);

		division_nDIV(dPHI_dx.data(), dPHI_dy.data(), dPHI_dz.data(), Gradientmagnitude_phi.data(), d2PHI_dx2_1.data(), d2PHI_dy2_1.data(), d2PHI_dz2_1.data(), width, height, length);

		gradient_cal_3_input(d2PHI_dx2_1.data(), d2PHI_dy2_1.data(), d2PHI_dz2_1.data(), d2PHI_dx2.data(), d2PHI_dy2.data(), d2PHI_dz2.data(), width, height, length);

		normalized_div_plus(normalized_DIV.data(), d2PHI_dx2.data(), d2PHI_dy2.data(), d2PHI_dz2.data(), width, height, length);

		DIV_cal(DIV.data(), dPHI_dx.data(), dPHI_dy.data(), dPHI_dz.data(), width, height, length);



		int k_size = (7 * sigma);


		// fout_x calculation
		adddevice_convolution_seperable(I_out_blurred.data(), I_out.data(), width, height, length, sigma, gkernel, p);

		adddevice_convolution_seperable(HD_out_blurred.data(), HD_out.data(), width, height, length, sigma, gkernel, p);

                //Border Padding
		float m = (k_size - 1) / 2;
		pad_border_cal(I_ob.data(), I_out_blurred.data(), width - (K.X() - 1), height - (K.Y() - 1), length - (K.Z() - 1), m);
                pad_border_cal(HD_ob.data(), HD_out_blurred.data(), width - (K.X() - 1), height - (K.Y() - 1), length - (K.Z() - 1), m);

		division_cal(fout_x.data(), I_ob.data(), HD_ob.data(), width, height, length);


		//fin_x calculation

		adddevice_convolution_seperable(I_in_blurred.data(), I_in.data(), width, height, length, sigma, gkernel, p);
		adddevice_convolution_seperable(HD_in_blurred.data(), HD_in.data(), width, height, length, sigma, gkernel, p);

		//Border Padding
		pad_border_cal(I_ib.data(), I_in_blurred.data(), width - (K.X() - 1), height - (K.Y() - 1), length - (K.Z() - 1), m);
		pad_border_cal(HD_ib.data(), HD_in_blurred.data(), width - (K.X() - 1), height - (K.Y() - 1), length - (K.Z() - 1), m);

		division_cal(fin_x.data(), I_ib.data(), HD_ib.data(), width, height, length);

		// Eout and Ein calculation

		adddevice(input_image.data(), fout_x.data(), fin_x.data(), Eout.data(), Ein.data(), width, height, length, sigma2);

		// Calculating Derivative of Heaviside function
		deri_heaviside_cal(derivative_heaviside.data(), phi.data(), width, height, length);


		//Updating Final Phi
		Final_PHI_cal(phi.data(), phi.data(), derivative_heaviside.data(), Eout.data(), Ein.data(), normalized_DIV.data(), DIV.data(), e, dt, v, meu, width, height, length);


		//// timer ends

		cudaEventRecord(c_stop, NULL);
		cudaEventSynchronize(c_stop);
		float time_difference_gpu;
		cudaEventElapsedTime(&time_difference_gpu, c_start, c_stop);
		std::cout << "It takes " << time_difference_gpu << " ms to calculate " << std::endl;




	}

	phi.save_npy(output_filename);
	return 0;

}
