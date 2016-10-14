// kernel.cu
//This is the project for optimizing memory access stride.
//-------------------------------------------------------------------
#include "header.h"
#define SHARED_TEST
#include <time.h>
#include <stdlib.h>

#define N 16
#define BLK_SIZE 4 //blocksize(ThreadDim)
#define WIN_SIZE 2


//ORIGINAL KERNEL FOR REFERENCE

#ifdef DEFAULT
__global__ void A5_fast_lo_stats_kernel(float* xVal, float* outStd, float* outSkw, float* outKrt)
{
	//Declarations
	float xVal_local[256];
	float mean = 0, stdev = 0, skw = 0, krt = 0, stmp = 0;
	int iB, jB;
	//https://cs.calvin.edu/courses/cs/374/CUDA/CUDA-Thread-Indexing-Cheatsheet.pdf
	//int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	//int threadId = blockId * (blockDim.x * blockDim.y)+ (threadIdx.y * blockDim.x) + threadIdx.x;
	int i = 4 * (threadIdx.x + blockIdx.x * blockDim.x);
	int j = 4 * (threadIdx.y + blockIdx.y * blockDim.y);
	if ((i < 497) && (j < 497))//512-15=497
	{
		//for (j = 0; j<512 - 15; j += 4)
		// THE FOLLOWING SET OF RUNNING SUMS CAN BE A set of PARALLEL REDUCTIONs (in shared memory?)
		// 256 itteratios -> log2(256)=8 itterations
		// Store block into registers (256 x 4Bytes = 1kB)
		int idx = 0;
		for (iB = i; iB < i + 16; iB++)
		{
			for (jB = j; jB < j + 16; jB++)
			{
				//xVal_local[idx] = xVal[iB * 512 + jB];
				xVal_local[idx] = xVal[iB * 512 + jB];
				//printf("%d ", iB * 512 + jB);
				idx++;
			}
		}
		//Traverse through and get mean
		float mean = 0;
		for (idx = 0; idx < 256; idx++)
			mean += xVal_local[idx];				//this can be a simple reduction in shared memory
		mean = mean / 256.0f;
		//Traverse through and get stdev, skew and kurtosis
		stdev = 0;
		skw = 0;
		krt = 0;
		float xV_mean = 0;
		for (idx = 0; idx < 256; idx++)
		{
			// Place this commonly re-used value into a register to preserve temporal localitiy
			xV_mean = xVal_local[idx] - mean;
			stdev = stdev + (xV_mean * xV_mean);
			skw = skw + (xV_mean * xV_mean * xV_mean);
			krt = krt + (xV_mean * xV_mean * xV_mean * xV_mean);
		}
		stmp = sqrt(stdev / 256.0f);
		stdev = sqrt(stdev / 255.0f);//MATLAB's std is a bit different
		/*
		if (i + j <5)
		{
		printf("%f %f %f %f %f \n", stdev,stmp,stdev, skw, krt);
		}
		*/
		if (stmp != 0){
			skw = (skw / 256.0f) / ((stmp)*(stmp)*(stmp));
			krt = (krt / 256.0f) / ((stmp)*(stmp)*(stmp)*(stmp));
		}
		else{
			skw = 0;
			krt = 0;
		}
		/*
		if (i + j <5)
		{
		printf("%f %f \n", skw, krt);
		}*/
		//---------------------------------------------------------------------------
		// This is the nearest neighbor interpolation - ACTUALLY NOT NEEDED!!!!!!!!
		// To remove the nested for loop here we need to modifie the algorithm to 
		// adjust for the pointwise muliplication done far later that uses a
		// 512x512 dimension matrix derived from the matrices this kernel produces
		// The modified output would be PxP (as described mathematically in the paper).
		//---------------------------------------------------------------------------
		// Only this final output should be written to global memory:

		for (iB = i; iB < i + 4; iB++)
		{
			for (jB = j; jB < j + 4; jB++)
			{
				// Added if-else statement here:
				if (i > 500 || j > 500)
				{
					outStd[(iB * 512) + jB] = 0;
					outSkw[(iB * 512) + jB] = 0;
					outKrt[(iB * 512) + jB] = 0;
				}
				else
				{
					outStd[(iB * 512) + jB] = stdev;
					outSkw[(iB * 512) + jB] = skw;
					outKrt[(iB * 512) + jB] = krt;
				}
				// Added if-else statement here:
				//if (i > 500 || j > 500)
				//{
				//	outStd[threadId] = 0;
				//	outSkw[threadId] = 0;
				//	outKrt[threadId] = 0;
				//}
				//else
				//{
				//	outStd[threadId] = stdev;
				//	outSkw[threadId] = skw;
				//	outKrt[threadId] = krt;
				//}
			}
		}
	}
}
#endif

void check(float *a, float *b)
{
	std::vector<float> vc;
	for (int i = 0; i < N*N; i++)
	{
		if (a[i] != b[i])
			vc.push_back(i);
	}
}
#ifdef SHARED_TEST

__global__ void min_kernel(float* xVal, float* out)
{
	//Declarations
	//__shared__ float xVal_Shm[256];
	__shared__ float xVal_smem[BLK_SIZE + WIN_SIZE][BLK_SIZE + WIN_SIZE]; //threadDim.x, threadDim.y size
	float mean = 0, stdev = 0, skw = 0, krt = 0, stmp = 0;
	float iB, jB;
	//https://cs.calvin.edu/courses/cs/374/CUDA/CUDA-Thread-Indexing-Cheatsheet.pdf

	//Used in code
	int global_idx = (threadIdx.x + blockIdx.x * blockDim.x);
	int global_idy = (threadIdx.y + blockIdx.y * blockDim.y);
	if ((global_idx < N) && (global_idy < N))
	{
		xVal_smem[threadIdx.x][threadIdx.y] = *(xVal + global_idx * N + global_idy);
		if ((threadIdx.x >= (BLK_SIZE - WIN_SIZE)) && (threadIdx.y >= (BLK_SIZE - WIN_SIZE)))
		{
			xVal_smem[threadIdx.x + WIN_SIZE][threadIdx.y + WIN_SIZE] = *(xVal + (global_idx + WIN_SIZE)* N + (global_idy + WIN_SIZE));
		}
		if (threadIdx.y >= (BLK_SIZE - WIN_SIZE))
		{
			xVal_smem[threadIdx.x][threadIdx.y + WIN_SIZE] = *(xVal + global_idx * N + global_idy + WIN_SIZE);
		}
		if (threadIdx.x >= (BLK_SIZE - WIN_SIZE))
		{
			xVal_smem[threadIdx.x + WIN_SIZE][threadIdx.y] = *(xVal + (global_idx + WIN_SIZE) * N + global_idy);
		}
		__syncthreads();
		//printf("%d %d %d\n", threadIdx.x, threadIdx.y, temp);
	}
	if ((global_idx < N - WIN_SIZE) && (global_idy < N - WIN_SIZE))
	{
		float mean = 0; 
		for (int x = 0; x < WIN_SIZE; x++)
		{
			for (int y = 0; y < WIN_SIZE; y++)
			{
					mean += xVal_smem[threadIdx.x + x][threadIdx.y + y];
			}
		}
		mean = mean / 4.0f;
		out[global_idx * N + global_idy] = mean;
	}
	else
	{
		out[global_idx * N + global_idy] = xVal_smem[threadIdx.x][threadIdx.y];
	}
}
#endif
__global__ void A5_fast_lo_stats_kernel(float* xVal, float* out)
{
	//Declarations
	float xVal_local[256];
	float mean = 0;
	int iB, jB;
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	int i = 4 * (threadIdx.x + blockIdx.x * blockDim.x);
	int j = 4 * (threadIdx.y + blockIdx.y * blockDim.y);
	int global_idx = (threadIdx.x + blockIdx.x * blockDim.x);
	int global_idy = (threadIdx.y + blockIdx.y * blockDim.y);
	if ((i < 497) && (j < 497))//512-15=497
	{
		//for (j = 0; j<512 - 15; j += 4)
		// THE FOLLOWING SET OF RUNNING SUMS CAN BE A set of PARALLEL REDUCTIONs (in shared memory?)
		// 256 itteratios -> log2(256)=8 itterations
		// Store block into registers (256 x 4Bytes = 1kB)
		int idx = 0;
		for (iB = i; iB < i + 4; iB++)
		{
			for (jB = j; jB < j + 4; jB++)
			{
				xVal_local[idx] = xVal[iB * N + jB];
				//printf("%d ", iB * 512 + jB);
				idx++;
			}
		}
		//Traverse through and get mean
		float mean = 0;
		for (idx = 0; idx < WIN_SIZE * WIN_SIZE; idx++)
			mean += xVal_local[idx];				//this can be a simple reduction in shared memory
		mean = mean / 4.0f;

		out[global_idx * N + global_idy] = mean;
	}
	else
		out[global_idx * N + global_idy] = xVal[global_idx * N + global_idy];
}
void min_CPU(float init_array[N*N], float target[])
{
	/*memset(init_array, 0, sizeof(init_array));
	for (int i = 0; i < N + WIN_SIZE; i++)
	{
	for (int j = 0; j < N + WIN_SIZE; j++)
	{
	if (i < N && j < N)
	{
	init_array[i][j] = src[i * N + j];
	}
	else
	{
	if (i >= N && j < N)
	{
	init_array[i][j] = init_array[(i - WIN_SIZE)][j];
	}
	else if (j >= N && i < N)
	{
	init_array[i][j] = init_array[i][(j - WIN_SIZE)];
	}
	else if (i >= N && j >= N)
	{
	init_array[i][j] = init_array[(i - WIN_SIZE)][(j - WIN_SIZE)];
	}
	}
	}
	}
	*///First copy WIN_SIZE cols and WIN_SIZE rows of the array
	for (int i = 0; i < N - WIN_SIZE; i++)
	{
		for (int j = 0; j < N - WIN_SIZE; j++)
		{
			float mean = 0;
			for (int p = 0; p < WIN_SIZE; p++)
			{
				for (int q = 0; q < WIN_SIZE; q++)
				{
					mean += init_array[(i + p) *N + (j + q)];
				}
			}
			target[i * N + j] = mean / 4.0f;
		}
	}
}

void fill_input(float *a)
{
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			a[i * N + j] = i + j;
		}
	}
}
void write_to_file_DEBUG(float* w, int length)
{
	std::ofstream outFile;
	outFile.open("TEST.txt");
	for (int i = 0; i < length; i++)  // Itterate over rows
	{
		for (int j = 0; j < length; j++) // Itterate over cols
			outFile << w[i * length + j] << " ";
		if (i != length - 1)
			outFile << ";\n";
	}
	outFile.close();
}
void kernel_wrapper()
{
#ifdef SHARED_TEST
	FILE *fp = NULL;
	fp = fopen("Xval.txt", "r");
	if (fp == NULL)
		printf("ERROR\n");
	float a[N*N];
	fill_input(a);
	//for (int i = 0; i < N; i++)
	//{
	//	for (int j = 0; j < N; j++)
	//	{
	//		fscanf(fp, "%f", &a[i*N + j]);
	//	}
	//}

	dim3 gridSize(4, 4, 1);
	dim3 blockSize(4, 4, 1);
	static float h_orig_out[N * N];
	static float h_shared_out[N * N];
	static float h_orig_out_cpu[N*N];
	memset(h_orig_out_cpu, 0, sizeof(h_orig_out_cpu));
	//min_CPU(a, h_orig_out_cpu);
	//write_to_file_DEBUG(h_orig_out, sizeof(h_orig_out) / sizeof(float));
	float* d_a, *d_out;
	cudaMalloc(&d_a, sizeof(float) * N*N);
	cudaMalloc(&d_out, sizeof(float)*N*N);

	//cudaMemset(d_a, 0, sizeof(float)* N*N);
	//cudaMemset(d_out, 0, sizeof(float)* N*N);

	cudaMemcpy(d_a, a, sizeof(float) * N*N, cudaMemcpyHostToDevice);
	A5_fast_lo_stats_kernel << < gridSize, blockSize >> > (d_a, d_out);
	cudaMemcpy(h_orig_out, d_out, sizeof(float) * N*N, cudaMemcpyDeviceToHost);
	write_to_file_DEBUG(h_orig_out, 256);
	//cudaMemset(d_out, 0, sizeof(float)* N*N);
	//min_kernel<< < gridSize, blockSize >> > (d_a, d_out);
	cudaMemcpy(h_shared_out, d_out, sizeof(float) * N*N, cudaMemcpyDeviceToHost);

#if 0
	cudaMalloc(&d_a, sizeof(int) * (N + WIN_SIZE) * (N + WIN_SIZE));
	cudaMalloc(&d_b, sizeof(int) * (N + WIN_SIZE) * (N + WIN_SIZE));
	cudaMalloc(&d_out, sizeof(int)*N*N);

	cudaMemcpy(d_a, init_array, sizeof(int) * (N + WIN_SIZE) * (N + WIN_SIZE), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, init_array, sizeof(int) * (N + WIN_SIZE) * (N + WIN_SIZE), cudaMemcpyHostToDevice);
	min_kernel_no_shared<< < gridSize, blockSize >> > (d_a, d_b, d_out);
	cudaMemcpy(h_shared_out, d_out, sizeof(int) * N*N, cudaMemcpyDeviceToHost);
#endif
#endif
#if 0
	A5_fast_lo_stats_kernel << < gridSize, blockSize, 0 >> >(d_a,d_b d_out);
	cudaMemcpy(h_orig_out, d_out, N*N*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemset(d_out, 0, N * N);
	//write_to_file_DEBUG(h_orig_out, N);
	A5_fast_lo_stats_kernel_SHARED << < gridSize, blockSize, 0 >> >(d_a, d_out);
	cudaMemcpy(h_shared_out, d_out, N*N*sizeof(int), cudaMemcpyDeviceToHost);
#endif
	check(h_orig_out_cpu, h_shared_out);
	//check(h_orig_out_cpu, h_orig_out);
	cudaFree(d_a);
	cudaFree(d_out);
	cudaDeviceReset();
	getchar();
	
}
