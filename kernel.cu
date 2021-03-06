// kernel.cu
//This is the project for optimizing memory access stride.
//-------------------------------------------------------------------
#define DEFAULT

#include "header.h"
#include <time.h>
#include <stdlib.h>

#define N 512
#define BLK_SIZE 16 //blocksize(ThreadDim)
#define WIN_SIZE 16


void check(float *a, float *b)
{
	std::vector<float> vc;
	for (int i = 0; i < N*N; i++)
	{
		if (a[i] != b[i])
			vc.push_back(i);
	}
	if (vc.size() == 0)
		cout << "CLEAN";
}
__global__ void min_kernel(float* xVal, float* outStd, float* outSkw, float* outKrt)
{
	//Declarations
	__shared__ float xVal_smem[2 * (WIN_SIZE)][2 * (WIN_SIZE)]; //32*32 for shared memory 32 * 32 for mean calculation
	float  stdev = 0, skw = 0, krt = 0, stmp = 0;
	float iB, jB;
	float mean = 0;
	//https://cs.calvin.edu/courses/cs/374/CUDA/CUDA-Thread-Indexing-Cheatsheet.pdf

	//Used in code
	int global_idx = (threadIdx.x + blockIdx.x * blockDim.x);
	int global_idy = (threadIdx.y + blockIdx.y * blockDim.y);

	/* Shared memory intake
		1. Using column stride to avoid bank conflicts
		2. No need for condition check. Values taken in without issues.
		3. Every block will be holding 32x32 data element, processing from 2x2 places. start and end 
	*/
		xVal_smem[threadIdx.y][threadIdx.x] = xVal[global_idx* N + global_idy];
		xVal_smem[threadIdx.y + WIN_SIZE][threadIdx.x] = xVal[global_idx* N + global_idy + WIN_SIZE];
		xVal_smem[threadIdx.y][threadIdx.x + WIN_SIZE] = xVal[(global_idx + WIN_SIZE)* N + global_idy];
		xVal_smem[threadIdx.y + WIN_SIZE][threadIdx.x + WIN_SIZE] = xVal[(global_idx + WIN_SIZE)* N + global_idy + WIN_SIZE];

	__syncthreads();

	if (!(threadIdx.x & 3) && !(threadIdx.y & 3) && blockIdx.x < 31 && blockIdx.y < 31) //Obsucre: Check if they are divisible by 4
	{
		int y = threadIdx.y;
		int idx = 0;
		for (int x = threadIdx.x; x < WIN_SIZE + threadIdx.x; x++)
		{
			mean += (xVal_smem[y][x] + xVal_smem[y + 1][x] + xVal_smem[y + 2][x] + xVal_smem[y + 3][x]
				+ xVal_smem[y + 4][x] + xVal_smem[y + 5][x] + xVal_smem[y + 6][x] + xVal_smem[y + 7][x]
				+ xVal_smem[y + 8][x] + xVal_smem[y + 9][x] + xVal_smem[y + 10][x] + xVal_smem[y + 11][x]
				+ xVal_smem[y + 12][x] + xVal_smem[y + 13][x] + xVal_smem[y + 14][x] + xVal_smem[y + 15][x]);
		}

		mean = mean / 256.0f; //Cannot use >> because it is illagal on float

		int x = blockIdx.x;
		int y1 = blockIdx.y;
		//out[global_idx*N + global_idy] = xVal[global_idx*N + global_idy];
		//if (blockIdx.x != 31 && blockIdx.y != 31)
		//{
		//	out[((x * 4) + threadIdx.x / 4) * N + (y1 * 4) + threadIdx.y / 4] = mean; //Too much jugaad!!
		//}
		//Sectino 2
		//Traverse through and get stdev, skew and kurtosis
		float xV_mean = 0;
		for (int x = threadIdx.x; x < WIN_SIZE + threadIdx.x; x++)
		{
			for (int y = threadIdx.y; y < WIN_SIZE + threadIdx.y; y++)
			{
				xV_mean = xVal_smem[y][x] - mean;
				stdev = stdev + (xV_mean * xV_mean);
				skw = skw + (xV_mean * xV_mean * xV_mean);
				krt = krt + (xV_mean * xV_mean * xV_mean * xV_mean);
			}
		}
		
		stmp = sqrt(stdev / 256.0f);
		stdev = sqrt(stdev / 255.0f);//MATLAB's std is a bit different
		if (stmp != 0)
		{
			skw = (skw / 256.0f) / ((stmp)*(stmp)*(stmp));
			krt = (krt / 256.0f) / ((stmp)*(stmp)*(stmp)*(stmp));
		}
		else
		{
			skw = 0;
			krt = 0;
		}
		for (int i = 0; i < 4; i++)
		{
			if (blockIdx.y == 0 && threadIdx.y == 0 && threadIdx.x == 0 && blockIdx.x == 19)
			{
				printf("%d\n", (global_idx + i)* N + global_idy + 0);
			}
			outStd[(global_idx + i)* N + global_idy + 0] = stdev;
			outSkw[(global_idx + i)* N + global_idy + 0] = skw;
			outKrt[(global_idx + i)* N + global_idy + 0] = krt;

			outStd[(global_idx + i) * N + global_idy + 1] = stdev;
			outSkw[(global_idx + i)* N + global_idy + 1] = skw;
			outKrt[(global_idx + i)* N + global_idy + 1] = krt;

			outStd[(global_idx + i)* N + global_idy + 2] = stdev;
			outSkw[(global_idx + i)* N + global_idy + 2] = skw;
			outKrt[(global_idx + i)* N + global_idy + 2] = krt;

			outStd[(global_idx + i)* N + global_idy + 3] = stdev;
			outSkw[(global_idx + i)* N + global_idy + 3] = skw;
			outKrt[(global_idx + i)* N + global_idy + 3] = krt;
		}
	}
}
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
			}
		}
	}
}

void fill_input(float *a)
{
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			a[i * N + j] = i*N + j;
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

	FILE *fp = NULL;
	fp = fopen("Xval.txt", "r");
	if (fp == NULL)
		printf("ERROR\n");
	float a[N*N];
	fill_input(a);
	#if 0
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			fscanf(fp, "%f", &a[i*N + j]);
		}
	}
	#end if

	dim3 gridSize(32, 32, 1);
	dim3 blockSize(BLK_SIZE, BLK_SIZE, 1);
	static float h_orig_out[N * N];
	static float h_shared_out[N * N];

	static float h_shared_out_outStd[N * N];
	static float h_shared_out_outSkw[N * N];
	static float h_shared_out_outKrt[N * N];

	static float h_orig_out_outStd[N * N];
	static float h_orig_out_outSkw[N * N];
	static float h_orig_out_outKrt[N * N];

	
	//min_CPU(a, h_orig_out_cpu);
	//write_to_file_DEBUG(h_orig_out, sizeof(h_orig_out) / sizeof(float));
	float* d_a, *d_out;
	float* d_outStd, *d_outSkw,  *d_outKrt;
	cudaMalloc(&d_a, sizeof(float) * N*N);
	cudaMalloc(&d_out, sizeof(float)*N*N);
	cudaMalloc(&d_outStd, sizeof(float)*N*N);
	cudaMalloc(&d_outSkw, sizeof(float)*N*N);
	cudaMalloc(&d_outKrt, sizeof(float)*N*N);

	cudaMemset(d_a, 0, sizeof(float)* N*N);
	cudaMemset(d_out, 0, sizeof(float)* N*N);
	cudaMemset(d_outStd, 0, sizeof(float)* N*N);
	cudaMemset(d_outSkw, 0, sizeof(float)* N*N);
	cudaMemset(d_outKrt, 0, sizeof(float)* N*N);

	cudaMemcpy(d_a, a, sizeof(float) * N*N, cudaMemcpyHostToDevice);
	A5_fast_lo_stats_kernel << < gridSize, blockSize >> > (d_a,  d_outStd, d_outSkw, d_outKrt);
	cudaMemcpy(h_orig_out_outStd, d_outStd, sizeof(float) * N*N, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_orig_out_outSkw, d_outSkw, sizeof(float) * N*N, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_orig_out_outKrt, d_outKrt, sizeof(float) * N*N, cudaMemcpyDeviceToHost);
	write_to_file_DEBUG(h_orig_out_outStd, N);
	write_to_file_DEBUG(h_orig_out_outSkw, N);
	write_to_file_DEBUG(h_orig_out_outKrt, N);


	cout << "Break" << endl;
	cudaMemcpy(d_a, a, sizeof(float) * N*N, cudaMemcpyHostToDevice);
	min_kernel << < gridSize, blockSize >> > (d_a, d_outStd, d_outSkw, d_outKrt);

	cudaMemcpy(h_shared_out_outStd, d_outStd, sizeof(float) * N*N, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_shared_out_outSkw, d_outSkw, sizeof(float) * N*N, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_shared_out_outKrt, d_outKrt, sizeof(float) * N*N, cudaMemcpyDeviceToHost);
	write_to_file_DEBUG(h_shared_out_outStd, N);
	write_to_file_DEBUG(h_shared_out_outSkw, N);
	write_to_file_DEBUG(h_shared_out_outKrt, N);

	check(h_orig_out_outStd, h_shared_out_outStd);
	//check(h_shared_out_outStd, h_orig_out_outStd);
	//check(h_shared_out_outSkw, h_orig_out_outSkw);
	//check(h_shared_out_outKrt, h_orig_out_outKrt);
	//check(h_orig_out_cpu, h_orig_out);
	cudaFree(d_a);
	cudaFree(d_out);
	cudaDeviceReset();
	getchar();

}
