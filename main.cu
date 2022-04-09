#include <cstdio>
#include <malloc.h>
#include <ctime>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "host_defines.h"


__global__ void sumRows(float *matrix, int i, int length) {
    int t_id = threadIdx.x;
    if (t_id > i && t_id <= length - 1) {
        float factor = -(matrix[length * t_id + i] / matrix[i * length + i]);  //calculating factor
        for (int j = i; j < length; j++)
            matrix[length * t_id + j] += matrix[i * length + j] * factor;
    }
}

__global__ void getDeterminant(float *matrix, int length) {
    __syncthreads();
    for (int i = 0; i < length - 1; i++) {
        sumRows<<<1, length>>>(matrix, i, length);
        cudaDeviceSynchronize();
    }
}

void speedTest() {
    FILE *result_fptr = fopen("gpu_time.txt", "w+");
    FILE *matrixes_fptr = fopen("read.txt", "r");
    if (result_fptr) {
        while (!feof(matrixes_fptr)) {
            int size = 0;
            fscanf(matrixes_fptr, "%d\n", &size);
            float *a = (float *) malloc(size * size * sizeof(float));
            for (int k = 0; k < size; k++)
                for (int l = 0; l < size; l++)
                    fscanf(matrixes_fptr, "%f ", a + (k * size + l));
            clock_t start = clock();
            float *a_d;
            cudaMalloc(&a_d, size * size * sizeof(float));
            cudaMemcpy(a_d, a, size * size * sizeof(float), cudaMemcpyHostToDevice);
            getDeterminant<<<1, 1>>>( a_d, size);
            cudaThreadSynchronize();
            cudaMemcpy(a, a_d, size * size * sizeof(float), cudaMemcpyDeviceToHost);
            cudaFree(a_d);
            float res = a[0];
            for (int i = 1; i < size; i++) {
                res *= a[i * size + i];
            }
            clock_t end = clock();
            fprintf(result_fptr, "%d - %lf\n", size, (double) (end - start) / CLOCKS_PER_SEC);
            printf("%d\n", size);
            free(a);
        }
    }
    fclose(matrixes_fptr);
    fclose(result_fptr);
}

int main() {
    speedTest();
    return 0;
}
