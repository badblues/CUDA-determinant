#include <cstdio>
#include <malloc.h>
#include <ctime>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "host_defines.h"


void printMatrix(float *matrix, int length) {
    printf("\n");
    for (int i = 0; i < length; i++) {
        for (int j = 0; j < length; j++) {
            printf("%.3f ", matrix[i * length + j]);
        }
        printf("\n");
    }
    printf("\n");
}

//adding rows using different threads
__global__ void sumRows(float *matrix, int i, int length) {
    int t_id = threadIdx.x;  //getting thread id
    if (t_id > i && t_id <= length - 1) {
        float factor = -(matrix[length * t_id + i] / matrix[i * length + i]);  //calculating factor
        for (int j = i; j < length; j++)  //each thread calculating its own row
            matrix[length * t_id + j] += matrix[i * length + j] * factor;
    }
}

//this function converts matrix into triangular form
__global__ void getDeterminant(float *matrix, int length) {
    __syncthreads();  //putting together all the threads
    for (int i = 0; i < length - 1; i++) {
        sumRows<<<1, length>>>(matrix, i, length);  //calling block of [1 x length] size
        cudaDeviceSynchronize();  //waiting all threads to calculate
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
            cudaMalloc(&a_d, size * size * sizeof(float));  //allocating memory on gpu device
            cudaMemcpy(a_d, a, size * size * sizeof(float), cudaMemcpyHostToDevice);  //copying matrix
            getDeterminant<<<1, 1>>>( a_d, size);  //getting triangular matrix
            cudaThreadSynchronize();  //waiting for all the calculations to end
            cudaMemcpy(a, a_d, size * size * sizeof(float), cudaMemcpyDeviceToHost);  //copying matrix back
            cudaFree(a_d);  //freeing memory on gpu
            float res = a[0];  //actually calculating result
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
    //speedTest();
    int size = 3;
    float *a = (float*)malloc(size * size * sizeof(float));
    a[0] = 1; a[1] = 2; a[2] = 3;
    a[3] = 4; a[4] = 0; a[5] = 7;
    a[6] = 7; a[7] = 8; a[8] = 9;
    float *a_d;
    cudaMalloc(&a_d, size * size * sizeof(float));  //allocating memory on gpu device
    cudaMemcpy(a_d, a, size * size * sizeof(float), cudaMemcpyHostToDevice);  //copying matrix
    getDeterminant<<<1, 1>>>( a_d, size);  //getting triangular matrix
    cudaThreadSynchronize();  //waiting for all the calculations to end
    cudaMemcpy(a, a_d, size * size * sizeof(float), cudaMemcpyDeviceToHost);  //copying matrix back
    cudaFree(a_d);  //freeing memory on gpu
    printMatrix(a, size);
    float res = a[0];  //actually calculating result
    for (int i = 1; i < size; i++) {
        res *= a[i * size + i];
    }
    printf("result = %f", res);
    return 0;
}
