#include <stdio.h>
#include <malloc.h>
#include <time.h>
#include <stdlib.h>
#include <float.h>

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

void swapRows(float *arr1, float *arr2, int length) {
    float tmp;
    for (int j = 0; j < length; j++) {
        tmp = arr1[j];
        arr1[j] = arr2[j];
        arr2[j] = tmp;
    }
}

float getDeterminant(float *matrix, int length) {
    for (int i = 0; i < length - 1; i++) {
        for (int k = i + 1; k < length; k++) {
            if (matrix[i * length + i] == 0) {  //swapping in case of nulls on diagonal
                int l = i + 1;
                for (; matrix[l * length + i] == 0 && l < length; l++) {}
                swapRows(matrix + i * length, matrix + l * length, length);
            }
            float factor = -(matrix[k * length + i] / matrix[i * length + i]);  //calculating factor
            for (int j = i; j < length; j++) {
                matrix[k * length + j] += matrix[i * length + j] * factor;
            }
        }
    }
    float det = matrix[0];
    for (int i = 0; i < length; i++)
        det *= matrix[i * length + i];
    return det;
}

void speedTest() {
    FILE *result_fptr = fopen("time.txt", "a+");
    FILE *matrixes_fptr = fopen("read.txt", "r");
    if (result_fptr) {
        while (!feof(matrixes_fptr)) {
            int i = 0;
            fscanf(matrixes_fptr, "%d\n", &i);
            float *a = (float*)malloc(i * i * sizeof(float));
            for (int k = 0; k < i; k++)
                for (int l = 0; l < i; l++)
                    fscanf(matrixes_fptr, "%f ", a + (k * i + l));
            clock_t start = clock();
            getDeterminant(a, i);
            clock_t end = clock();
            fprintf(result_fptr, "%d - %lf\n", i, (double)(end - start) / CLOCKS_PER_SEC);
            free(a);
        }
    }
    fclose(matrixes_fptr);
    fclose(result_fptr);
}

void fillRead(int max_size) {
    FILE* fptr = fopen("read.txt", "w+");
    if (fptr) {
        for (int i = 1; i <= max_size; i++) {
            fprintf(fptr, "\n%d\n", i);
            for (int j = 0; j < (i * i); j++)
                fprintf(fptr, "%d ", rand() % 1000);
        }
    }
    fclose(fptr);
}

int main() {
    srand(time(NULL));
    fillRead(100);
    speedTest();
    return 0;
}
