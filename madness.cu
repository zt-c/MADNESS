#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <assert.h>
#include <chrono>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

using milli = std::chrono::milliseconds;

float *copyToDevice(float *in, int N, int D) {
    float *out;
    cudaMalloc(&out, sizeof(float) * N * D);
    cudaMemcpy(out, in, sizeof(float) * N * D, cudaMemcpyHostToDevice);
    return out;
}

__global__ void square_kernel(float *in, int N, int D, int col) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        in[D * index + col] *= in[D * index + col];
    }
}

void square(float *in, int N, int D, int col) {
    square_kernel<<<(N + 255) / 256, 256>>>(in, N, D, col);
}

__global__ void sum_kernel(float *in, int N, int D, int col, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int step = 1 << n;
    int el = index * step;
    int eln = el + (step >> 1);
    if (el < N && eln < N) {
        //printf("%d %d-\n", el, eln);
        in[D * el + col] += in[D * eln + col];
    }
}

//3 2 0
float sum(float *in, int N, int D, int col) {
    float a[6];
    cudaMemcpy(a, in, sizeof(float) * 6, cudaMemcpyDeviceToHost);
    for (int i = 1; i < log2((float)N)+2; i++) {
        sum_kernel<<<(N + 255) / 256, 256>>>(in, N, D, col, i);
        cudaMemcpy(a, in, sizeof(float) * 6, cudaMemcpyDeviceToHost);
    }
    float out;
    cudaMemcpy(&out, in + col, sizeof(float), cudaMemcpyDeviceToHost);
    return out;
}



float varNG(float *in, int N, int D, int col) {
    double mean = 0;
    double mean2 = 0;
    float *copy;
    cudaMalloc(&copy, sizeof(float) * N * D);
    cudaMemcpy(copy, in, sizeof(float) * N * D, cudaMemcpyHostToDevice);
    mean = sum(copy, N, D, col);

    cudaMemcpy(copy, in, sizeof(float) * N * D, cudaMemcpyHostToDevice);
    square(copy, N, D, col);
    mean2 = sum(copy, N, D, col);

    //std::cout << mean2 << "-" << mean << std::endl;
    return (mean2 - mean * mean / N);
}

void argTop4G(double *in, int N, int *top4) {
    for (int i = 0; i < 4; i++) {
        top4[i] = -1;
    }
    for (int i = 0; i < N; i++) {
	int temp = i;
        for (int j = 3; j >= 0; j--) {
	    if (top4[j] == -1 || in[top4[j]] < in[temp]) {
	    	int temp2 = top4[j];
                top4[j] = temp;
                temp = temp2;
	    }
	}
    }
}

void cumssesG(float *in, int N, int D, float *out) {
    float cumsum[D];
    float cumsum2[D];
    for (int i = 0; i < D; i++) {
        cumsum[i] = in[i];
        cumsum2[i] = in[i] * in[i];
        out[i] = 0;
    }
    for (int i = 1; i < N; i++) {
        for (int j = 0; j < D; j++) {
            cumsum[j] += in[i * D + j];
            cumsum2[j] += in[i * D + j] * in[i * D + j];
            float mean = cumsum[j] / (i + 1);
            out[i * D + j] = cumsum2[j] - (cumsum[j] * mean);
        }
    }
}

__global__ void sort_kernel(float *in, int N, int D, int idx, bool even) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int el = index * 2 + (!even ? 1 : 0);
    if (el + 1 < N) {
        return;
    }
    if (in[D * el + idx] < in[D * (el + 1) + idx]) {
        float temp[27];
        memcpy(temp, in + D * el, sizeof(float) * D);
        memcpy(in + D * el, in + D * (el + 1), sizeof(float) * D);
        memcpy(in + D * (el + 1), temp, sizeof(float) * D);
    }
}


void sortGh(float *in, int N, int D, int idx) {
    if (N < 2) {
        return;
    }

    float *above = (float *) malloc(N * D * sizeof(float));
    float *belowe = (float *) malloc(N * D * sizeof(float));
    float temp[D];
    memcpy(temp, in + N / 2 * D, sizeof(float) * D);
    memcpy(in + N / 2 * D, in, sizeof(float) * D);
    memcpy(in, temp, sizeof(float) * D);
    float pivot = in[idx];
    int hi = 0;
    int lo = 0;
    int e = 0;
    for (int i = 1; i < N; i++) {
        if (in[D * i + idx] > pivot) {
            memcpy(above + D * hi, in + D * i, D * sizeof(float));
            hi++;
        } else if (in[D * i + idx] == pivot) {
            e++;
            memcpy(belowe + D * lo, in + D * i, D * sizeof(float));
            lo++;
	} else {
            memcpy(belowe + D * lo, in + D * i, D * sizeof(float));
            lo++;
        }
    }
    memcpy(in + D * lo, in, D * sizeof(float));
    memcpy(in, belowe, lo * D * sizeof(float));
    memcpy(in + D * (lo + 1), above, hi * D * sizeof(float));
    free(above);
    free(belowe);
    if (e != lo) {
        sortGh(in, lo, D, idx);
    }
    sortGh(in + lo + 1, hi, D, idx);
}

void sortG(float *in, int N, int D, int idx) {
    // uses bubble sort to sort on the GPU
    float* in2;
    cudaMalloc(&in2, N * D * sizeof(float));
    cudaMemcpy(in2, in, N * D * sizeof(float), cudaMemcpyHostToDevice);
    for (int i = 0; i < N + 2; i++) {
        sort_kernel<<<(N + 255)/256, 256>>>(in2, N, D, idx, i % 2 == 0);
        if (i % 1000 == 0) {
            std::cout << i << std::endl;
        }
    }
    cudaMemcpy(in2, in, N * D * sizeof(float), cudaMemcpyDeviceToHost);
}

void sort2(float *in, int N, int D, int idx) {
    if (N < 2) {
        return;
    }

    float *above = (float *) malloc(N * D * sizeof(float));
    float *belowe = (float *) malloc(N * D * sizeof(float));
    float temp[D];
    memcpy(temp, in + N / 2 * D, sizeof(float) * D);
    memcpy(in + N / 2 * D, in, sizeof(float) * D);
    memcpy(in, temp, sizeof(float) * D);
    float pivot = in[idx];
    int hi = 0;
    int lo = 0;
    int e = 0;
    for (int i = 1; i < N; i++) {
        if (in[D * i + idx] > pivot) {
            memcpy(above + D * hi, in + D * i, D * sizeof(float));
            hi++;
        } else if (in[D * i + idx] == pivot) {
            e++;
            memcpy(belowe + D * lo, in + D * i, D * sizeof(float));
            lo++;
	} else {
            memcpy(belowe + D * lo, in + D * i, D * sizeof(float));
            lo++;
        }
    }
    memcpy(in + D * lo, in, D * sizeof(float));
    memcpy(in, belowe, lo * D * sizeof(float));
    memcpy(in + D * (lo + 1), above, hi * D * sizeof(float));
    free(above);
    free(belowe);
    if (e != lo) {
        sort2(in, lo, D, idx);
    }
    sort2(in + lo + 1, hi, D, idx);
}

void optimal_split_valsG(float **in2, int *lengths, int D, int B,
                       int idx, float **err, float *val) {

    // std::cout << __LINE__ << std::endl;
    for (int b = 0; b < B; b++) {
        //std::cout << __LINE__ << std::endl;
        // std::cout << lengths[0] << std::endl;
        int N = lengths[b];
        float *in = (float *)malloc(N * D * sizeof(float));
        float *out = (float *)malloc(N * D * sizeof(float));
        float *in_R = (float *)malloc(N * D * sizeof(float));
        float *out_R = (float *)malloc(N * D * sizeof(float));
        // std::cout << __LINE__ << std::endl;
        memcpy(in, in2[b], N * D * sizeof(float));
        // std::cout << __LINE__ << std::endl;
        sort2(in, N, D, idx);
        // std::cout << __LINE__ << std::endl;

        cumssesG(in, N, D, out);

        for (int i = 1; i < N; i++) {
            for (int j = 0; j < D; j++) {
                in_R[((int)N - i - 1) * (int)D + j] = in[i * D + j];
            }
        }
        cumssesG(in_R, N, D, out_R);
        for (int i = 0; i < N; i++) {
            if (N - i - 2 > 0) {
                for (int j = 0; j < D; j++) {
                    out[i * D + j] += out_R[(N - i - 2) * D + j];
                }
            }
            for (int j = 1; j < D; j++) {
                out[i * D] += out[i * D + j];
            }
        }

        int min = 0;
        for (int i = 1; i < N; i++) {
            if (out[i * D] < out[min * D]) {
                min = i;
            }
        }

        *(val + b)  = in[min * D + idx];
        free(in);
        free(out);
        free(in_R);
        free(out_R);
    }
}

float varN2(float *in, int N, int D, int col) {
    double mean = 0;
    double mean2 = 0;
    for (int i = 0; i < N; i++) {
    	mean += in[i*D + col];
    	mean2 += in[i*D + col] * in[i*D + col];
    }
    return (mean2 - mean * mean / N);
}

void idxCandidatesG(float **buckets, int *lengths, int B, int D, int *top4) {
    double vars[D];
    for (int i = 0; i < D; i++) {
        vars[i] = 0;
    }
    for (int b = 0; b < B; b++) {
        for (int i = 0; i < D; i++) {
            vars[i] += varN2(buckets[b], lengths[b], D, i);
            // varNG(buckets[b], lengths[b], D, i);
            //std::cout << varN2(buckets[b], lengths[b], D, i) << " " <<
            //             varNG(buckets[b], lengths[b], D, i) << std::endl;
        }
    }
    argTop4G(vars, D, top4);
}

__global__ void is_lower_kernel(int *mask, float *bucket, int D, int length,
                                int idx, float val) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index > length) {
        return;
    }
    if (bucket[index * D + idx] < val) {
        mask[index] = 1;
    } else {
        mask[index] = 0;
    }
}

void splitG(float ***buckets, int B, int D, int **lengths, float *vals,
            int idx) {
    float **oldBuckets = *buckets;
    int *oldLengths = *lengths;
    *buckets = (float **) calloc(2 * B, sizeof(float *));
    *lengths = (int *) calloc(2 * B, sizeof(int));

    std::cout << __LINE__ << std::endl;
    for (int b = 0; b < B; b++) {
        int l = oldLengths[b];
        float *bucket;
        cudaMalloc(&bucket, l * D * sizeof(float));
        cudaMemcpy(bucket, oldBuckets[b], l * D * sizeof(float),
                   cudaMemcpyHostToDevice);
        thrust::device_ptr<int> d_mask = thrust::device_malloc<int>(l);
        is_lower_kernel<<<(oldLengths[b] + 255) / 256, 256>>>(d_mask.get(),
                bucket, D, l, idx, vals[b]);
        thrust::inclusive_scan(d_mask, d_mask + l, d_mask);
        cudaMemcpy(*lengths + 2*b, d_mask.get() + l - 1, sizeof(int),
                   cudaMemcpyDeviceToHost);
        (*lengths)[2 * b + 1] = l - (*lengths)[2 * b];
        (*buckets)[2*b] = (float *)malloc(D * (*lengths)[2*b] * sizeof(float));
        (*buckets)[2*b+1] = (float *)malloc(
                                        D * (*lengths)[2*b+1] * sizeof(float));
        int *mask = (int *)malloc(l * sizeof(int));
        cudaMemcpy(mask, d_mask.get(), l * sizeof(int),
                   cudaMemcpyDeviceToHost);
        for (int i = 0; i < l; i++) {
            int m;
            cudaMemcpy(&m, d_mask.get() + i, sizeof(int),
                       cudaMemcpyDeviceToHost);
            if (l-1 == i) {
                // std::cout << m << "-" << std::endl;
            }
        }
        //std::cout << (*lengths)[2*b] << std::endl;
        // std::cout << (*lengths)[2*b+1] << std::endl;

        int hi = 0;
        int lo = 0;
        for (int i = 0; i < l; i++) {
            if (mask[i]) {
                if (lo >= (*lengths)[2*b]) {
                    //std::cout << lo << " " << (*lengths)[2*b] << std::endl;
                    continue;
                }
                assert(lo < (*lengths)[2*b]);

                memcpy((*buckets)[2*b] + lo * D, oldBuckets[b] + lo * D,
                       D * sizeof(float));
                lo++;
            } else {
                assert(hi < (*lengths)[2*b+1]);
                memcpy((*buckets)[2*b + 1] + hi * D, oldBuckets[b] + lo * D,
                       D * sizeof(float));
                hi++;
            }
        }
        std::cout << __LINE__ << std::endl;
        // cudaMemcpy(o, a, sizeof(float) * 6, cudaMemcpyHostToDevice);
    }
    std::cout << __LINE__ << std::endl;
}

void split2(float ***buckets, int B, int D, int **lengths, float *vals,
           int idx) {
    float **oldBuckets = *buckets;
    int *oldLengths = *lengths;
    *buckets = (float **) calloc(2 * B, sizeof(float *));
    *lengths = (int *) calloc(2 * B, sizeof(int));

    for (int b = 0; b < B; b++) {
        for (int i = 0; i < oldLengths[b]; i++) {
            if (oldBuckets[b][i * D + idx] > vals[b]) {
                (*lengths)[2 * b + 1]++;
            } else {
                (*lengths)[2 * b]++;
            }
        }

        assert((*lengths)[2*b] + (*lengths)[2*b + 1] == oldLengths[b]);
        (*buckets)[2*b] = (float *) malloc(D*(*lengths)[2*b] *sizeof(float));
        (*buckets)[2*b+1] = (float *) malloc(D*(*lengths)[2*b+1] *sizeof(float));
        assert((*buckets)[2*b+1] != NULL);
        assert((*buckets)[2*b] != NULL);
        int lo = 0;
        int hi = 0;
        for (int i = 0; i < oldLengths[b]; i++) {
            if (oldBuckets[b][i * D + idx] > vals[b]) {
                assert(hi < (*lengths)[2*b+1]);
                std::memcpy((*buckets)[2*b+1] + D*hi, oldBuckets[b] + i * D,
                            sizeof(float) * D);
                hi++;
            } else {
                assert(lo < (*lengths)[2*b]);
                std::memcpy((*buckets)[2*b] + D*hi, oldBuckets[b] + i * D,
                            sizeof(float) * D);
                lo++;
            }
        }
    }
}


void mainGPU() {
    std::cout << std::fixed << std::setprecision(2);

    float a[] = {1, 20, 3, 4, 5, 6, -4, -9};
    float *o;
    cudaMalloc(&o, sizeof(float) * 6);
    cudaMemcpy(o, a, sizeof(float) * 6, cudaMemcpyHostToDevice);

    // load data (616050, 27)
    int N = 616050;
    int D = 27;
    //cnpy::NpyArray Xr = cnpy::npy_load("npy/X_train.npy");
    //float *X = Xr.data<float>();
    float *X = (float *)malloc(N * D * sizeof(float));
    std::srand(0);
    for (int i = 0; i < N * D; i++) {
        X[i] = (std::rand() % 1000) / 1000. * 5;
    }

    float **buckets = &X;
    int *lengths = &N;
    int B = 1;

	long int candidates_timer = 0;
	long int split_timer = 0;
	long int optimal_timer = 0;

    for (int i = 0; i < 4; i++) {
        // get top 4 indices
        int top4[4];
		auto start = std::chrono::high_resolution_clock::now();
        idxCandidatesG(buckets, lengths, B, D, top4);
		auto finish = std::chrono::high_resolution_clock::now();
		candidates_timer +=
			std::chrono::duration_cast<milli>(finish - start).count();

        // get best_indx and best_val for split
        float best_idx = top4[0];
        float best_error;
        float best_vals[B];
        float *best_errorp = &best_error;

        start = std::chrono::high_resolution_clock::now();
        optimal_split_valsG(buckets, lengths, D, B, 0, &best_errorp, best_vals);
        for (int j = 1; j < 4; j++) {
            float vals[B];
            float error = 0;
            float *errorp = &error;
            optimal_split_valsG(buckets, lengths, D, B, top4[j], &errorp, vals);
            if (error < best_error) {
                best_error = error;
                best_idx = top4[i];
                for (int b = 0; b < 4; b++) {
                    best_vals[b] = vals[b];
                }
            }
        }
		finish = std::chrono::high_resolution_clock::now();
		optimal_timer +=
			std::chrono::duration_cast<milli>(finish - start).count();

		start = std::chrono::high_resolution_clock::now();
        splitG(&buckets, B, D, &lengths, best_vals, best_idx);
		finish = std::chrono::high_resolution_clock::now();
		split_timer =
            std::chrono::duration_cast<milli>(finish - start).count();
        std::cout << "-- " << split_timer << std::endl;
        B *= 2;
    }


    std::cout << split_timer << std::endl;
    std::cout << optimal_timer << std::endl;
    std::cout << candidates_timer << std::endl;
}
