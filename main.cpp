#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <assert.h>
#include <chrono>

using milli = std::chrono::milliseconds;

float varN(float *in, int N, int D, int col) {
    double mean = 0;
    double mean2 = 0;
    for (int i = 0; i < N; i++) {
    	mean += in[i*D + col];
    	mean2 += in[i*D + col] * in[i*D + col];
    }
    return (mean2 - mean * mean / N);
}

void argTop4(double *in, int N, int *top4) {
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

void cumsses(float *in, int N, int D, float *out) {
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

void sort(float *in, int N, int D, int idx) {
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
    //std::cout << N << " " << lo + hi << std::endl;
    //std::cout << __LINE__ << std::endl;
    memcpy(in + D * lo, in, D * sizeof(float));
    //std::cout << __LINE__ << std::endl;
    memcpy(in, belowe, lo * D * sizeof(float));
    // std::cout << in + D * sizeof(float) * (lo + 1) << std::endl;
    //std::cout << __LINE__ << " " << lo << " " << hi << std::endl;
    memcpy(in + D * (lo + 1), above, hi * D * sizeof(float));
    //std::cout << __LINE__ << std::endl;
    free(above);
    free(belowe);
    //std::cout << e << " " << lo << " " << hi << std::endl;
    if (e != lo) {
        sort(in, lo, D, idx);
    }
    sort(in + lo + 1, hi, D, idx);
}

void optimal_split_vals(float **in2, int *lengths, int D, int B,
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
        sort(in, N, D, idx);
        // std::cout << __LINE__ << std::endl;

        cumsses(in, N, D, out);

        for (int i = 1; i < N; i++) {
            for (int j = 0; j < D; j++) {
                in_R[((int)N - i - 1) * (int)D + j] = in[i * D + j];
            }
        }
        cumsses(in_R, N, D, out_R);
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

void idxCandidates(float **buckets, int *lengths, int B, int D, int *top4) {
    double vars[D];
    for (int i = 0; i < D; i++) {
        vars[i] = 0;
    }
    for (int b = 0; b < B; b++) {
        for (int i = 0; i < D; i++) {
            vars[i] += varN(buckets[b], lengths[b], D, i);
        }
    }
    argTop4(vars, D, top4);
}

void split(float ***buckets, int B, int D, int **lengths, float *vals,
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


void mainCPU() {
    std::cout << std::fixed << std::setprecision(2);

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
        idxCandidates(buckets, lengths, B, D, top4);
		auto finish = std::chrono::high_resolution_clock::now();
		candidates_timer +=
			std::chrono::duration_cast<milli>(finish - start).count();

        // get best_indx and best_val for split
        float best_idx = top4[0];
        float best_error;
        float best_vals[B];
        float *best_errorp = &best_error;

		start = std::chrono::high_resolution_clock::now();
        optimal_split_vals(buckets, lengths, D, B, 0, &best_errorp, best_vals);
        for (int j = 1; j < 4; j++) {
            float vals[B];
            float error = 0;
            float *errorp = &error;
            optimal_split_vals(buckets, lengths, D, B, top4[j], &errorp, vals);
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
        split(&buckets, B, D, &lengths, best_vals, best_idx);
		finish = std::chrono::high_resolution_clock::now();
		split_timer +=
			std::chrono::duration_cast<milli>(finish - start).count();
        B *= 2;
    }


	std::cout << split_timer << std::endl;
	std::cout << optimal_timer << std::endl;
	std::cout << candidates_timer << std::endl;
	std::cout << optimal_timer + split_timer + candidates_timer << std::endl;
}

float *copyToDevice(float *in, int N, int D);
void mainGPU();

int main() {
    std::cout << "GPU" << std::endl;
	mainGPU();
    std::cout << "CPU" << std::endl;
	mainCPU();
}
