#include "CycleTimer.h"
#include <fstream>
#include <iostream>
#include <memory.h>
#include <string>
using namespace std;

// A.shape=[M,D], B.shape=[D,N]
// result: C.shape=[M,N]
#define M 1777
#define D 64
#define N 100

#define N_BUCKETS 16
#define N_SPLIT_INDICES 4
#define N_MAX_BUCKET_IN_ONE_LEVEL (1 << (N_SPLIT_INDICES - 1))

// Split List and Centroids
struct SplitParams {
    int dims[N_SPLIT_INDICES];
    double thresholds[N_SPLIT_INDICES][N_MAX_BUCKET_IN_ONE_LEVEL];

    double offsets[N_SPLIT_INDICES];
    double scalebys[N_SPLIT_INDICES];

    double centroids[N_BUCKETS][D];
};

__constant__ SplitParams cuConstParams;

// Matrix A and B
struct InputData {
    double X[M][D];
    double Q[N][D];
};

struct Result {
    int X_enc[M];
    double Q_luts[N][N_BUCKETS];
    double Y[M][N];
};

__global__ void kernelEncodeX(InputData *inputs, Result *result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M)
        return;
    int group_id = 0;
    for (int i = 0; i != N_SPLIT_INDICES; ++i) {
        int dim = cuConstParams.dims[i];
        double scaleby = cuConstParams.scalebys[i];
        double offset = cuConstParams.offsets[i];

        double threshold = cuConstParams.thresholds[i][group_id];

        bool indicator = (inputs->X[idx][dim] - offset) * scaleby > threshold;
        group_id = (group_id << 1) + indicator;
    }
    result->X_enc[idx] = group_id;
}

void encodeX(SplitParams *params, InputData *inputs, Result *result) {
    for (int j = 0; j != M; ++j) {
        int group_id = 0;
        for (int i = 0; i != N_SPLIT_INDICES; ++i) {
            int dim = params->dims[i];
            double scaleby = params->scalebys[i];
            double offset = params->offsets[i];

            double threshold = params->thresholds[i][group_id];
            bool indicator = (inputs->X[j][dim] - offset) * scaleby > threshold;
            group_id = (group_id << 1) + indicator;
        }
        result->X_enc[j] = group_id;
    }
}

__global__ void kernelEncodeQ(InputData *inputs, Result *result) {
    int i = threadIdx.x;
    int j = blockIdx.x;
    double sum = 0;
    for (int k = 0; k != D; ++k) {
        sum += cuConstParams.centroids[j][k] * inputs->Q[i][k];
    }
    result->Q_luts[i][j] = sum;
}

void encodeQ(SplitParams *params, InputData *inputs, Result *result) {
    for (int j = 0; j != N_BUCKETS; ++j) {
        for (int i = 0; i != N; ++i) {
            double sum = 0;
            for (int k = 0; k != D; ++k) {
                sum += params->centroids[j][k] * inputs->Q[i][k];
            }
            result->Q_luts[i][j] = sum;
        }
    }
}

__global__ void kernelDists_enc(Result *result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > M)
        return;
    for (int j = 0; j != N; ++j) {
        result->Y[idx][j] = result->Q_luts[j][result->X_enc[idx]];
    }
}

void dists_enc(Result *result) {
    for (int i = 0; i != M; ++i) {
        for (int j = 0; j != N; ++j) {
            result->Y[i][j] = result->Q_luts[j][result->X_enc[i]];
        }
    }
}

void loadInputTo(SplitParams &params, InputData &inputs) {
    std::ifstream infile;
    infile.open("input/dims.txt");
    for (int i = 0; i != N_SPLIT_INDICES; ++i)
        infile >> params.dims[i];
    infile.close();

    infile.open("input/offsets.txt");
    for (int i = 0; i != N_SPLIT_INDICES; ++i)
        infile >> params.offsets[i];
    infile.close();

    infile.open("input/scalebys.txt");
    for (int i = 0; i != N_SPLIT_INDICES; ++i)
        infile >> params.scalebys[i];
    infile.close();

    infile.open("input/thresholds.txt");
    for (int i = 0; i != N_SPLIT_INDICES; ++i)
        for (int j = 0; j != (1 << i); ++j)
            infile >> params.thresholds[i][j];
    infile.close();

    infile.open("input/centroids.txt");
    for (int i = 0; i != N_BUCKETS; ++i)
        for (int j = 0; j != D; ++j)
            infile >> params.centroids[i][j];
    infile.close();

    infile.open("input/A.txt");
    for (int i = 0; i != M; ++i)
        for (int j = 0; j != D; ++j)
            infile >> inputs.X[i][j];
    infile.close();

    infile.open("input/B.txt");
    for (int i = 0; i != D; ++i)
        for (int j = 0; j != N; ++j)
            infile >> inputs.Q[j][i];
    infile.close();
}

int main() {
    SplitParams params;
    InputData inputs, *d_inputs;
    Result results, *d_results;

    loadInputTo(params, inputs);

    int n_runs = 1000;

    // CPU time
    double startTime = CycleTimer::currentSeconds();
    for (int i = 0; i != n_runs; ++i) {
        encodeX(&params, &inputs, &results);
        encodeQ(&params, &inputs, &results);
        dists_enc(&results);
    }
    double endTime = CycleTimer::currentSeconds();
    double overallDurationCPU = endTime - startTime;

    // GPU time
    cudaMalloc(&d_inputs, sizeof(InputData));
    cudaMemcpy(d_inputs, &inputs, sizeof(InputData), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(cuConstParams, &params, sizeof(params));
    cudaMalloc(&d_results, sizeof(Result));

    startTime = CycleTimer::currentSeconds();
    for (int i = 0; i != n_runs; ++i) {
        int threadPerBlock = 128;
        kernelEncodeX<<<(M + threadPerBlock - 1) / threadPerBlock, threadPerBlock>>>(d_inputs, d_results);
        kernelEncodeQ<<<N_BUCKETS, N>>>(d_inputs, d_results);
        cudaDeviceSynchronize();
        kernelDists_enc<<<(M + threadPerBlock - 1) / threadPerBlock, threadPerBlock>>>(d_results);
    }
    cudaDeviceSynchronize();
    endTime = CycleTimer::currentSeconds();

    double cu_Y[M][N];
    cudaMemcpy(&cu_Y, &(d_results->Y), sizeof(cu_Y), cudaMemcpyDeviceToHost);
    double overallDurationGPU = endTime - startTime;

    cout << overallDurationCPU * 1000 / n_runs << "ms " << overallDurationGPU * 1000 / n_runs << "ms " << overallDurationCPU / overallDurationGPU << endl;

    // for (int i = 0; i != 10; ++i)
    // cout << "cu" << cu_Y[N][i] << endl;
    for (int i = 0; i != 30; ++i)
        if (cu_Y[N][i] != results.Y[N][i])
            cout << "mismatch detected" << endl;
    return 0;
}