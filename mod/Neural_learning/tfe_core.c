#include "type.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define TFE_VAD_DIM 3

ErrorCode tfe_network_init(NeuralNetwork *net, int in_dim) {
    if (net == NULL) return TFE_ERR_NULL_POINTER;
    if (in_dim <= 0) return TFE_ERR_INVALID_ARG;

    const int out_dim = TFE_VAD_DIM;
    double *W = (double*)calloc((size_t)in_dim * (size_t)out_dim, sizeof(double));

    if (W == NULL) return TFE_ERR_OUT_OF_MEMORY;

    for (int i = 0; i < in_dim * out_dim; i++){
        W[i] = ((double)rand() / RAND_MAX - 0.5) * 0.02;
    }

    net->W = W;
    net->in_dim = in_dim;
    net->out_dim = out_dim;

    memset(net->B, 0, sizeof(net->B));
    memset(net->Y, 0, sizeof(net->Y));
    return TFE_SUCCESS;
}

void tfe_network_free(NeuralNetwork *net) {
    if (net == NULL) return;
    free(net->W);
    net->W = NULL;
    net->in_dim  = 0;
    net->out_dim = 0;
}

ErrorCode tfe_extractor_init(TextFeatureExtractor *T, const char **vocab, int vocab_size) {
    if (T == NULL || vocab == NULL) return TFE_ERR_NULL_POINTER;
    if (vocab_size <= 0) return TFE_ERR_INVALID_ARG;

    const char **v = (const char**)malloc(sizeof(const char *) * vocab_size);
    if (v == NULL) return TFE_ERR_OUT_OF_MEMORY;

    for (int i = 0; i < vocab_size; i++) {
        if (vocab[i] == NULL) {
            free((void *)v);
            return TFE_ERR_INVALID_TEXT;
        }
        v[i] = vocab[i];
    }

    double *x = (double*)calloc(vocab_size, sizeof(double));
    if (x == NULL) {
        free((void*)v);
        return TFE_ERR_OUT_OF_MEMORY;
    }

    T->vocab = v;
    T->vocab_size = vocab_size;
    T->X = x;
    return TFE_SUCCESS;
}

void tfe_extractor_free(TextFeatureExtractor *T) {
    if (T == NULL) return;
    free((void *)T->vocab);
    free(T->X);

    T->vocab = NULL;
    T->vocab_size = 0;
    T->X = NULL;
}

ErrorCode tfe_forward(TextFeatureExtractor *ext, NeuralNetwork *net) {
    if (ext == NULL || net == NULL) return TFE_ERR_NULL_POINTER;
    if (ext->vocab_size <= 0 || ext->X == NULL) return TFE_ERR_INVALID_TEXT;
    if (net->W == NULL) return TFE_ERR_NULL_POINTER;

    const int out_dim = net->out_dim;
    for (int i = 0; i < out_dim; i++) {
        net->Y[i] = net->B[i];
    }
    for (int in = 0; in < ext->vocab_size; in++) {
        for (int i = 0; i < out_dim; i++) {
            net->Y[i] += ext->X[in] * net->W[in * out_dim + i];
        }
    }
    return TFE_SUCCESS;
}

ErrorCode tfe_backward(TextFeatureExtractor *ext, NeuralNetwork *net, const double *Y_true, double lr) {
    if (ext == NULL || net == NULL || Y_true == NULL) return TFE_ERR_NULL_POINTER;
    if (ext->X == NULL || net->W == NULL) return TFE_ERR_NULL_POINTER;
    const int out_dim = net->out_dim;
    for (int i = 0; i < out_dim; i++) {
        double err = Y_true[i] - net->Y[i];
        for (int in = 0; in < net->in_dim; in++) {
            net->W[in * out_dim + i] += lr * err * ext->X[in];
        }
        net->B[i] += lr * err;
    }
    return TFE_SUCCESS;
}

void train_batch_for_python(NeuralNetwork *net, double *batch_X, double *batch_Y, int batch_size, double lr) {
    if (net == NULL || batch_X == NULL || batch_Y == NULL) return;

    int in_dim = net->in_dim;
    int out_dim = net->out_dim;

    for (int b = 0; b < batch_size; b++) {
        double *X = &batch_X[b * in_dim];
        double *Y_true = &batch_Y[b * out_dim];

        // Forward
        for (int i = 0; i < out_dim; i++) net->Y[i] = net->B[i];
        for (int in = 0; in < in_dim; in++) {
            if (X[in] == 0.0) continue; // 稀疏优化
            for (int i = 0; i < out_dim; i++) {
                net->Y[i] += X[in] * net->W[in * out_dim + i];
            }
        }

        // Backward
        for (int i = 0; i < out_dim; i++) {
            double err = Y_true[i] - net->Y[i];
            for (int in = 0; in < in_dim; in++) {
                if (X[in] == 0.0) continue;
                net->W[in * out_dim + i] += lr * err * X[in];
            }
            net->B[i] += lr * err;
        }
    }
}

void forward_only_for_python(NeuralNetwork *net, double *X, double *out_y) {
    int in_dim = net->in_dim;
    int out_dim = net->out_dim;

    for (int i = 0; i < out_dim; i++) net->Y[i] = net->B[i];
    for (int in = 0; in < in_dim; in++) {
        if (X[in] == 0.0) continue;
        for (int i = 0; i < out_dim; i++) {
            net->Y[i] += X[in] * net->W[in * out_dim + i];
        }
    }
    for (int i = 0; i < out_dim; i++) out_y[i] = net->Y[i];
}


int save_weights(NeuralNetwork *net, const char *filename) {
    if (net == NULL || net->W == NULL || filename == NULL) return -1;
    FILE *fp = fopen(filename, "wb");
    if (fp == NULL) return -2;

    fwrite(&net->in_dim, sizeof(int), 1, fp);
    fwrite(&net->out_dim, sizeof(int), 1, fp);
    fwrite(net->W, sizeof(double), net->in_dim * net->out_dim, fp);
    fwrite(net->B, sizeof(double), 3, fp);

    fclose(fp);
    return 0;
}

int load_weights(NeuralNetwork *net, const char *filename) {
    if (net == NULL || filename == NULL) return -1;
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL) return -2;

    int in_dim, out_dim;
    fread(&in_dim, sizeof(int), 1, fp);
    fread(&out_dim, sizeof(int), 1, fp);

    if (net->W != NULL) free(net->W);
    net->W = (double*)malloc(sizeof(double) * in_dim * out_dim);
    if (net->W == NULL) { fclose(fp); return -3; }

    net->in_dim = in_dim;
    net->out_dim = out_dim;

    fread(net->W, sizeof(double), in_dim * out_dim, fp);
    fread(net->B, sizeof(double), 3, fp);

    fclose(fp);
    return 0;
}