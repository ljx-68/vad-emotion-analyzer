#include "type.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define TFE_VAD_DIM 3
void predict_for_python(TextFeatureExtractor *ext, NeuralNetwork *net, CSample *csample, double *out_y)
{
    if (ext == NULL || net == NULL || csample == NULL || out_y == NULL) return;

    Sample s;
    s.text = csample->text;
    for (int i = 0; i < 3; i++) s.VAD[i] = csample->VAD[i];

    tfe_extract(ext, &s);
    tfe_forward(ext, net);
    for (int i = 0; i < 3; i++) {
        out_y[i] = net->Y[i];
    }
}
ErrorCode tfe_network_init(
    NeuralNetwork *net,
    int in_dim
    )
{
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

void tfe_network_free(NeuralNetwork *net)
{
    if (net == NULL) return;
    free(net->W);
    net->W = NULL;
    net->in_dim  = 0;
    net->out_dim = 0;
}

ErrorCode tfe_sample_init(Sample *s, const char *text_address, const double *vad){
    if (s == NULL || text_address == NULL || vad == NULL) return TFE_ERR_NULL_POINTER;
    char *copy = (char*)malloc(strlen(text_address) + 1);
    if (copy == NULL) 
        return TFE_ERR_OUT_OF_MEMORY;
    strcpy(copy, text_address);

    s->text = copy;
    memcpy(s->VAD, vad, sizeof(s->VAD));
    return TFE_SUCCESS;
}

void tfe_sample_free(Sample *s){
    if (s == NULL) return;
    free((void *)s->text);
    s->text = NULL;
    memset(s->VAD, 0, sizeof(s->VAD));
}

ErrorCode tfe_extractor_init(TextFeatureExtractor *T, const char **vocad, int vocad_size){
    if (T == NULL || vocad == NULL) return TFE_ERR_NULL_POINTER;
    if (vocad_size <= 0) return TFE_ERR_INVALID_ARG;
    const char **v = (const char**)malloc(sizeof(const char *) * vocad_size);
    if (v == NULL)return TFE_ERR_OUT_OF_MEMORY;

    for (int i = 0; i < vocad_size; i++){
        if (vocad[i] == NULL){
            free((void *)v);
            return TFE_ERR_INVALID_TEXT;
        }
        v[i] = vocad[i];
    }

    double *x = (double*)calloc(vocad_size, sizeof(double));
    if (x == NULL){
        free((void*)v);
        return TFE_ERR_OUT_OF_MEMORY;
    }

    T->vocab = v;
    T->vocab_size = vocad_size;
    T->X = x;
    return TFE_SUCCESS;
}

void tfe_extractor_free(TextFeatureExtractor *T){
    if (T == NULL)return;
    free((void *)T->vocab);
    free(T->X);
    
    T->vocab = NULL;
    T->vocab_size = 0;
    T->X = NULL;
}
//ErrorCode tfe_extract(TextFeatureExtractor *self, Sample *sample){
//    if (sample == NULL || self == NULL) {
//        return TFE_ERR_NULL_POINTER;
//    }
//    if (self->X == NULL || sample->text == NULL) {
//        return TFE_ERR_INVALID_TEXT;
//    }
//    double *T = self->X;
//    for (int i = 0; i < self->vocab_size; i++){
//        const char *word = self->vocab[i];
//        int hit = (word && word[0] && strstr(sample->text, word) != NULL);
//        *T = hit ? 1.0 : 0.0;
//        T++;
//    }
//    return TFE_SUCCESS;
//}
ErrorCode tfe_forward(TextFeatureExtractor *ext, NeuralNetwork *net){
    if (ext == NULL || net == NULL)
        return TFE_ERR_NULL_POINTER;
    if (ext->vocab_size <= 0 || ext->X == NULL)
        return TFE_ERR_INVALID_TEXT;
    if (net->W == NULL)
        return TFE_ERR_NULL_POINTER;
    const int out_dim = net->out_dim;
    for (int i = 0; i < out_dim; i++){
        net->Y[i] = net->B[i];
    }
    for (int in = 0; in < ext->vocab_size; in++){
        for (int i = 0; i < out_dim; i++){
            net->Y[i] += ext->X[in] * net->W[in * out_dim + i];
        }
    }
    return TFE_SUCCESS;
}
ErrorCode tfe_backward(TextFeatureExtractor *ext, NeuralNetwork *net, const double *Y_true, double lr){
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

int save_weights(NeuralNetwork *net, const char *filename) {
    if (net == NULL || net->W == NULL || filename == NULL) return -1;
    FILE *fp = fopen(filename, "wb");
    if (fp == NULL) return -2;

    // 存维度
    fwrite(&net->in_dim, sizeof(int), 1, fp);
    fwrite(&net->out_dim, sizeof(int), 1, fp);
    // 存 W 数组
    fwrite(net->W, sizeof(double), net->in_dim * net->out_dim, fp);
    // 存 B 数组
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

    if (net->W != NULL) free(net->W); // 释放旧内存
    net->W = (double*)malloc(sizeof(double) * in_dim * out_dim);
    if (net->W == NULL) { fclose(fp); return -3; }

    net->in_dim = in_dim;
    net->out_dim = out_dim;

    fread(net->W, sizeof(double), in_dim * out_dim, fp);
    fread(net->B, sizeof(double), 3, fp);

    fclose(fp);
    return 0;
}
void train_batch_for_python(
    NeuralNetwork *net,
    double *batch_X,
    double *batch_Y,
    int batch_size,
    double lr
) {
    if (net == NULL || batch_X == NULL || batch_Y == NULL) return;

    int in_dim = net->in_dim;
    int out_dim = net->out_dim;

    for (int b = 0; b < batch_size; b++) {
        double *X = &batch_X[b * in_dim];
        double *Y_true = &batch_Y[b * out_dim];

        for (int i = 0; i < out_dim; i++) net->Y[i] = net->B[i];
        for (int in = 0; in < in_dim; in++) {
            if (X[in] == 0.0) continue;
            for (int i = 0; i < out_dim; i++) {
                net->Y[i] += X[in] * net->W[in * out_dim + i];
            }
        }

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