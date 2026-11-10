#include <stdio.h>
#include "type.h"
#include <time.h>
#include <stdlib.h>
ErrorCode Preloading_Data_Processing(
    TextFeatureExtractor *ext, NeuralNetwork *net, Sample *samples,
    const char **text, int vocab_size,
    const char **texts, const double **vads, int n_samples)
{
    if (ext == NULL || net == NULL || samples == NULL) return TFE_ERR_NULL_POINTER;
    if (text == NULL || vocab_size <= 0 || texts == NULL || vads == NULL || n_samples <= 0)
        return TFE_ERR_INVALID_ARG;

    ErrorCode rc = tfe_extractor_init(ext, text, vocab_size);
    if (rc != TFE_SUCCESS) return rc;          

    rc = tfe_network_init(net, vocab_size);
    if (rc != TFE_SUCCESS) {
        tfe_extractor_free(ext);
        return rc;
    }

    for (int i = 0; i < n_samples; i++) {
        rc = tfe_sample_init(&samples[i], texts[i], vads[i]);
        if (rc != TFE_SUCCESS) {
            for (int j = 0; j < i; j++) tfe_sample_free(&samples[j]);
            tfe_network_free(net);
            tfe_extractor_free(ext);
            return rc;
        }
    }
    return TFE_SUCCESS;
}

int main(void)
{
    srand((unsigned)time(NULL));
    const char *vocab[] = {"我", "很", "开心", "感动", "伤心"};
    int vocab_size = 5;

    const char *texts[3] = {"我很开心", "我很感动", "我很伤心"};
    double v0[3] = {0.43, 0.12, 0.56};
    double v1[3] = {0.42, 0.54, 0.50};
    double v2[3] = {-0.50, 0.30, 0.20};
    const double *vads[3] = {v0, v1, v2};
    int n_samples = 3;

    TextFeatureExtractor ext = {0};
    NeuralNetwork net = {0};
    Sample samples[3] = {{0}};

    if (Preloading_Data_Processing(&ext, &net, samples, vocab, vocab_size,
                                   texts, vads, n_samples) != TFE_SUCCESS) {
        return 1;
    }
    double lr = 0.01;
    for (int epoch = 1; epoch <= 500; epoch++) {
        double epoch_loss = 0.0;
        for (int i = 0; i < n_samples; i++) {
            if (tfe_extract(&ext, &samples[i]) != TFE_SUCCESS) goto cleanup;
            if (tfe_forward(&ext, &net) != TFE_SUCCESS) goto cleanup;
            if (tfe_backward(&ext, &net, samples[i].VAD, lr) != TFE_SUCCESS) goto cleanup;

            for (int j = 0; j < net.out_dim; j++) {
                double d = net.Y[j] - samples[i].VAD[j];
                epoch_loss += d * d;
            }
        }
        if (epoch % 50 == 0)
            printf("Epoch %4d | avg_loss = %.6f\n", epoch, epoch_loss / n_samples);
    }
    printf("\n=== 训练后每条样本 ===\n");
    for (int i = 0; i < n_samples; i++) {
        tfe_extract(&ext, &samples[i]);
        tfe_forward(&ext, &net);
        printf("样本%d: 预测[%.3f,%.3f,%.3f] 目标[%.3f,%.3f,%.3f]\n",
            i, net.Y[0], net.Y[1], net.Y[2],
            samples[i].VAD[0], samples[i].VAD[1], samples[i].VAD[2]);
    }

    Sample test = {0};
    double dummy[3] = {0};

    save_weights(&net, "emotion_model.bin");
    if (save_weights(&net, "emotion_model.bin") == 0) {
        printf("权重已成功保存到 emotion_model.bin\n");
    } else {
        printf("保存权重失败\n");
    }
cleanup:
    for (int i = 0; i < n_samples; i++) tfe_sample_free(&samples[i]);
    tfe_network_free(&net);
    tfe_extractor_free(&ext);
    return 0;
}