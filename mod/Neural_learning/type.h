#ifndef TPYE_H
#define TPYE_H
#include <string.h>
#include <stddef.h>
#ifdef __cplusplus

extern "C"{
#endif
#define TFE_MAX_TEXT_LEN 256

#define TFE_VERSION_MAJOR 1
#define TFE_VERSION_MINOR 0
#define TFE_VERSION_PATCH 0

typedef struct {
    char text[TFE_MAX_TEXT_LEN];
    double VAD[3];
} CSample;

typedef enum {
    TFE_SUCCESS          =  0,   // 成功
    TFE_ERR_NULL_POINTER = -1,   // 空指针错误
    TFE_ERR_INVALID_TEXT = -2,   // 文本无效
    TFE_ERR_OUT_OF_MEMORY= -3,   // 内存分配失败
    TFE_ERR_INVALID_ARG  = -4,   // 参数非法
} ErrorCode;

typedef struct {
    const char **vocab; 
    int vocab_size;
    double *X;
} TextFeatureExtractor;

typedef struct {
    const char *text;
    double VAD[3];
} Sample;

typedef struct {
    double *W;       
    double B[3];
    double Y[3];
    int in_dim;
    int out_dim;     
} NeuralNetwork;

ErrorCode tfe_forward(TextFeatureExtractor *ext, NeuralNetwork *net);
ErrorCode tfe_backward(TextFeatureExtractor *ext, NeuralNetwork *net, const double *Y_true, double lr);

void train_batch_for_python(NeuralNetwork *net, double *batch_X, double *batch_Y, int batch_size, double lr);
void forward_only_for_python(NeuralNetwork *net, double *X, double *out_y);

int save_weights(NeuralNetwork *net, const char *filename);
int load_weights(NeuralNetwork *net, const char *filename);
#ifdef __cpluspleus
}
#endif

#endif