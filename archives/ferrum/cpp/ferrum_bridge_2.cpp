#ifndef FERRUM_BRIDGE_H
#define FERRUM_BRIDGE_H
#include <stdint.h>
#ifdef __cplusplus
extern "C"{
#endif
void* create_context_(const char* kernels_dir);
void* create_gpu_buffer_(void* ctx, int capacity);
void gemm_(void* stream, void* A, void* B, void* C, uint32_t m, uint32_t n, uint32_t p, uint32_t b);
void gemv_(void* stream, void* A, void* B, void* C, uint32_t m, uint32_t n, uint32_t b);
#ifdef __cplusplus
}
#endif
#endif