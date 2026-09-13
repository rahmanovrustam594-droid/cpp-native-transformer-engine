#pragma once

#include "tensor.hpp"
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>

namespace native_nn {

// ============================================================================
// MATRIX ENGINE: Optimized parallel matrix operations
// ============================================================================

class MatrixEngine {
public:
    MatrixEngine(size_t num_threads = 0) {
        if (num_threads == 0) {
            num_threads = std::thread::hardware_concurrency();
            if (num_threads == 0) num_threads = 4;
        }
        num_threads_ = num_threads;
    }

    ~MatrixEngine() = default;

    // Matrix multiplication: C = A @ B (with optional bias)
    // A: [M, K], B: [K, N] -> C: [M, N]
    void matmul(const Tensor& A, const Tensor& B, Tensor& C, 
                const Tensor* bias = nullptr);

    // Element-wise operations
    void add(const Tensor& A, const Tensor& B, Tensor& C);
    void multiply(const Tensor& A, const Tensor& B, Tensor& C);
    void scalar_multiply(const Tensor& A, float scalar, Tensor& C);

    // Activation functions
    void relu(const Tensor& input, Tensor& output);
    void gelu(const Tensor& input, Tensor& output);
    void swiglu(const Tensor& input, Tensor& output);  // SwiGLU activation
    void softmax(const Tensor& input, Tensor& output, int dim);

    // Normalization
    void layer_norm(const Tensor& input, Tensor& output, 
                    const Tensor& weight, const Tensor& bias, float eps = 1e-5f);
    void rmsnorm(const Tensor& input, Tensor& output, 
                 const Tensor& weight, float eps = 1e-5f);

    // Attention operations
    void scaled_dot_product_attention(const Tensor& Q, const Tensor& K, 
                                     const Tensor& V, Tensor& output,
                                     float scale, const Tensor* mask = nullptr);

    // Rotary Position Embeddings (RoPE)
    void apply_rope(Tensor& Q, Tensor& K, size_t seq_pos, float base = 10000.0f);

    // Transpose
    void transpose(const Tensor& input, Tensor& output);

    // Reshape
    void reshape(const Tensor& input, Tensor& output, const std::vector<size_t>& new_shape);

    size_t num_threads() const { return num_threads_; }

private:
    size_t num_threads_;

    // Helper for parallel execution
    void parallel_for(size_t count, std::function<void(size_t)> fn);

    // Cache-friendly matrix multiplication
    void matmul_block(const float* A, const float* B, float* C,
                     size_t M, size_t K, size_t N, size_t lda, size_t ldb, size_t ldc);
};

// ============================================================================
// GRADIENT COMPUTATION (Autograd)
// ============================================================================

class GradientComputation {
public:
    // Backpropagation for matmul
    static void backward_matmul(const Tensor& A, const Tensor& B, const Tensor& grad_C,
                               Tensor& grad_A, Tensor& grad_B);

    // Backpropagation for element-wise add
    static void backward_add(const Tensor& grad_C, Tensor& grad_A, Tensor& grad_B);

    // Backpropagation for ReLU
    static void backward_relu(const Tensor& input, const Tensor& grad_output,
                             Tensor& grad_input);

    // Backpropagation for softmax
    static void backward_softmax(const Tensor& output, const Tensor& grad_output,
                                Tensor& grad_input);
};

// ============================================================================
// INLINE IMPLEMENTATIONS
// ============================================================================

inline void MatrixEngine::parallel_for(size_t count, std::function<void(size_t)> fn) {
    if (count <= 1 || num_threads_ <= 1) {
        for (size_t i = 0; i < count; ++i) fn(i);
        return;
    }

    std::vector<std::thread> threads;
    size_t chunk_size = (count + num_threads_ - 1) / num_threads_;

    for (size_t t = 0; t < num_threads_; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, count);
        if (start >= count) break;

        threads.emplace_back([start, end, &fn]() {
            for (size_t i = start; i < end; ++i) {
                fn(i);
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

} // namespace native_nn
