#pragma once

#include <vector>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <numeric>

namespace native_nn {

// ============================================================================
// TENSOR CLASS: Core data structure for all neural network operations
// ============================================================================

class Tensor {
public:
    enum class DataType {
        FP32,   // 32-bit floating point
        FP16,   // 16-bit floating point (half precision)
        INT8    // 8-bit integer (quantized)
    };

    // Constructor: Create tensor with shape
    Tensor(const std::vector<size_t>& shape, DataType dtype = DataType::FP32)
        : shape_(shape), dtype_(dtype), grad_(nullptr) {
        size_t total_elements = compute_size();
        
        switch (dtype_) {
            case DataType::FP32:
                data_fp32_ = std::make_unique<float[]>(total_elements);
                std::fill(data_fp32_.get(), data_fp32_.get() + total_elements, 0.0f);
                break;
            case DataType::FP16:
                data_fp16_ = std::make_unique<uint16_t[]>(total_elements);
                std::fill(data_fp16_.get(), data_fp16_.get() + total_elements, 0);
                break;
            case DataType::INT8:
                data_int8_ = std::make_unique<int8_t[]>(total_elements);
                std::fill(data_int8_.get(), data_int8_.get() + total_elements, 0);
                break;
        }
    }

    // Copy constructor
    Tensor(const Tensor& other) 
        : shape_(other.shape_), dtype_(other.dtype_), 
          scale_(other.scale_), zero_point_(other.zero_point_) {
        size_t total = compute_size();
        
        switch (dtype_) {
            case DataType::FP32:
                data_fp32_ = std::make_unique<float[]>(total);
                std::copy(other.data_fp32_.get(), other.data_fp32_.get() + total, 
                         data_fp32_.get());
                break;
            case DataType::FP16:
                data_fp16_ = std::make_unique<uint16_t[]>(total);
                std::copy(other.data_fp16_.get(), other.data_fp16_.get() + total, 
                         data_fp16_.get());
                break;
            case DataType::INT8:
                data_int8_ = std::make_unique<int8_t[]>(total);
                std::copy(other.data_int8_.get(), other.data_int8_.get() + total, 
                         data_int8_.get());
                break;
        }
    }

    // Move constructor
    Tensor(Tensor&& other) noexcept 
        : shape_(std::move(other.shape_)), dtype_(other.dtype_),
          data_fp32_(std::move(other.data_fp32_)),
          data_fp16_(std::move(other.data_fp16_)),
          data_int8_(std::move(other.data_int8_)),
          scale_(other.scale_), zero_point_(other.zero_point_),
          grad_(std::move(other.grad_)),
          requires_grad_(other.requires_grad_) {}

    ~Tensor() = default;

    // Get raw data pointers
    float* data_fp32() { return data_fp32_.get(); }
    const float* data_fp32() const { return data_fp32_.get(); }
    
    uint16_t* data_fp16() { return data_fp16_.get(); }
    const uint16_t* data_fp16() const { return data_fp16_.get(); }
    
    int8_t* data_int8() { return data_int8_.get(); }
    const int8_t* data_int8() const { return data_int8_.get(); }

    // Convert between float and FP16
    static uint16_t float_to_fp16(float x);
    static float fp16_to_float(uint16_t x);

    // Quantization to INT8
    void quantize_int8(float scale, int32_t zero_point);
    void dequantize_int8(float* out) const;

    // Shape and size operations
    const std::vector<size_t>& shape() const { return shape_; }
    size_t size() const { return compute_size(); }
    
    size_t dim(size_t i) const { 
        if (i >= shape_.size()) throw std::out_of_range("Dimension out of range");
        return shape_[i]; 
    }
    
    size_t rank() const { return shape_.size(); }

    DataType dtype() const { return dtype_; }

    // Gradient management
    void create_gradient() {
        if (!grad_) grad_ = std::make_unique<Tensor>(shape_, dtype_);
    }
    
    Tensor* gradient() { return grad_.get(); }
    const Tensor* gradient() const { return grad_.get(); }
    
    void zero_grad() {
        if (grad_) {
            size_t total = compute_size();
            switch (dtype_) {
                case DataType::FP32:
                    std::fill(grad_->data_fp32_.get(), grad_->data_fp32_.get() + total, 0.0f);
                    break;
                case DataType::FP16:
                    std::fill(grad_->data_fp16_.get(), grad_->data_fp16_.get() + total, 0);
                    break;
                case DataType::INT8:
                    std::fill(grad_->data_int8_.get(), grad_->data_int8_.get() + total, 0);
                    break;
            }
        }
    }

    void set_requires_grad(bool req) { requires_grad_ = req; }
    bool requires_grad() const { return requires_grad_; }

    // Element access (assumes FP32)
    float& at(const std::vector<size_t>& indices) {
        if (dtype_ != DataType::FP32) 
            throw std::runtime_error("Element access only for FP32 tensors");
        return data_fp32_[compute_index(indices)];
    }

    float at(const std::vector<size_t>& indices) const {
        if (dtype_ != DataType::FP32) 
            throw std::runtime_error("Element access only for FP32 tensors");
        return data_fp32_[compute_index(indices)];
    }

    // Fill with value
    void fill(float value) {
        size_t total = compute_size();
        if (dtype_ == DataType::FP32) {
            std::fill(data_fp32_.get(), data_fp32_.get() + total, value);
        }
    }

    // Initialize with normal distribution
    void normal_(float mean = 0.0f, float stddev = 1.0f);

private:
    std::vector<size_t> shape_;
    DataType dtype_;
    
    std::unique_ptr<float[]> data_fp32_;
    std::unique_ptr<uint16_t[]> data_fp16_;
    std::unique_ptr<int8_t[]> data_int8_;
    
    float scale_ = 1.0f;
    int32_t zero_point_ = 0;
    
    std::unique_ptr<Tensor> grad_;
    bool requires_grad_ = false;

    size_t compute_size() const {
        return std::accumulate(shape_.begin(), shape_.end(), 
                              size_t(1), std::multiplies<size_t>());
    }

    size_t compute_index(const std::vector<size_t>& indices) const {
        size_t idx = 0;
        size_t stride = 1;
        for (int i = indices.size() - 1; i >= 0; --i) {
            idx += indices[i] * stride;
            stride *= shape_[i];
        }
        return idx;
    }
};

// ============================================================================
// INLINE IMPLEMENTATIONS
// ============================================================================

inline uint16_t Tensor::float_to_fp16(float x) {
    // Simplified float to half precision conversion
    uint32_t bits;
    std::memcpy(&bits, &x, sizeof(float));
    
    uint32_t sign = (bits >> 31) & 0x1;
    uint32_t exp = (bits >> 23) & 0xFF;
    uint32_t mant = bits & 0x7FFFFF;
    
    if (exp == 0) return sign << 15;
    if (exp == 0xFF) {
        return (sign << 15) | 0x7C00 | (mant >> 13);
    }
    
    int new_exp = (int)exp - 127 + 15;
    if (new_exp >= 31) return (sign << 15) | 0x7C00;
    if (new_exp <= 0) return sign << 15;
    
    uint16_t half = (sign << 15) | (new_exp << 10) | (mant >> 13);
    return half;
}

inline float Tensor::fp16_to_float(uint16_t x) {
    uint32_t sign = (x >> 15) & 0x1;
    uint32_t exp = (x >> 10) & 0x1F;
    uint32_t mant = x & 0x3FF;
    
    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        exp = 1;
    } else if (exp == 31) {
        float result = mant ? std::nanf("") : std::numeric_limits<float>::infinity();
        return sign ? -result : result;
    }
    
    uint32_t float_exp = (exp - 15 + 127) << 23;
    uint32_t float_mant = mant << 13;
    uint32_t bits = (sign << 31) | float_exp | float_mant;
    
    float result;
    std::memcpy(&result, &bits, sizeof(float));
    return result;
}

inline void Tensor::normal_(float mean, float stddev) {
    if (dtype_ != DataType::FP32)
        throw std::runtime_error("normal_ only supported for FP32");
    
    size_t total = compute_size();
    for (size_t i = 0; i < total; ++i) {
        // Box-Muller transform
        float u1 = (rand() + 1.0f) / (RAND_MAX + 1.0f);
        float u2 = (rand() + 1.0f) / (RAND_MAX + 1.0f);
        float z = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * M_PI * u2);
        data_fp32_[i] = mean + stddev * z;
    }
}

inline void Tensor::quantize_int8(float scale, int32_t zero_point) {
    if (dtype_ != DataType::FP32)
        throw std::runtime_error("Can only quantize from FP32");
    
    scale_ = scale;
    zero_point_ = zero_point;
    
    size_t total = compute_size();
    for (size_t i = 0; i < total; ++i) {
        float val = data_fp32_[i];
        int32_t q = static_cast<int32_t>(std::round(val / scale)) + zero_point;
        q = std::max(-128, std::min(127, q));
        data_int8_[i] = static_cast<int8_t>(q);
    }
}

inline void Tensor::dequantize_int8(float* out) const {
    if (dtype_ != DataType::INT8)
        throw std::runtime_error("Can only dequantize from INT8");
    
    size_t total = compute_size();
    for (size_t i = 0; i < total; ++i) {
        float val = static_cast<float>(data_int8_[i]);
        out[i] = (val - zero_point_) * scale_;
    }
}

} // namespace native_nn
