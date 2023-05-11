#include <iostream>
#include <utility>
#include <vector>
#include <string>
#include <omp.h>
#include <fmt/core.h>

#include "tensor.hpp"

namespace tensor {

inline void eval_block(float *result, const float *left, const float *right,
                       int N, int K,
                       int t0_i, int t1_i, int t0_j, int t1_j) {
    int i0, j0, k;
    for (i0 = t0_i; i0 < t1_i; i0++) {
        for (j0 = t0_j; j0 < t1_j; j0++) {
            #pragma omp simd
            for (k = 0; k < K; k++) {
                result[i0 * N + j0] += left[i0 * K + k] * right[k * N + j0];
            }
        }
    }
}

inline void eval_block_reorder(float *result, const float *left, const float *right,
                               int N, int K,
                               int t0_i, int t1_i, int t0_j, int t1_j) {
    int i0, j0, k;
    for (i0 = t0_i; i0 < t1_i; i0++) {
        for (k = 0; k < K; k++) {
            #pragma omp simd
            for (j0 = t0_j; j0 < t1_j; j0++) {
                result[i0 * N + j0] += left[i0 * K + k] * right[k * N + j0];
            }
        }
    }
}

Matrix::Matrix(std::vector<float> &&data, int M, int N) : data_(data), M_(M), N_(N) {};

Matrix::operator std::string() const {
    std::string result = "[\n";
    for (int i = 0; i < M_; i++) {
        result += " [";
        for (int j = 0; j < N_; j++) {
            result += fmt::format("{}", data_[i * N_ + j]);
            if (j < N_ - 1) {
                result += ", ";
            }
        }
        result += "]";
        if (i < M_ - 1) {
            result += ",";
        }
        result += "\n";
    }
    result += "]";
    return result;
}

bool operator==(const Matrix &lhs, const Matrix &rhs) {
    if (lhs.M_ != rhs.M_ || lhs.N_ != rhs.N_) {
        return false;
    }
    for (int i = 0; i < lhs.M_; i++) {
        for (int j = 0; j < lhs.N_; j++) {
            if (lhs.data_[i * lhs.N_ + j] != rhs.data_[i * lhs.N_ + j]) {
                return false;
            }
        }
    }
    return true;
}

std::ostream &operator<<(std::ostream &os, const Matrix &m) {
    return os << static_cast<std::string>(m);
}

Mul::Mul(Matrix &left, Matrix &right) : left_(left), right_(right) {
    if (left_.N_ != right_.M_) {
        throw std::invalid_argument("left column dim not equal right row dim");
    }
};

Mul Mul::reorder(bool reorder_n) {
    reorder_ = reorder_n;
    return *this;
}

Mul Mul::tile(int block_i, int block_j) {
    block_i_ = block_i;
    block_j_ = block_j;
    return *this;
}

Mul Mul::parallel(int parallel_n) {
    parallel_ = parallel_n;
    return *this;
}

Matrix Mul::eval() const {
    int M = left_.M_;
    int N = right_.N_;
    int K = left_.N_;
    const float *left = left_.data_.data();
    const float *right = right_.data_.data();

    int block_i = block_i_ != 0 ? block_i_ : M;
    int block_j = block_j_ != 0 ? block_j_ : N;
    int grid_i = M / block_i + (M % block_i != 0);
    int grid_j = N / block_j + (N % block_j != 0);

    std::vector<float> res(M * N, 0);
    float *result = res.data();

    omp_set_num_threads(parallel_);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < grid_i; i++) {
        for (int j = 0; j < grid_j; j++) {
            int t0_i = i * block_i;
            int t1_i = std::min(t0_i + block_i, M);
            int t0_j = j * block_j;
            int t1_j = std::min(t0_j + block_j, N);
            if (!reorder_) {
                eval_block(result, left, right, N, K, t0_i, t1_i, t0_j, t1_j);
            } else {
                eval_block_reorder(result, left, right, N, K, t0_i, t1_i, t0_j, t1_j);
            }
        }
    }

    return {std::move(res), M, N};
}

}