#include <vector>
#include <string>

#ifndef TENSOR_H
#define TENSOR_H

namespace tensor {

class Matrix {
    friend class Mul;
    friend std::ostream &operator<<(std::ostream &os, const Matrix &m);
    friend bool operator==(const Matrix &lhs, const Matrix &rhs);

private:
    std::vector<float> data_;
    int M_;
    int N_;

public:
    Matrix(std::vector<float> &&data, int M, int N);
    explicit operator std::string() const;
};

bool operator==(const Matrix &lhs, const Matrix &rhs);
std::ostream &operator<<(std::ostream &os, const Matrix &m);

class Mul {
private:
    Matrix left_;
    Matrix right_;

    bool reorder_ = false;
    int block_i_ = 0;
    int block_j_ = 0;
    int parallel_ = 1;

public:
    Mul(Matrix &left, Matrix &right);
    Mul reorder(bool reorder_n);
    Mul tile(int block_i, int block_j);
    Mul parallel(int parallel_n);
    Matrix eval() const;
};

}

#endif