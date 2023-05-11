#include "gtest/gtest.h"

#include "tensor.hpp"

GTEST_TEST(Tensor, Mul) {
    tensor::Matrix left = {{1, 2, 3, 4, 5, 6}, 2, 3};
    tensor::Matrix right = {{1, 2, 3, 4, 5, 6}, 3, 2};
    tensor::Matrix actual = tensor::Mul(left, right).eval();
    tensor::Matrix except = {{22, 28, 49, 64}, 2, 2};
    EXPECT_EQ(actual, except);
}