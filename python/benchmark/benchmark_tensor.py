import random
from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt
import load_test

import tensor
import numpy as np
import tensorflow as tf
import torch


def fn_tensor(arg):
    return tensor.Mul(arg, arg).reorder(True).tile(128, 128).parallel(16).eval()


def fn_numpy(arg):
    return np.matmul(arg, arg)


def fn_tensorflow(arg):
    return tf.linalg.matmul(arg, arg)


def fn_torch(arg):
    return torch.matmul(arg, arg)


def gen_tensor_arg_fn(n):
    def arg_fn():
        return tensor.Matrix([random.random() for _ in range(n * n)], n, n)
    return arg_fn


def gen_numpy_arg_fn(n):
    def arg_fn():
        return np.random.rand(n, n)
    return arg_fn


def gen_tensorflow_arg_fn(n):
    def arg_fn():
        return tf.random.uniform([n, n])
    return arg_fn


def gen_torch_arg_fn(n):
    def arg_fn():
        return torch.rand(n, n)
    return arg_fn


def format_d(d):
    if d == 0:
        return "0"
    if d * 100 >= 1:
        return "{:.3}".format(d)
    return "{:.2E}".format(d)


print()
print()

console = Console()
fn_list = [
    fn_tensor,
    fn_numpy,
    fn_tensorflow,
    fn_torch
]
name_arg_res_list_list = [
    [(2 ** _, gen_tensor_arg_fn(2 ** _)) for _ in range(11)],
    [(2 ** _, gen_numpy_arg_fn(2 ** _)) for _ in range(11)],
    [(2 ** _, gen_tensorflow_arg_fn(2 ** _)) for _ in range(11)],
    [(2 ** _, gen_torch_arg_fn(2 ** _)) for _ in range(11)],
]
for fn, name_arg_res_list in zip(fn_list, name_arg_res_list_list):
    name = fn.__name__.split("fn_")[1]
    load = load_test.module_load_test(fn, name_arg_res_list, skip=True)

    table = Table(title=name)
    table.add_column("scale (n)", justify="right")
    table.add_column("latency (s)", justify="right")
    for _ in load:
        table.add_row(f"{_[0]}", format_d(_[1]))
    console.print(table)

    x = [_[0] for _ in load]
    y = [_[1] for _ in load]
    plt.plot(x, y, label=name)
plt.legend(loc='upper center')
plt.show()
