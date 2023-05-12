import random
from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt
import load_test

import tensor


def fn_basic(arg):
    return tensor.Mul(arg, arg).eval()


def fn_order(arg):
    return tensor.Mul(arg, arg).reorder(True).eval()


def fn_tile(arg):
    return tensor.Mul(arg, arg).reorder(True).tile(64, 64).eval()


def fn_parallel(arg):
    return tensor.Mul(arg, arg).reorder(True).tile(64, 64).parallel(8).eval()


def gen_arg_fn(n):
    def arg_fn():
        return tensor.Matrix([random.random() for _ in range(n * n)], n, n)
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
fn_list = [fn_basic, fn_order, fn_tile, fn_parallel]
name_arg_res_list = [(2 ** _, gen_arg_fn(2 ** _)) for _ in range(11)]
for fn in fn_list:
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
