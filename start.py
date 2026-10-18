from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict, deque
from collections.abc import Iterable
from copy import deepcopy as copy
from fractions import Fraction
from functools import cache, reduce
from heapq import heappop, heappush
from itertools import (
    combinations,
    combinations_with_replacement,
    groupby,
    permutations,
    product,
    repeat,
    chain,
    accumulate,
)
from sortedcontainers import SortedSet, SortedList, SortedDict
import io
import math
import random
import string
import sys
import time
import operator

# ===== 設定・制限解除 =====
# https://github.com/hirunogohan08/hirunogohan-templates/blob/main/start.py
# fmt: off
sys.setrecursionlimit(2 * 10**6)
input = lambda: sys.stdin.readline().rstrip()
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
def print(*args, sep=" ", end="\n", file=None, flush=False):
    file = file or sys.stdout
    sep = " " if sep is None else sep
    end = "\n" if end is None else end
    file.write(sep.join(map(str, args)) + end)
    flush and file.flush()
# fmt: on

# ===== 定数・リスト =====
LOW = list(string.ascii_lowercase)
UPP = list(string.ascii_uppercase)
NUM = list(string.digits)
INF = float("inf")
# MOD = 998244353
# MOD = 10**9 + 7
DIR4 = [(0, 1), (0, -1), (1, 0), (-1, 0)]
DIR8 = [(-1, 1), (0, 1), (1, 1), (-1, 0), (1, 0), (-1, -1), (0, -1), (1, -1)]
DIR9 = DIR8 + [(0, 0)]
flag, ans, cnt = False, 0, 0

# ===== Go Writing =====
