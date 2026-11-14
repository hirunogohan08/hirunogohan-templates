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
import numpy as np
import random
import string
import sys
import time
import operator

# ===== 設定・制限解除 =====
# https://github.com/hirunogohan08/hirunogohan-templates/blob/main/start.py
# https://docs.google.com/document/d/1hXZidjUHJUFaYYOSN34VY6HnevXTv1rL7eIJ5bRqhjQ/edit?tab=t.ylp570je24ai#heading=h.7lfmyfde7zw1
# fmt: off
# --- 実行制限 ---
sys.setrecursionlimit(2 * 10**6)

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)

try:
    import pypyjit  # type: ignore
    pypyjit.set_param("max_unroll_recursion=-1")
except ImportError:
    pass

# --- 入出力 ---
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
N, M = map(int, input().split())
G = [[] for _ in range(N)]

for _ in range(M):
    U, V = map(int, input().split())
    U, V = U - 1, V - 1
    G[U].append(V)
    G[V].append(U)

# R: 根, P: 親, D: 深さ, I: 入った順, Q: 出た順
# T: DFS木（木辺のみ）, B: 後退辺 (子孫, 祖先), C: タイマー
# X: 各頂点が次に見る隣接リストの位置, S: スタック
R = 0
P, D, I, Q = [-1] * N, [0] * N, [-1] * N, [-1] * N
T = [[] for _ in range(N)]
B, X, S = [], [0] * N, [R]
I[R], C = 0, 1

while S:
    U = S[-1]
    if X[U] < len(G[U]):
        V = G[U][X[U]]
        X[U] += 1
        if I[V] == -1:
            P[V], D[V] = U, D[U] + 1
            T[U].append(V)
            I[V], C = C, C + 1
            S.append(V)
        elif V != P[U] and I[V] < I[U]:
            B.append((U, V))
    else:
        Q[U], C = C, C + 1
        S.pop()
