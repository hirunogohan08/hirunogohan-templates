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
# https://github.com/hirunogohan08/hirunogohan-templates/blob/main/BFS%EF%BC%88%E3%82%B0%E3%83%AA%E3%83%83%E3%83%89%EF%BC%89.py
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
H, W = map(int, input().split())
G = [list(input().strip()) for _ in range(H)]
D = deque([(0, 0)])
dist = [[-1] * W for _ in range(H)]
cnt = [[0] * W for _ in range(H)]
dist[0][0], cnt[0][0] = 0, 1

while D:
    V = D.popleft()

    for d in DIR4:
        i, j = V[0] + d[0], V[1] + d[1]

        if not (0 <= i < H and 0 <= j < W):
            continue

        if G[i][j] == "#":
            continue

        if dist[i][j] == -1:
            dist[i][j] = dist[V[0]][V[1]] + 1
            cnt[i][j] = cnt[V[0]][V[1]]
            D.append((i, j))
        elif dist[i][j] == dist[V[0]][V[1]] + 1:
            cnt[i][j] += cnt[V[0]][V[1]]

for i in dist:
    print(*i)
print(cnt[-1][-1])
