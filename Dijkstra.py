from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict, deque
from collections.abc import Iterable
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
)
import io
import math
import random
import string
import sys
import time
import operator

# ===== 設定・制限解除 =====
# fmt: off
sys.setrecursionlimit(2 * 10**6)
input = sys.stdin.readline
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
def print(*args, sep=" ", end="\n"):
    sys.stdout.write(sep.join(map(str, args)) + end)
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
D = [(0, 0)]
dist = [INF] * N
dist[0] = 0

for _ in range(M):
    U, V, C = map(int, input().split())
    U, V = U - 1, V - 1

    G[U].append((V, C))
    G[V].append((U, C))

while D:
    V, C = heappop(D)

    if V > dist[C]:
        continue

    for i, j in G[C]:
        A = V + j

        if A < dist[i]:
            dist[i] = A
            heappush(D, (A, i))

if dist[-1] == INF:
    print(-1)
else:
    print(dist[-1])
