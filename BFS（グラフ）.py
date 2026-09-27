from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict, deque
from collections.abc import Iterable
from fractions import Fraction
from functools import cache
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
import string
import sys

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
MOD = 998244353
# MOD = 10**9 + 7
DIR4 = [(0, 1), (0, -1), (1, 0), (-1, 0)]
DIR8 = [(-1, 1), (0, 1), (1, 1), (-1, 0), (1, 0), (-1, -1), (0, -1), (1, -1)]
DIR9 = DIR8 + [(0, 0)]
flag, ans = False, 0

# ===== Go Writing =====
N, M = map(int, input().split())
G = [[] for _ in range(N)]
D = deque([0])
dist = [-1] * N
dist[0] = 0

for _ in range(M):
    U, V = map(int, input().split())
    U, V = U - 1, V - 1

    G[U].append(V)
    G[V].append(U)

while D:
    V = D.popleft()

    for i in G[V]:
        if dist[i] != -1:
            continue

        dist[i] = dist[V] + 1
        D.append(i)

for i in dist:
    print(i)
