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


# ===== Union-Find =====
class UnionFind:
    # Union-Findの初期化
    # 入力: N = 要素数
    # 計算量: O(N)
    def __init__(self, N):
        self.N = N
        self.P, self.S = list(range(N)), [1] * N
        self.C, self.M = N, 1

    # 根を探す(経路圧縮あり・反復版)
    # 入力: I = 根を探したい要素
    # 出力: Iが属するグループの根
    # 計算量: 償却 O(α(N)) ≒ O(1)
    def find(self, I):
        R = I

        while self.P[R] != R:
            R = self.P[R]

        while self.P[I] != R:
            self.P[I], I = R, self.P[I]

        return R

    # 同じグループか判定する
    # 入力: I, J = 判定する2つの要素
    # 出力: 同じグループならTrue, 別ならFalse
    # 計算量: 償却 O(α(N)) ≒ O(1)
    def same(self, I, J):
        flag = self.find(I) == self.find(J)

        return flag

    # グループを結合する
    # 入力: I, J = 結合する2つの要素
    # 出力: 結合したならTrue, 元々同じならFalse
    # 計算量: 償却 O(α(N)) ≒ O(1)
    def union(self, I, J):
        I, J = self.find(I), self.find(J)

        if I == J:
            return False

        if self.S[I] < self.S[J]:
            I, J = J, I

        self.P[J], self.S[I] = I, self.S[I] + self.S[J]
        self.C, self.M = self.C - 1, max(self.M, self.S[I])

        return True

    # 根か判定する
    # 入力: I = 判定する要素
    # 出力: 根ならTrue, 根でなければFalse
    # 計算量: O(1)
    def isroot(self, I):
        flag = self.P[I] == I

        return flag

    # グループの要素数を入手
    # 入力: I = 調べたい要素
    # 出力: Iが属するグループの要素数
    # 計算量: 償却 O(α(N)) ≒ O(1)
    def querysize(self, I):
        return self.S[self.find(I)]

    # グループ数を入手
    # 出力: 現在のグループ数
    # 計算量: O(1)
    def querygroups(self):
        return self.C

    # 最大グループの要素数を入手
    # 出力: 最大グループの要素数
    # 計算量: O(1)
    def querymaxsize(self):
        return self.M

    # Iが属するグループの全要素を入手
    # 入力: I = 調べたい要素
    # 出力: Iと同じグループに属する要素のlist
    # 計算量: O(N α(N))
    def querymembers(self, I):
        I = self.find(I)
        A = []

        for i in range(self.N):
            if self.find(i) == I:
                A.append(i)

        return A

    # 全グループの根を入手
    # 出力: 各グループの根のlist
    # 計算量: O(N α(N))
    def queryroots(self):
        A = []

        for i in range(self.N):
            if self.find(i) == i:
                A.append(i)

        return A

    # 全グループを入手
    # 出力: [[グループ1の要素], [グループ2の要素], ...]
    # 計算量: O(N α(N))
    def queryall(self):
        A = [[] for _ in range(self.N)]

        for i in range(self.N):
            A[self.find(i)].append(i)

        return [o for o in A if o]

    # 初期状態に戻す
    # 計算量: O(N)
    def reset(self):
        self.P, self.S = list(range(self.N)), [1] * self.N
        self.C, self.M = self.N, 1


# ===== Go Writing =====
