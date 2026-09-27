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


# ===== セグメント木 =====
class SegTree:
    # セグ木の初期化
    # 入力: N = 配列の長さ, A = 初期配列(list)
    # 計算量: O(N)
    def __init__(self, N, A):
        self.N, self.A = N, list(A)

        self.X = 1
        while self.X < N:
            self.X *= 2

        K = 2 * self.X
        self.S, self.U, self.D = [0] * K, [(-INF, -1)] * K, [(INF, -1)] * K
        self.H = {}

        for i in range(N):
            self._set_leaf(i, self.A[i])
            self.H.setdefault(self.A[i], set()).add(i)

        for i in range(self.X - 1, 0, -1):
            self._pull(i)

    # 葉(末端)の値をセットする(内部用)
    # 入力: I = 配列のindex, X = 入れる値
    # 計算量: O(1)
    def _set_leaf(self, I, X):
        I += self.X
        self.S[I], self.U[I], self.D[I] = X, (X, I - self.X), (X, I - self.X)

    # 子2つから親1つを再計算する(内部用)
    # 入力: I = 親ノードの番号(セグ木上の番号)
    # 計算量: O(1)
    def _pull(self, I):
        L, R = I * 2, I * 2 + 1
        self.S[I] = self.S[L] + self.S[R]
        self.U[I], self.D[I] = max(self.U[L], self.U[R]), min(self.D[L], self.D[R])

    # 葉から根まで再計算する(内部用)
    # 入力: I = 配列のindex
    # 計算量: O(log N)
    def _rebuild(self, I):
        I = (I + self.X) // 2

        while I >= 1:
            self._pull(I)
            I //= 2

    # 値 -> indexの対応を付け替える(内部用)
    # 入力: I = 配列のindex, P = 古い値, Q = 新しい値
    # 計算量: O(1)
    def _move_pos(self, I, P, Q):
        self.H[P].discard(I)

        if not self.H[P]:
            del self.H[P]

        self.H.setdefault(Q, set()).add(I)

    # セグ木の更新(1点)  A[I] を X にする
    # 入力: I = 更新するindex, X = 新しい値
    # 計算量: O(log N)
    def update(self, I, X):
        P = self.A[I]

        if P == X:
            return

        self._move_pos(I, P, X)
        self.A[I] = X
        self._set_leaf(I, X)
        self._rebuild(I)

    # セグ木の更新(交換)  A[I] と A[J] を入れ替える
    # 入力: I, J = 交換する2つのindex
    # 計算量: O(log N)
    def switch(self, I, J):
        P, Q = self.A[I], self.A[J]

        if I == J or P == Q:
            return

        self.A[I], self.A[J] = Q, P
        self._move_pos(I, P, Q)
        self._move_pos(J, Q, P)

        self._set_leaf(I, Q)
        self._set_leaf(J, P)

        self._rebuild(I)
        self._rebuild(J)

    # indexを入手  値 X のindexを返す(なければ -1, 複数あれば最小)
    # 入力: X = 探したい値
    # 計算量: O(1) ※値が重複しない場合。重複ありは同じ値の個数分
    def queryindex(self, X):
        T = self.H.get(X)

        return min(T) if T else -1

    # 合計値を入手  区間 [L, R) の和を返す
    # 入力: L = 左端(含む), R = 右端(含まない)
    # 計算量: O(log N)
    def querysum(self, L, R):
        L, R = L + self.X, R + self.X
        ans = 0

        while L < R:
            if L & 1:
                ans += self.S[L]
                L += 1

            if R & 1:
                R -= 1
                ans += self.S[R]

            L, R = L >> 1, R >> 1

        return ans

    # 最大値を入手  区間 [L, R) の (最大値, index) を返す(空区間は -INF, 同値は indexが大きい方)
    # 入力: L = 左端(含む), R = 右端(含まない)
    # 計算量: O(log N)
    def querymax(self, L, R):
        L, R = L + self.X, R + self.X
        ans = (-INF, -1)

        while L < R:
            if L & 1:
                ans = max(ans, self.U[L])
                L += 1

            if R & 1:
                R -= 1
                ans = max(ans, self.U[R])

            L, R = L >> 1, R >> 1

        return ans

    # 最小値を入手  区間 [L, R) の (最小値, index) を返す(空区間は INF, 同値は indexが小さい方)
    # 入力: L = 左端(含む), R = 右端(含まない)
    # 計算量: O(log N)
    def querymin(self, L, R):
        L, R = L + self.X, R + self.X
        ans = (INF, -1)

        while L < R:
            if L & 1:
                ans = min(ans, self.D[L])
                L += 1

            if R & 1:
                R -= 1
                ans = min(ans, self.D[R])

            L, R = L >> 1, R >> 1

        return ans


# ===== Go Writing =====
