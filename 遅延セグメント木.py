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


# ===== 遅延セグメント木 =====
class LazySegTree:
    # 説明：遅延セグ木を作る。区間加算・区間代入と、区間和・区間最大・区間最小ができる
    #       使う機能を F で選べる(S=区間和 / M=区間最大 / N=区間最小)。使わない機能は配列を作らず更新もしない
    #       遅延は (P, Q) = 「値 → P * 値 + Q」の形で持つ(加算は P = 1, 代入は P = 0)
    #       ※値は整数のみ。最大・最小は値だけを返す(位置は返さない)。区間は 0 <= L <= R <= N で指定すること
    #       ※M, E は max/min の単位元(-無限, +無限), H は木の高さ
    # 入力：N = 配列の長さ, A = 初期配列(整数のlist, len(A) == N であること), F = 使う機能(省略時は全部)
    # 出力：なし
    # 計算量：O(N)
    def __init__(self, N, A, F="SMN"):
        assert len(A) == N, "len(A) と N が違う"
        assert set(F) <= set("SMN"), "F は S M N のどれか"
        self.N, self.X, self.F, self.M, self.E = N, 1, F, -(1 << 100), 1 << 100

        while self.X < N:
            self.X *= 2

        X, K, T = self.X, 2 * self.X, list(A)
        self.H = X.bit_length() - 1
        S, U, D = (
            [0] * K if "S" in F else None,
            [self.M] * K if "M" in F else None,
            [self.E] * K if "N" in F else None,
        )
        self.S, self.U, self.D = S, U, D
        self.P, self.Q = [1] * X, [0] * X

        if S:
            S[X : X + N] = T
        if U:
            U[X : X + N] = T
        if D:
            D[X : X + N] = T

        if S:
            for i in range(X - 1, 0, -1):
                S[i] = S[i << 1] + S[(i << 1) | 1]

        if U:
            for i in range(X - 1, 0, -1):
                L, R = U[i << 1], U[(i << 1) | 1]
                U[i] = max(L, R)

        if D:
            for i in range(X - 1, 0, -1):
                L, R = D[i << 1], D[(i << 1) | 1]
                D[i] = min(L, R)

        for i, j in zip("SMN", ("querysum", "querymax", "querymin")):
            if i not in F:
                setattr(self, j, self._off(i, j))

    # 説明：無効な機能のクエリ用に、呼ぶと理由つきでエラーになる関数を作る(内部用)
    # 入力：C = 機能の文字(S M N), Q = クエリ名
    # 出力：呼ぶと RuntimeError を出す関数
    # 計算量：O(1)
    @staticmethod
    def _off(C, Q):
        def E(*_):
            raise RuntimeError(f"{Q} は無効: F に '{C}' を入れること")

        return E

    # 説明：ノード K に「値 → B * 値 + C」を適用する(内部用)。葉でなければ遅延も合成する
    # 入力：K = ノード番号, B = 掛ける数(加算は 1, 代入は 0), C = 足す数
    # 出力：なし
    # 計算量：O(1)
    def _ap(self, K, B, C):
        if self.S:
            self.S[K] = B * self.S[K] + C * (self.X >> (K.bit_length() - 1))
        if self.U:
            self.U[K] = B * self.U[K] + C
        if self.D:
            self.D[K] = B * self.D[K] + C

        if K < self.X:
            self.P[K], self.Q[K] = B * self.P[K], B * self.Q[K] + C

    # 説明：ノード K の遅延を2つの子に渡して、K の遅延を空にする(内部用)
    # 入力：K = ノード番号(葉でないこと)
    # 出力：なし
    # 計算量：O(1)
    def _push(self, K):
        B, C = self.P[K], self.Q[K]

        if B == 1 and C == 0:
            return

        self._ap(K << 1, B, C)
        self._ap((K << 1) | 1, B, C)
        self.P[K], self.Q[K] = 1, 0

    # 説明：2つの子からノード K を再計算する(内部用。K の遅延は空であること)
    # 入力：K = ノード番号(葉でないこと)
    # 出力：なし
    # 計算量：O(1)
    def _pull(self, K):
        L, R = K << 1, (K << 1) | 1

        if self.S:
            self.S[K] = self.S[L] + self.S[R]

        if self.U:
            self.U[K] = max(self.U[R], self.U[L])

        if self.D:
            self.D[K] = min(self.D[R], self.D[L])

    # 説明：区間 [L, R) の両端に関わるノードの遅延を、上から順に子へ渡す(内部用)
    # 入力：L = 左端(含む), R = 右端(含まない)
    # 出力：(L + X, R + X)(葉の番号に直した値)
    # 計算量：O(log N)
    def _prep(self, L, R):
        L, R = L + self.X, R + self.X

        for i in range(self.H, 0, -1):
            if ((L >> i) << i) != L:
                self._push(L >> i)

            if ((R >> i) << i) != R:
                self._push((R - 1) >> i)

        return L, R

    # 説明：区間 [L, R) に「値 → B * 値 + C」を適用する(内部用)
    # 入力：L = 左端(含む), R = 右端(含まない), B = 掛ける数(加算は 1, 代入は 0), C = 足す数
    # 出力：なし
    # 計算量：O(log N)
    def _range(self, L, R, B, C):
        if L == R:
            return

        L, R = self._prep(L, R)
        J, K = L, R

        while L < R:
            if L & 1:
                self._ap(L, B, C)
                L += 1

            if R & 1:
                R -= 1
                self._ap(R, B, C)

            L, R = L >> 1, R >> 1

        for i in range(1, self.H + 1):
            if ((J >> i) << i) != J:
                self._pull(J >> i)

            if ((K >> i) << i) != K:
                self._pull((K - 1) >> i)

    # 説明：区間加算  A[L], ..., A[R - 1] に X を加える
    # 入力：L = 左端(含む), R = 右端(含まない), X = 加える値
    # 出力：なし
    # 計算量：O(log N)
    def add(self, L, R, X):
        self._range(L, R, 1, X)

    # 説明：区間代入  A[L], ..., A[R - 1] を X にする
    # 入力：L = 左端(含む), R = 右端(含まない), X = 入れる値
    # 出力：なし
    # 計算量：O(log N)
    def update(self, L, R, X):
        self._range(L, R, 0, X)

    # 説明：区間 [L, R) の和を求める(F に "S")
    # 入力：L = 左端(含む), R = 右端(含まない)
    # 出力：和(空区間は 0)
    # 計算量：O(log N)
    def querysum(self, L, R):
        L, R = self._prep(L, R)
        T, ans = self.S, 0

        while L < R:
            if L & 1:
                ans += T[L]
                L += 1

            if R & 1:
                R -= 1
                ans += T[R]

            L, R = L >> 1, R >> 1

        return ans

    # 説明：区間 [L, R) の最大値を求める(F に "M")
    # 入力：L = 左端(含む), R = 右端(含まない)
    # 出力：最大値(空区間は -INF)
    # 計算量：O(log N)
    def querymax(self, L, R):
        L, R = self._prep(L, R)
        T, ans = self.U, -INF

        while L < R:
            if L & 1:
                ans = max(ans, T[L])

                L += 1

            if R & 1:
                R -= 1

                ans = max(ans, T[R])

            L, R = L >> 1, R >> 1

        return ans

    # 説明：区間 [L, R) の最小値を求める(F に "N")
    # 入力：L = 左端(含む), R = 右端(含まない)
    # 出力：最小値(空区間は INF)
    # 計算量：O(log N)
    def querymin(self, L, R):
        L, R = self._prep(L, R)
        T, ans = self.D, INF

        while L < R:
            if L & 1:
                ans = min(ans, T[L])

                L += 1

            if R & 1:
                R -= 1

                ans = min(ans, T[R])

            L, R = L >> 1, R >> 1

        return ans


# ===== Go Writing =====
