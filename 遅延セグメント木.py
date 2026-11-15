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


# ===== 遅延セグメント木 =====
# https://github.com/hirunogohan08/hirunogohan-templates/edit/main/%E9%81%85%E5%BB%B6%E3%82%BB%E3%82%B0%E3%83%A1%E3%83%B3%E3%83%88%E6%9C%A8.py
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
            for i in range(X - 1, 0, -1):
                S[i] = S[i << 1] + S[(i << 1) | 1]

        if U:
            U[X : X + N] = T
            for i in range(X - 1, 0, -1):
                a, b = U[i << 1], U[(i << 1) | 1]
                U[i] = max(b, a)

        if D:
            D[X : X + N] = T
            for i in range(X - 1, 0, -1):
                a, b = D[i << 1], D[(i << 1) | 1]
                D[i] = min(b, a)

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

    # 説明：区間 [L, R) の両端に関わるノードの遅延を、上から順に子へ渡して空にする(内部用)
    #       遅延を渡す処理(「値 → B * 値 + C」の適用と、子の遅延への合成)はここに直接書いてある
    #       ※子が葉(i == 1)のときは、葉に遅延はないので合成しない
    # 入力：L = 左端(含む), R = 右端(含まない)
    # 出力：(L + X, R + X)(葉の番号に直した値)
    # 計算量：O(log N)
    def _prep(self, L, R):
        X, H, S, U, D, P, Q = self.X, self.H, self.S, self.U, self.D, self.P, self.Q
        L, R = L + X, R + X

        for i in range(H, 0, -1):
            a = (L >> i) if ((L >> i) << i) != L else 0
            b = ((R - 1) >> i) if ((R >> i) << i) != R else 0

            if a == b:
                if a == 0:
                    continue

                ks = (a,)
            else:
                ks = (a, b)

            z = 1 << (i - 1)

            for K in ks:
                if K == 0:
                    continue

                B, C = P[K], Q[K]

                if B == 1 and C == 0:
                    continue

                l = K << 1
                r = l | 1

                if S:
                    c = C * z
                    S[l] = B * S[l] + c
                    S[r] = B * S[r] + c

                if U:
                    U[l] = B * U[l] + C
                    U[r] = B * U[r] + C

                if D:
                    D[l] = B * D[l] + C
                    D[r] = B * D[r] + C

                if i > 1:
                    P[l], Q[l] = B * P[l], B * Q[l] + C
                    P[r], Q[r] = B * P[r], B * Q[r] + C

                P[K], Q[K] = 1, 0

        return L, R

    # 説明：区間の両端に関わるノードを、下から順に2つの子から再計算する(内部用。遅延は _prep で空にしてあること)
    # 入力：J = 左端の葉の番号(L + X), K = 右端の葉の番号(R + X, 含まない)
    # 出力：なし
    # 計算量：O(log N)
    def _fix(self, J, K):
        H, S, U, D = self.H, self.S, self.U, self.D

        for i in range(1, H + 1):
            a = (J >> i) if ((J >> i) << i) != J else 0
            b = ((K - 1) >> i) if ((K >> i) << i) != K else 0

            if a == b:
                if a == 0:
                    continue

                ks = (a,)
            else:
                ks = (a, b)

            for k in ks:
                if k == 0:
                    continue

                l = k << 1
                r = l | 1

                if S:
                    S[k] = S[l] + S[r]

                if U:
                    x, y = U[l], U[r]
                    U[k] = max(y, x)

                if D:
                    x, y = D[l], D[r]
                    D[k] = min(y, x)

    # 説明：区間加算  A[L], ..., A[R - 1] に C を加える
    #       ※ノードの幅 z(そのノードが担当する葉の数)を、ループの中で 1, 2, 4, ... と持ち回る
    # 入力：L = 左端(含む), R = 右端(含まない), C = 加える値
    # 出力：なし
    # 計算量：O(log N)
    def add(self, L, R, C):
        if L == R:
            return

        L, R = self._prep(L, R)
        J, K = L, R
        S, U, D, Q = self.S, self.U, self.D, self.Q
        z = 1

        while L < R:
            if L & 1:
                if S:
                    S[L] += C * z
                if U:
                    U[L] += C
                if D:
                    D[L] += C
                if z > 1:
                    Q[L] += C
                L += 1

            if R & 1:
                R -= 1
                if S:
                    S[R] += C * z
                if U:
                    U[R] += C
                if D:
                    D[R] += C
                if z > 1:
                    Q[R] += C

            L, R, z = L >> 1, R >> 1, z << 1

        self._fix(J, K)

    # 説明：区間代入  A[L], ..., A[R - 1] を C にする
    # 入力：L = 左端(含む), R = 右端(含まない), C = 入れる値
    # 出力：なし
    # 計算量：O(log N)
    def update(self, L, R, C):
        if L == R:
            return

        L, R = self._prep(L, R)
        J, K = L, R
        S, U, D, P, Q = self.S, self.U, self.D, self.P, self.Q
        z = 1

        while L < R:
            if L & 1:
                if S:
                    S[L] = C * z
                if U:
                    U[L] = C
                if D:
                    D[L] = C
                if z > 1:
                    P[L], Q[L] = 0, C
                L += 1

            if R & 1:
                R -= 1
                if S:
                    S[R] = C * z
                if U:
                    U[R] = C
                if D:
                    D[R] = C
                if z > 1:
                    P[R], Q[R] = 0, C

            L, R, z = L >> 1, R >> 1, z << 1

        self._fix(J, K)

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
    # 出力：最大値(空区間は -(1 << 100))
    # 計算量：O(log N)
    def querymax(self, L, R):
        L, R = self._prep(L, R)
        T, ans = self.U, self.M

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
    # 出力：最小値(空区間は 1 << 100)
    # 計算量：O(log N)
    def querymin(self, L, R):
        L, R = self._prep(L, R)
        T, ans = self.D, self.E

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
