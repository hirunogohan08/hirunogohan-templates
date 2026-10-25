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


# ===== セグメント木 =====
# https://github.com/hirunogohan08/hirunogohan-templates/blob/main/%E3%82%BB%E3%82%B0%E3%83%A1%E3%83%B3%E3%83%88%E6%9C%A8.py
class SegTree:
    # 説明：セグ木を作る。使う機能を F で選べる(使わない機能は配列を作らず更新もしないので速く・軽い)
    #       S=和 / X=XOR / A=AND / O=OR / M=最大と位置 / N=最小と位置 / I=値→index
    #       例) SegTree(N, A, "S") は和だけ / SegTree(N, A, "MI") は最大と値→indexだけ
    #       ※値は整数のみ。max/min は (値 << B | index) の整数1個で持つので |値| < 2^60 程度まで
    #       ※M, P は max/min の単位元(-無限, +無限), B は index を入れるビット数
    # 入力：N = 配列の長さ, A = 初期配列(整数のlist, len(A) == N であること), F = 使う機能(省略時は全部)
    # 出力：なし
    # 計算量：O(N)
    def __init__(self, N, A, F="SXAOMNI"):
        assert len(A) == N, "len(A) と N が違う"
        assert set(F) <= set("SXAOMNI"), "F は S X A O M N I のどれか"
        self.N, self.A, self.X, self.F, self.M, self.P = (
            N,
            list(A),
            1,
            F,
            -(1 << 100),
            1 << 100,
        )

        while self.X < N:
            self.X *= 2

        X, K, T = self.X, 2 * self.X, self.A
        self.B = B = max(1, N.bit_length())
        S, V, W, Y, U, D = (
            [0] * K if "S" in F else None,
            [0] * K if "X" in F else None,
            [-1] * K if "A" in F else None,
            [0] * K if "O" in F else None,
            [self.M] * K if "M" in F else None,
            [self.P] * K if "N" in F else None,
        )
        self.S, self.V, self.W, self.Y, self.U, self.D = S, V, W, Y, U, D
        self.H = H = {} if "I" in F else None

        if S:
            S[X : X + N] = T
        if V:
            V[X : X + N] = T
        if W:
            W[X : X + N] = T
        if Y:
            Y[X : X + N] = T

        if U or D:
            C = [(T[i] << B) | i for i in range(N)]
            if U:
                U[X : X + N] = C
            if D:
                D[X : X + N] = C

        if S:
            for i in range(X - 1, 0, -1):
                S[i] = S[i << 1] + S[(i << 1) | 1]

        if V:
            for i in range(X - 1, 0, -1):
                V[i] = V[i << 1] ^ V[(i << 1) | 1]

        if W:
            for i in range(X - 1, 0, -1):
                W[i] = W[i << 1] & W[(i << 1) | 1]

        if Y:
            for i in range(X - 1, 0, -1):
                Y[i] = Y[i << 1] | Y[(i << 1) | 1]

        if U:
            for i in range(X - 1, 0, -1):
                L, R = U[i << 1], U[(i << 1) | 1]
                U[i] = max(L, R)

        if D:
            for i in range(X - 1, 0, -1):
                L, R = D[i << 1], D[(i << 1) | 1]
                D[i] = min(L, R)

        if H is not None:
            for i in range(N):
                E = T[i]
                C = H.get(E)

                if C is None:
                    H[E] = i
                elif type(C) is int:
                    H[E] = {C, i}
                else:
                    C.add(i)

        for i, j in zip(
            "SXAOMN",
            ("querysum", "queryxor", "queryand", "queryor", "querymax", "querymin"),
        ):
            if i not in F:
                setattr(self, j, self._off(i, j))

        if "I" not in F:
            for i in ("queryindex", "queryindexmax", "queryindexall"):
                setattr(self, i, self._off("I", i))

    # 説明：無効な機能のクエリ用に、呼ぶと理由つきでエラーになる関数を作る(内部用)
    # 入力：C = 機能の文字(S X A O M N I), Q = クエリ名
    # 出力：呼ぶと RuntimeError を出す関数
    # 計算量：O(1)
    @staticmethod
    def _off(C, Q):
        def E(*_):
            raise RuntimeError(f"{Q} は無効: F に '{C}' を入れること")

        return E

    # 説明：葉の値をセットする(親の再計算はしない。内部用)
    # 入力：I = 配列のindex, X = 入れる値
    # 出力：なし
    # 計算量：O(1)
    def _set_leaf(self, I, X):
        J = I + self.X

        if self.S:
            self.S[J] = X
        if self.V:
            self.V[J] = X
        if self.W:
            self.W[J] = X
        if self.Y:
            self.Y[J] = X
        if self.U:
            self.U[J] = (X << self.B) | I
        if self.D:
            self.D[J] = (X << self.B) | I

    # 説明：葉の親から根まで再計算する(内部用)。and/or/max/min は値が変わらなければそこで打ち切る
    # 入力：I = 配列のindex
    # 出力：なし
    # 計算量：O(log N)
    def _rebuild(self, I):
        J = (I + self.X) >> 1

        if self.S:
            T, K = self.S, J

            while K:
                T[K] = T[K << 1] + T[(K << 1) | 1]
                K >>= 1

        if self.V:
            T, K = self.V, J

            while K:
                T[K] = T[K << 1] ^ T[(K << 1) | 1]
                K >>= 1

        if self.W:
            T, K = self.W, J

            while K:
                E = T[K << 1] & T[(K << 1) | 1]

                if T[K] == E:
                    break

                T[K] = E
                K >>= 1

        if self.Y:
            T, K = self.Y, J

            while K:
                E = T[K << 1] | T[(K << 1) | 1]

                if T[K] == E:
                    break

                T[K] = E
                K >>= 1

        if self.U:
            T, K = self.U, J

            while K:
                L, R = T[K << 1], T[(K << 1) | 1]

                L = max(L, R)

                if T[K] == L:
                    break

                T[K] = L
                K >>= 1

        if self.D:
            T, K = self.D, J

            while K:
                L, R = T[K << 1], T[(K << 1) | 1]

                L = min(L, R)

                if T[K] == L:
                    break

                T[K] = L
                K >>= 1

    # 説明：値→index(H)を付け替える(内部用)。1個ならint, 重複したらsetで持つ
    # 入力：I = 配列のindex, P = 古い値, Q = 新しい値
    # 出力：なし
    # 計算量：O(1)
    def _move_pos(self, I, P, Q):
        H, T = self.H, self.H[P]

        if type(T) is int:
            del H[P]
        else:
            T.discard(I)

            if not T:
                del H[P]

        T = H.get(Q)

        if T is None:
            H[Q] = I
        elif type(T) is int:
            H[Q] = {T, I}
        else:
            T.add(I)

    # 説明：1点更新  A[I] を X にする
    # 入力：I = 更新するindex, X = 新しい値
    # 出力：なし
    # 計算量：O(log N)
    def update(self, I, X):
        P = self.A[I]

        if P == X:
            return

        if self.H is not None:
            self._move_pos(I, P, X)

        self.A[I] = X
        self._set_leaf(I, X)
        self._rebuild(I)

    # 説明：交換  A[I] と A[J] を入れ替える(update を2回行う。途中の状態も常に正しい木なので安全)
    # 入力：I, J = 交換する2つのindex
    # 出力：なし
    # 計算量：O(log N)
    def switch(self, I, J):
        P, Q = self.A[I], self.A[J]

        if P == Q:
            return

        self.update(I, Q)
        self.update(J, P)

    # 説明：加算  A[I] に X を加える
    # 入力：I = 更新するindex, X = 加える値
    # 出力：なし
    # 計算量：O(log N)
    def add(self, I, X):
        self.update(I, self.A[I] + X)

    # 説明：値 X のindexを調べる(複数あれば最小)(F に "I")
    # 入力：X = 探したい値
    # 出力：index(なければ -1)
    # 計算量：O(K)  K = 値 X の個数(重複なしなら O(1))
    def queryindex(self, X):
        T = self.H.get(X)

        if T is None:
            return -1

        return T if type(T) is int else min(T)

    # 説明：値 X のindexを調べる(複数あれば最大)(F に "I")
    # 入力：X = 探したい値
    # 出力：index(なければ -1)
    # 計算量：O(K)  K = 値 X の個数(重複なしなら O(1))
    def queryindexmax(self, X):
        T = self.H.get(X)

        if T is None:
            return -1

        return T if type(T) is int else max(T)

    # 説明：値 X のindexを全部調べる(F に "I")
    # 入力：X = 探したい値
    # 出力：indexを昇順に並べたlist(なければ空list)
    # 計算量：O(K log K)  K = 値 X の個数
    def queryindexall(self, X):
        T = self.H.get(X)

        if T is None:
            return []

        return [T] if type(T) is int else sorted(T)

    # 説明：区間 [L, R) の和を求める(F に "S")
    #       ※区間クエリ共通：R > N でもエラーにならない(余白は単位元)ので index の範囲ミスに注意
    # 入力：L = 左端(含む), R = 右端(含まない)
    # 出力：和(空区間は 0)
    # 計算量：O(log N)
    def querysum(self, L, R):
        T, ans, L, R = self.S, 0, L + self.X, R + self.X

        while L < R:
            if L & 1:
                ans += T[L]
                L += 1

            if R & 1:
                R -= 1
                ans += T[R]

            L, R = L >> 1, R >> 1

        return ans

    # 説明：区間 [L, R) のXORを求める(F に "X")
    # 入力：L = 左端(含む), R = 右端(含まない)
    # 出力：XOR(空区間は 0)
    # 計算量：O(log N)
    def queryxor(self, L, R):
        T, ans, L, R = self.V, 0, L + self.X, R + self.X

        while L < R:
            if L & 1:
                ans ^= T[L]
                L += 1

            if R & 1:
                R -= 1
                ans ^= T[R]

            L, R = L >> 1, R >> 1

        return ans

    # 説明：区間 [L, R) のANDを求める(F に "A")
    # 入力：L = 左端(含む), R = 右端(含まない)
    # 出力：AND(空区間は -1 ※-1は負の数を含む全整数に対する単位元)
    # 計算量：O(log N)
    def queryand(self, L, R):
        T, ans, L, R = self.W, -1, L + self.X, R + self.X

        while L < R:
            if L & 1:
                ans &= T[L]
                L += 1

            if R & 1:
                R -= 1
                ans &= T[R]

            L, R = L >> 1, R >> 1

        return ans

    # 説明：区間 [L, R) のORを求める(F に "O")
    # 入力：L = 左端(含む), R = 右端(含まない)
    # 出力：OR(空区間は 0)
    # 計算量：O(log N)
    def queryor(self, L, R):
        T, ans, L, R = self.Y, 0, L + self.X, R + self.X

        while L < R:
            if L & 1:
                ans |= T[L]
                L += 1

            if R & 1:
                R -= 1
                ans |= T[R]

            L, R = L >> 1, R >> 1

        return ans

    # 説明：区間 [L, R) の最大値を求める(F に "M")
    # 入力：L = 左端(含む), R = 右端(含まない)
    # 出力：(最大値, index)(空区間は (-INF, -1), 同値は indexが大きい方)
    # 計算量：O(log N)
    def querymax(self, L, R):
        T, ans, L, R = self.U, self.M, L + self.X, R + self.X

        while L < R:
            if L & 1:
                E = T[L]

                ans = max(ans, E)

                L += 1

            if R & 1:
                R -= 1
                E = T[R]

                ans = max(ans, E)

            L, R = L >> 1, R >> 1

        if ans == self.M:
            return (-INF, -1)

        return (ans >> self.B, ans & ((1 << self.B) - 1))

    # 説明：区間 [L, R) の最小値を求める(F に "N")
    # 入力：L = 左端(含む), R = 右端(含まない)
    # 出力：(最小値, index)(空区間は (INF, -1), 同値は indexが小さい方)
    # 計算量：O(log N)
    def querymin(self, L, R):
        T, ans, L, R = self.D, self.P, L + self.X, R + self.X

        while L < R:
            if L & 1:
                E = T[L]

                ans = min(ans, E)

                L += 1

            if R & 1:
                R -= 1
                E = T[R]

                ans = min(ans, E)

            L, R = L >> 1, R >> 1

        if ans == self.P:
            return (INF, -1)

        return (ans >> self.B, ans & ((1 << self.B) - 1))


# ===== Go Writing =====
