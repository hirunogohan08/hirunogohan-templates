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


# ===== 遅延セグメント木 =====
class LazySegTree:
    # 遅延セグ木の初期化
    # 入力: N = 配列の長さ, A = 初期配列(list)
    # 計算量: O(N)
    def __init__(self, N, A):
        self.N, self.A = N, list(A)

        self.X = 1
        while self.X < N:
            self.X *= 2

        K = 2 * self.X
        self.S = [0] * K
        self.U = [(-INF, -1)] * K
        self.D = [(INF, -1)] * K
        self.E = [0] * K
        self.F = [None] * K
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

        self.S[I] = X
        self.U[I] = (X, I - self.X)
        self.D[I] = (X, I - self.X)

        self.E[I] = 0
        self.F[I] = None

    # 子2つから親1つを再計算する(内部用)
    # 入力: I = 親ノードの番号
    # 計算量: O(1)
    def _pull(self, I):
        L, R = I * 2, I * 2 + 1

        self.S[I] = self.S[L] + self.S[R]
        self.U[I] = max(self.U[L], self.U[R])
        self.D[I] = min(self.D[L], self.D[R])

    # ノードに「代入」を適用する(内部用)
    # 入力: I = ノード番号, X = 代入する値, P, Q = このノードが担当する実区間 [P, Q)
    # 計算量: O(1)
    def _apply_set(self, I, X, P, Q):
        K = Q - P

        self.S[I] = X * K
        self.U[I] = (X, Q - 1)
        self.D[I] = (X, P)

        self.F[I] = X
        self.E[I] = 0

    # ノードに「加算」を適用する(内部用)
    # 入力: I = ノード番号, X = 加算する値, L = 区間長
    # 計算量: O(1)
    def _apply_add(self, I, X, L):
        self.S[I] += X * L

        self.U[I] = (self.U[I][0] + X, self.U[I][1])

        self.D[I] = (self.D[I][0] + X, self.D[I][1])

        if self.F[I] is not None:
            self.F[I] += X
        else:
            self.E[I] += X

    # 遅延を子へ伝える(内部用)
    # 入力: I = ノード番号, P, Q = Iが担当する実区間 [P, Q)
    # 計算量: O(1)
    def _push(self, I, P, Q):
        if I >= self.X:
            return

        L, R = I * 2, I * 2 + 1
        M = (P + Q) // 2

        if self.F[I] is not None:
            X = self.F[I]

            self._apply_set(L, X, P, M)
            self._apply_set(R, X, M, Q)

            self.F[I] = None

        if self.E[I] != 0:
            X = self.E[I]

            self._apply_add(L, X, M - P)
            self._apply_add(R, X, Q - M)

            self.E[I] = 0

    # 葉から根まで再計算する(内部用)
    # 入力: I = 配列のindex
    # 計算量: O(log N)
    def _rebuild(self, I):
        I = (I + self.X) // 2

        while I >= 1:
            self._pull(I)
            I //= 2

    # 指定した葉まで遅延を伝播する(内部用)
    # 入力: I = 配列のindex
    # 計算量: O(log N)
    def _push_path(self, I):
        O, L, R = 1, 0, self.X

        while R - L > 1:
            self._push(O, L, R)

            M = (L + R) // 2

            if I < M:
                O = O * 2
                R = M
            else:
                O = O * 2 + 1
                L = M

    # 値 -> indexの対応を付け替える(内部用)
    # 入力: I = 配列のindex, P = 古い値, Q = 新しい値
    # 計算量: O(1)
    def _move_pos(self, I, P, Q):
        self.H[P].discard(I)

        if not self.H[P]:
            del self.H[P]

        self.H.setdefault(Q, set()).add(I)

    # セグ木の更新(1点)
    # A[I] を X にする
    # 計算量: O(log N)
    def update(self, I, X):
        P = self.A[I]

        if P == X:
            return

        self._push_path(I)

        self._move_pos(I, P, X)
        self.A[I] = X

        self._set_leaf(I, X)
        self._rebuild(I)

    # セグ木の更新(交換)
    # A[I] と A[J] を入れ替える
    # 計算量: O(log N)
    def switch(self, I, J):
        P, Q = self.A[I], self.A[J]

        if I == J or P == Q:
            return

        self._push_path(I)

        if J != I:
            self._push_path(J)

        self.A[I], self.A[J] = Q, P

        self._move_pos(I, P, Q)
        self._move_pos(J, Q, P)

        self._set_leaf(I, Q)
        self._set_leaf(J, P)

        self._rebuild(I)
        self._rebuild(J)

    # 区間更新(加算)
    # [L, R) の全要素に X を加える
    # セグ木部分: O(log N)
    # A/Hの同期処理: O(R-L) (index逆引き用のHを維持するために必要なコスト。queryindexを使わないならA/Hの維持自体を削るとO(log N)化できる)
    def add(self, L, R, X):
        if L >= R or X == 0:
            return

        self._range_add(1, 0, self.X, L, R, X)

        for i in range(L, R):
            P = self.A[i]
            Q = P + X

            self._move_pos(i, P, Q)
            self.A[i] = Q

    # 区間加算(内部用)
    def _range_add(self, I, P, Q, L, R, X):
        if R <= P or Q <= L:
            return

        if L <= P and Q <= R:
            self._apply_add(I, X, Q - P)
            return

        self._push(I, P, Q)

        M = (P + Q) // 2

        self._range_add(I * 2, P, M, L, R, X)
        self._range_add(I * 2 + 1, M, Q, L, R, X)

        self._pull(I)

    # 区間更新(代入)
    # [L, R) の全要素を X にする
    # セグ木部分: O(log N)
    # A/Hの同期処理: O(R-L) (addと同様の理由)
    def set(self, L, R, X):
        if L >= R:
            return

        self._range_set(1, 0, self.X, L, R, X)

        for i in range(L, R):
            P = self.A[i]

            if P == X:
                continue

            self._move_pos(i, P, X)
            self.A[i] = X

    # 区間代入(内部用)
    def _range_set(self, I, P, Q, L, R, X):
        if R <= P or Q <= L:
            return

        if L <= P and Q <= R:
            self._apply_set(I, X, P, Q)
            return

        self._push(I, P, Q)

        M = (P + Q) // 2

        self._range_set(I * 2, P, M, L, R, X)
        self._range_set(I * 2 + 1, M, Q, L, R, X)

        self._pull(I)

    # indexを入手
    # 値 X のindexを返す
    # なければ -1、複数あれば最小
    # 計算量: O(|H[X]|)
    def queryindex(self, X):
        T = self.H.get(X)

        return min(T) if T else -1

    # 合計値を入手
    # 区間 [L, R) の和を返す
    # 計算量: O(log N)
    def querysum(self, L, R):
        return self._querysum(1, 0, self.X, L, R)

    def _querysum(self, I, P, Q, L, R):
        if R <= P or Q <= L:
            return 0

        if L <= P and Q <= R:
            return self.S[I]

        self._push(I, P, Q)

        M = (P + Q) // 2

        return self._querysum(I * 2, P, M, L, R) + self._querysum(I * 2 + 1, M, Q, L, R)

    # 最大値を入手
    # 区間 [L, R) の (最大値, index) を返す
    # 空区間は (-INF, -1)
    # 同値はindexが大きい方
    # 計算量: O(log N)
    def querymax(self, L, R):
        return self._querymax(1, 0, self.X, L, R)

    def _querymax(self, I, P, Q, L, R):
        if R <= P or Q <= L:
            return (-INF, -1)

        if L <= P and Q <= R:
            return self.U[I]

        self._push(I, P, Q)

        M = (P + Q) // 2

        return max(
            self._querymax(I * 2, P, M, L, R), self._querymax(I * 2 + 1, M, Q, L, R)
        )

    # 最小値を入手
    # 区間 [L, R) の (最小値, index) を返す
    # 空区間は (INF, -1)
    # 同値はindexが小さい方
    # 計算量: O(log N)
    def querymin(self, L, R):
        return self._querymin(1, 0, self.X, L, R)

    def _querymin(self, I, P, Q, L, R):
        if R <= P or Q <= L:
            return (INF, -1)

        if L <= P and Q <= R:
            return self.D[I]

        self._push(I, P, Q)

        M = (P + Q) // 2

        return min(
            self._querymin(I * 2, P, M, L, R), self._querymin(I * 2 + 1, M, Q, L, R)
        )


# ===== Go Writing =====
