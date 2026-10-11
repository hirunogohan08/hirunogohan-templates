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
import time
import operator

# ===== 設定・制限解除 =====
# fmt: off
sys.setrecursionlimit(2 * 10**6)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
input = sys.stdin.readline
def print(*args, sep=" ", end="\n"):
    sys.stdout.write(sep.join(map(str, args)) + end)
# fmt: on

# ===== 定数・リスト =====
LOW = list(string.ascii_lowercase)  # 小文字 a-z の文字列リスト
UPP = list(string.ascii_uppercase)  # 大文字 A-Z の文字列リスト
NUM = list(string.digits)  # 数字 0-9 の文字列リスト
INF = float("inf")
# MOD = 998244353
# MOD = 10**9 + 7
DIR4 = [(0, 1), (0, -1), (1, 0), (-1, 0)]
DIR8 = [(-1, 1), (0, 1), (1, 1), (-1, 0), (1, 0), (-1, -1), (0, -1), (1, -1)]
DIR9 = DIR8 + [(0, 0)]
flag, ans, cnt = False, 0, 0


# ===== セグメント木 =====
class SegTree:
    # セグ木の初期化
    # 入力：N = 配列の長さ, A = 初期配列(整数のlist, len(A) == N であること)
    # 出力：なし
    # 計算量：O(N)
    def __init__(self, N, A):
        self.N, self.A, self.X = N, list(A), 1

        while self.X < N:
            self.X *= 2

        X, K = self.X, 2 * self.X
        S, U, D, V, W, Y = (
            [0] * K,
            [(-INF, -1)] * K,
            [(INF, -1)] * K,
            [0] * K,
            [-1] * K,
            [0] * K,
        )
        self.S, self.U, self.D, self.V, self.W, self.Y = S, U, D, V, W, Y
        self.H = H = defaultdict(set)

        for i, a in enumerate(self.A):
            j = i + X
            S[j] = V[j] = W[j] = Y[j] = a
            U[j] = D[j] = (a, i)
            H[a].add(i)

        for i in range(X - 1, 0, -1):
            L, R = i * 2, i * 2 + 1
            S[i], V[i], W[i], Y[i] = S[L] + S[R], V[L] ^ V[R], W[L] & W[R], Y[L] | Y[R]
            U[i], D[i] = max(U[L], U[R]), min(D[L], D[R])

    # 葉(末端)の値をセットする(内部用。親の再計算はしない)
    # 入力：I = 配列のindex, X = 入れる値
    # 出力：なし
    # 計算量：O(1)
    def _set_leaf(self, I, X):
        j = I + self.X
        self.S[j] = self.V[j] = self.W[j] = self.Y[j] = X
        self.U[j] = self.D[j] = (X, I)

    # 子2つから親1つを再計算する(内部用)
    # 入力：I = 親ノードの番号(セグ木上の番号)
    # 出力：なし
    # 計算量：O(1)
    def _pull(self, I):
        L, R = I * 2, I * 2 + 1
        self.S[I] = self.S[L] + self.S[R]
        self.V[I] = self.V[L] ^ self.V[R]
        self.W[I] = self.W[L] & self.W[R]
        self.Y[I] = self.Y[L] | self.Y[R]
        self.U[I] = max(self.U[L], self.U[R])
        self.D[I] = min(self.D[L], self.D[R])

    # 葉の親から根まで再計算する(内部用)
    # 入力：I = 配列のindex
    # 出力：なし
    # 計算量：O(log N)
    def _rebuild(self, I):
        S, U, D, V, W, Y = self.S, self.U, self.D, self.V, self.W, self.Y
        I = (I + self.X) >> 1

        while I:
            L, R = I * 2, I * 2 + 1
            S[I], V[I], W[I], Y[I] = S[L] + S[R], V[L] ^ V[R], W[L] & W[R], Y[L] | Y[R]
            U[I], D[I] = max(U[L], U[R]), min(D[L], D[R])
            I >>= 1

    # 値→indexの対応(H)を付け替える(内部用)
    # 入力：I = 配列のindex, P = 古い値, Q = 新しい値
    # 出力：なし
    # 計算量：O(1)
    def _move_pos(self, I, P, Q):
        T = self.H[P]
        T.discard(I)

        if not T:
            del self.H[P]

        self.H[Q].add(I)

    # 1点更新  A[I] を X にする
    # 入力：I = 更新するindex, X = 新しい値
    # 出力：なし
    # 計算量：O(log N)
    def update(self, I, X):
        P = self.A[I]

        if P == X:
            return

        self._move_pos(I, P, X)
        self.A[I] = X
        self._set_leaf(I, X)
        self._rebuild(I)

    # 交換  A[I] と A[J] を入れ替える
    # 入力：I, J = 交換する2つのindex
    # 出力：なし
    # 計算量：O(log N)  ※I, J が近いほど共通の祖先が増えて速い
    def switch(self, I, J):
        P, Q = self.A[I], self.A[J]

        if I == J or P == Q:
            return

        self.A[I], self.A[J] = Q, P
        self._move_pos(I, P, Q)
        self._move_pos(J, Q, P)
        self._set_leaf(I, Q)
        self._set_leaf(J, P)

        a, b = (I + self.X) >> 1, (J + self.X) >> 1

        while a:
            self._pull(a)

            if a != b:
                self._pull(b)

            a, b = a >> 1, b >> 1

    # 加算  A[I] に X を加える
    # 入力：I = 更新するindex, X = 加える値
    # 出力：なし
    # 計算量：O(log N)
    def add(self, I, X):
        self.update(I, self.A[I] + X)

    # 値 X のindexを調べる(複数あれば最小)
    # 入力：X = 探したい値
    # 出力：index(なければ -1)
    # 計算量：O(K)  K = 値 X の個数(重複なしなら O(1))
    def queryindex(self, X):
        T = self.H.get(X)

        return min(T) if T else -1

    # 値 X のindexを調べる(複数あれば最大)
    # 入力：X = 探したい値
    # 出力：index(なければ -1)
    # 計算量：O(K)  K = 値 X の個数(重複なしなら O(1))
    def queryindexmax(self, X):
        T = self.H.get(X)

        return max(T) if T else -1

    # 値 X のindexを全部調べる
    # 入力：X = 探したい値
    # 出力：indexを昇順に並べたlist(なければ空list)
    # 計算量：O(K log K)  K = 値 X の個数
    def queryindexall(self, X):
        T = self.H.get(X)

        return sorted(T) if T else []

    # 区間 [L, R) の和を求める
    # 入力：L = 左端(含む), R = 右端(含まない)
    # 出力：和(空区間は 0)
    # 計算量：O(log N)
    def querysum(self, L, R):
        T, ans = self.S, 0
        L, R = L + self.X, R + self.X

        while L < R:
            if L & 1:
                ans += T[L]
                L += 1

            if R & 1:
                R -= 1
                ans += T[R]

            L, R = L >> 1, R >> 1

        return ans

    # 区間 [L, R) のXORを求める
    # 入力：L = 左端(含む), R = 右端(含まない)
    # 出力：XOR(空区間は 0)
    # 計算量：O(log N)
    def queryxor(self, L, R):
        T, ans = self.V, 0
        L, R = L + self.X, R + self.X

        while L < R:
            if L & 1:
                ans ^= T[L]
                L += 1

            if R & 1:
                R -= 1
                ans ^= T[R]

            L, R = L >> 1, R >> 1

        return ans

    # 区間 [L, R) のANDを求める
    # 入力：L = 左端(含む), R = 右端(含まない)
    # 出力：AND(空区間は -1 ※-1は負の数を含む全整数に対する単位元)
    # 計算量：O(log N)
    def queryand(self, L, R):
        T, ans = self.W, -1
        L, R = L + self.X, R + self.X

        while L < R:
            if L & 1:
                ans &= T[L]
                L += 1

            if R & 1:
                R -= 1
                ans &= T[R]

            L, R = L >> 1, R >> 1

        return ans

    # 区間 [L, R) のORを求める
    # 入力：L = 左端(含む), R = 右端(含まない)
    # 出力：OR(空区間は 0)
    # 計算量：O(log N)
    def queryor(self, L, R):
        T, ans = self.Y, 0
        L, R = L + self.X, R + self.X

        while L < R:
            if L & 1:
                ans |= T[L]
                L += 1

            if R & 1:
                R -= 1
                ans |= T[R]

            L, R = L >> 1, R >> 1

        return ans

    # 区間 [L, R) の最大値を求める
    # 入力：L = 左端(含む), R = 右端(含まない)
    # 出力：(最大値, index)(空区間は (-INF, -1), 同値は indexが大きい方)
    # 計算量：O(log N)
    def querymax(self, L, R):
        T, ans = self.U, (-INF, -1)
        L, R = L + self.X, R + self.X

        while L < R:
            if L & 1:
                ans = max(ans, T[L])
                L += 1

            if R & 1:
                R -= 1
                ans = max(ans, T[R])

            L, R = L >> 1, R >> 1

        return ans

    # 区間 [L, R) の最小値を求める
    # 入力：L = 左端(含む), R = 右端(含まない)
    # 出力：(最小値, index)(空区間は (INF, -1), 同値は indexが小さい方)
    # 計算量：O(log N)
    def querymin(self, L, R):
        T, ans = self.D, (INF, -1)
        L, R = L + self.X, R + self.X

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
