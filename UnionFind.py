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


# ===== Union-Find =====
# https://github.com/hirunogohan08/hirunogohan-templates/blob/main/UnionFind.py
class UnionFind:
    # 説明：Union-Findを作る。使う機能を F で選べる(使わない機能は配列を作らず更新もしないので速く・軽い)
    #       E=辺の数 / L=最小・最大要素 / V=値の合計 / G=要素の一覧(querymembers を速くする)
    #       例) UnionFind(N, F="") は基本機能だけ(最速) / UnionFind(N, F="E") は辺の数だけ
    #       ※要素数(querysize)・グループ数・最大サイズ・same・queryall などは F に関係なく常に使える
    # 入力：N = 要素数, A = 各要素の初期値(省略時は全部0。F に "V" のときだけ使う), F = 使う機能(省略時は全部)
    # 出力：なし
    # 計算量：O(N)
    def __init__(self, N, A=None, F="ELVG"):
        assert A is None or len(A) == N, "len(A) と N が違う"
        assert set(F) <= set("ELVG"), "F は E L V G のどれか"
        self.N, self.A, self.F = N, A, F
        self._init()

        for i, j in zip(
            "ELLVVG",
            (
                "queryedges",
                "querymin",
                "querymax",
                "addvalue",
                "queryvalue",
                "querymembers",
            ),
        ):
            if i not in F:
                setattr(self, j, self._off(i, j))

        if "E" not in F:
            for i in ("istree", "hascycle"):
                setattr(self, i, self._off("E", i))

    # 説明：状態を初期化する(内部用。__init__ と reset から呼ぶ)
    # 入力：なし
    # 出力：なし
    # 計算量：O(N)
    def _init(self):
        N, F, A = self.N, self.F, self.A
        self.P, self.S = list(range(N)), [1] * N
        self.C, self.M = N, 1
        self.E = [0] * N if "E" in F else None
        self.L = list(range(N)) if "L" in F else None
        self.H = list(range(N)) if "L" in F else None
        self.V = (list(A) if A is not None else [0] * N) if "V" in F else None
        self.G = list(range(N)) if "G" in F else None

    # 説明：無効な機能のクエリ用に、呼ぶと理由つきでエラーになる関数を作る(内部用)
    # 入力：C = 機能の文字(E L V G), Q = クエリ名
    # 出力：呼ぶと RuntimeError を出す関数
    # 計算量：O(1)
    @staticmethod
    def _off(C, Q):
        def X(*_):
            raise RuntimeError(f"{Q} は無効: F に '{C}' を入れること")

        return X

    # 説明：根を探す(経路圧縮あり・反復版)
    # 入力：I = 根を探したい要素
    # 出力：Iが属するグループの根
    # 計算量：償却 O(α(N)) ≒ O(1)
    def find(self, I):
        R = I

        while self.P[R] != R:
            R = self.P[R]

        while self.P[I] != R:
            self.P[I], I = R, self.P[I]

        return R

    # 説明：同じグループか判定する
    # 入力：I, J = 判定する2つの要素
    # 出力：同じグループならTrue, 別ならFalse
    # 計算量：償却 O(α(N)) ≒ O(1)
    def same(self, I, J):
        return self.find(I) == self.find(J)

    # 説明：グループを結合する(辺 I-J を1本足すイメージ。F に "E" なら同じグループ内の辺も数える)
    # 入力：I, J = 結合する2つの要素
    # 出力：結合したならTrue, 元々同じならFalse(= この辺でサイクルができる)
    # 計算量：償却 O(α(N)) ≒ O(1)
    def union(self, I, J):
        I, J, E = self.find(I), self.find(J), self.E

        if I == J:
            if E is not None:
                E[I] += 1

            return False

        S = self.S

        if S[I] < S[J]:
            I, J = J, I

        S[I] += S[J]
        self.P[J] = I
        self.C -= 1

        self.M = max(self.M, S[I])

        if E is not None:
            E[I] += E[J] + 1

        L = self.L

        if L is not None:
            H = self.H

            L[I] = min(L[I], L[J])

            H[I] = max(H[I], H[J])

        if self.V is not None:
            self.V[I] += self.V[J]

        G = self.G

        if G is not None:
            G[I], G[J] = G[J], G[I]

        return True

    # 説明：辺のlistをまとめて結合する
    # 入力：B = [(I, J), ...] 結合する辺のlist
    # 出力：実際に結合できた(サイクルにならなかった)辺の数
    # 計算量：O(len(B) α(N))
    def unionall(self, B):
        cnt, union = 0, self.union

        for i, j in B:
            if union(i, j):
                cnt += 1

        return cnt

    # 説明：根か判定する
    # 入力：I = 判定する要素
    # 出力：根ならTrue, 根でなければFalse
    # 計算量：O(1)
    def isroot(self, I):
        return self.P[I] == I

    # 説明：全部が1グループか判定する
    # 出力：全要素が連結ならTrue
    # 計算量：O(1)
    def isconnected(self):
        return self.C == 1

    # 説明：グループの要素数を入手
    # 入力：I = 調べたい要素
    # 出力：Iが属するグループの要素数
    # 計算量：償却 O(α(N)) ≒ O(1)
    def querysize(self, I):
        return self.S[self.find(I)]

    # 説明：グループ数を入手
    # 出力：現在のグループ数
    # 計算量：O(1)
    def querygroups(self):
        return self.C

    # 説明：最大グループの要素数を入手
    # 出力：最大グループの要素数
    # 計算量：O(1)
    def querymaxsize(self):
        return self.M

    # 説明：Iが属するグループの辺の数を入手(F に "E")
    # 入力：I = 調べたい要素
    # 出力：結合時に渡された辺の数(多重辺・自己ループも数える)
    # 計算量：償却 O(α(N)) ≒ O(1)
    def queryedges(self, I):
        return self.E[self.find(I)]

    # 説明：Iが属するグループが木か判定する(辺の数 = 要素数 - 1)(F に "E")
    #       ※多重辺・自己ループがあればFalse。要素1個で辺なしは木
    # 入力：I = 調べたい要素
    # 出力：木ならTrue
    # 計算量：償却 O(α(N)) ≒ O(1)
    def istree(self, I):
        R = self.find(I)

        return self.E[R] == self.S[R] - 1

    # 説明：Iが属するグループにサイクルがあるか判定する(辺の数 >= 要素数)(F に "E")
    #       ※自己ループ・多重辺もサイクル扱い
    # 入力：I = 調べたい要素
    # 出力：サイクルがあればTrue
    # 計算量：償却 O(α(N)) ≒ O(1)
    def hascycle(self, I):
        R = self.find(I)

        return self.E[R] >= self.S[R]

    # 説明：Iが属するグループの最小要素を入手(代表元を「最小の番号」にしたいとき用)(F に "L")
    # 入力：I = 調べたい要素
    # 出力：グループ内の最小のindex
    # 計算量：償却 O(α(N)) ≒ O(1)
    def querymin(self, I):
        return self.L[self.find(I)]

    # 説明：Iが属するグループの最大要素を入手(F に "L")
    # 入力：I = 調べたい要素
    # 出力：グループ内の最大のindex
    # 計算量：償却 O(α(N)) ≒ O(1)
    def querymax(self, I):
        return self.H[self.find(I)]

    # 説明：要素Iの値に X を加える(グループの合計にも反映される)(F に "V")
    # 入力：I = 要素, X = 加える値
    # 出力：なし
    # 計算量：償却 O(α(N)) ≒ O(1)
    def addvalue(self, I, X):
        self.V[self.find(I)] += X

    # 説明：Iが属するグループの値の合計を入手(F に "V")
    # 入力：I = 調べたい要素
    # 出力：グループ内の値の合計
    # 計算量：償却 O(α(N)) ≒ O(1)
    def queryvalue(self, I):
        return self.V[self.find(I)]

    # 説明：Iが属するグループの全要素を入手(F に "G")
    #       ※小さいグループは結合時につないだ循環リストをたどる(O(K log K))。
    #         大きいグループ(要素数 > N/8)は全要素を走査する方が速いので O(N α(N)) で求める
    # 入力：I = 調べたい要素
    # 出力：Iと同じグループに属する要素のlist(昇順)
    # 計算量：O(min(K log K, N α(N)))  K = グループの要素数
    def querymembers(self, I):
        R = self.find(I)

        if self.S[R] * 8 > self.N:
            find = self.find

            return [i for i in range(self.N) if find(i) == R]

        G, A, i = self.G, [I], self.G[I]

        while i != I:
            A.append(i)
            i = G[i]

        A.sort()

        return A

    # 説明：全グループの根を入手
    # 出力：各グループの根のlist(根のindex昇順)
    # 計算量：O(N)
    def queryroots(self):
        P = self.P

        return [i for i in range(self.N) if P[i] == i]

    # 説明：全グループの要素数を入手
    # 出力：各グループの要素数のlist(最小要素の昇順 = queryall と同じ順)
    # 計算量：O(N α(N))
    def querysizes(self):
        D, A = set(), []

        for i in range(self.N):
            R = self.find(i)

            if R not in D:
                D.add(R)
                A.append(self.S[R])

        return A

    # 説明：全グループを入手
    # 出力：[[グループ1の要素], ...] 各グループ内は昇順, グループは最小要素の昇順
    # 計算量：O(N α(N))
    def queryall(self):
        D, A = {}, []

        for i in range(self.N):
            R = self.find(i)

            if R not in D:
                D[R] = len(A)
                A.append([])

            A[D[R]].append(i)

        return A

    # 説明：各要素にグループ番号(0..グループ数-1)を振る(最小要素の昇順に番号を付ける)
    #       ※連結成分を縮約して別のグラフを作るときなどに使う
    # 出力：長さ N のlist。labels[i] = 要素iのグループ番号
    # 計算量：O(N α(N))
    def querylabels(self):
        D, X = {}, [0] * self.N

        for i in range(self.N):
            R = self.find(i)

            if R not in D:
                D[R] = len(D)

            X[i] = D[R]

        return X

    # 説明：初期状態に戻す(値は最初に渡した A に戻る)
    # 計算量：O(N)
    def reset(self):
        self._init()


# ===== Go Writing =====
