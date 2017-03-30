#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""粒子フィルタのpython実装
"""

import pyximport
pyximport.install()
import resample
import numpy as xp
# import cupy as xp


class GaussianNoiseModel(object):
    """多次元ガウス分布
    """

    def __init__(self, Cov):
        """コンストラクタ

        @param ndarray(n,n) Cov : 分散共分散行列
        """
        self._Cov = Cov

    def generate(self, num):
        """ノイズの生成

        @param  int num : 粒子数
        @return ndarray(n,num) : 雑音行列
        """
        n, _ = self._Cov.shape
        return xp.random.multivariate_normal(xp.zeros(n), self._Cov, num).T

    def logpdf(self, X):
        """対数確率密度関数

        @param  ndarray(n,num) X
        @param  int num : 粒子数
        @return ndarray(num) : 対数確率密度
        """
        Cov = self._Cov
        k, _ = Cov.shape
        det_Cov = xp.linalg.det(Cov)
        Cov_inv = xp.linalg.inv(Cov)
        coefs = xp.array([ x.dot(Cov_inv).dot(x.T) for x in X.T ])
        return -k*xp.log(2.*xp.pi)/2. -xp.log(det_Cov)/2. -coefs/2.


class CauchyNoiseModel(object):
    """多次元独立コーシー分布

    各変数に独立を仮定した多次元コーシー分布
    """

    def __init__(self, gma):
        """コンストラクタ

        @param ndarray(n) gma : 尺度母数
        """
        self._gma = gma

    def generate(self, num):
        gma = self._gma
        uni = xp.random.rand(gma.size, num)
        Gma = gma.reshape(gma.size,1) * xp.ones(num)
        return xp.arctan(uni / Gma) /xp.pi + 1./2.

    def logpdf(self, X):
        _, num = X.shape
        gma = self._gma
        Gma = gma.reshape(gma.size,1) * xp.ones(num)
        return xp.sum(xp.log(Gma/xp.pi) - xp.log(X**2 + Gma**2), axis=0)


def _normalize(w):
    """重みの正規化

    @param  ndarray(num) w : 各粒子の重み
    @return ndarray(num) : 各粒子の正規化された重み
    """
    return w / xp.sum(w)


class ParticleFilter(object):
    """Particle Filter (粒子フィルタ)
    """

    def __init__(self, f, g, h, t_noise, o_noise, pars_init):
        """コンストラクタ

        @param ndarray(nx,num) function( ndarray(nx,num) ) f : 状態遷移関数
        @param ndarray(nx,num) function( ndarray(nu,num) ) g : 入力伝搬関数
        @param ndarray(ny,num) function( ndarray(nx,num) ) h : 観測関数
        @param NoiseModel t_noise : システムノイズモデル
        @param NoiseModel o_noise : 観測ノイズモデル
        @param ndarray(nx,num)  pars_init : 粒子の初期値
        """
        self._f = f
        self._g = g
        self._h = h

        self._t_noise = t_noise
        self._o_noise = o_noise

        _, num = pars_init.shape
        self._num = num
        self._w = _normalize(xp.ones(num))
        self._pars = pars_init

    def update(self, y, u):
        """フィルタ更新

        @param ndarray(ny) y : nでの観測ベクトル
        @param ndarray(nu) u : nでの入力ベクトル
        """
        self._update_pars(u)
        self._update_weights(y)
        self._resample()

    def _update_pars(self, u):
        """状態遷移モデルに沿って粒子を更新

        - 状態方程式
            x_n = f(x_n-1) + g(u_n-1) + w
        - 状態ベクトル x_n
        - 入力ベクトル u_n
        - 状態遷移関数 f(x)
        - 入力伝搬関数 g(u)
        - システムノイズ w

        @param  ndarray(nu) u : n-1での入力ベクトル (u_n-1)
        """
        U = u.reshape(u.size,1) * xp.ones(self._num)
        self._pars = self._f(self._pars) + self._g(U) + self._t_noise.generate(self._num)

    def _update_weights(self, y):
        """観測モデルに沿って尤度を計算

        - 観測方程式
            y_n = h(x_n) + v
        - 状態ベクトル x_n
        - 観測ベクトル y_n
        - 観測関数 h(x)
        - 観測ノイズ v

        @param  ndarray(ny) y : nでの観測ベクトル (y_n)
        """
        Y = y.reshape(y.size,1) * xp.ones(self._num)
        loglh = self._o_noise.logpdf( xp.absolute(Y - self._h(self._pars)) )
        self._w = _normalize( xp.exp( xp.log(self._w) + loglh ) )

    def _resample(self):
        """リサンプリング
        """
        wcum = xp.cumsum(self._w)
        num = self._num

        # idxs = [ i for n in xp.sort(xp.random.rand(num))
        #     for i in range(num) if n <= wcum[i] ]

        # start = 0
        # idxs = [ 0 for i in xrange(num) ]
        # for i, n in enumerate( xp.sort(xp.random.rand(num)) ):
        #     for j in range(start, num):
        #         if n <= wcum[j]:
        #             idxs[i] = start = j
        #             break
        idxs = resample.resample(num, wcum)

        self._pars = self._pars[:,idxs]
        self._w = _normalize(self._w[idxs])

    def estimate(self):
        """状態推定

        @return ndarray(nx) : 状態ベクトルの推定値
        """
        return xp.sum(self._pars * self._w, axis=1)

    def particles(self):
        """粒子

        @return ndarray(nx,num)
        """
        return self._pars
