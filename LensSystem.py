import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def flip_normal(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    if np.dot(v, n) < 0:
        return n
    else:
        return -n


class Ray:
    def __init__(self, origin: np.ndarray, direction: np.ndarray):
        self.origin = origin
        self.direction = direction

    def __repr__(self):
        return "origin: {0}, direction: {1}".format(self.origin, self.direction)

    def position(self, t: float) -> np.ndarray:
        p = self.origin + t * self.direction
        return p


def refract(v: np.ndarray, n: np.ndarray, n1: float, n2: float) -> Optional[np.ndarray]:
    # 屈折ベクトルの水平方向
    t_h = -n1 / n2 * (v - np.dot(v, n)*n)

    # 全反射
    if np.linalg.norm(t_h) > 1:
        return None

    # 屈折ベクトルの垂直方向
    t_p = -np.sqrt(1 - np.linalg.norm(t_h)**2) * n

    # 屈折ベクトル
    t = t_h + t_p

    return t


class LensSurface:
    def __init__(self, r: float, h: float, d: float, ior: float):
        self.z = 0
        self.r = r
        self.h = h
        self.d = d
        self.ior = ior

    def intersect(self, ray: Ray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self.r != 0:
            # 球面との交差

            # レンズの中心位置
            center = np.array([0, 0, self.z + self.r])

            # 判別式
            b = np.dot(ray.direction, ray.origin - center)
            c = np.linalg.norm(ray.origin - center)**2 - self.r**2
            D = b**2 - c

            # D < 0の場合は交差しない
            if D < 0:
                return None, None

            # tの候補
            t_1 = -b - np.sqrt(D)
            t_2 = -b + np.sqrt(D)

            # 適切なtを選択
            t = None
            if ray.direction[2] > 0 and self.r > 0:
                t = t_1
            elif ray.direction[2] > 0 and self.r < 0:
                t = t_2
            elif ray.direction[2] < 0 and self.r < 0:
                t = t_1
            else:
                t = t_2

            # 交差位置
            p = ray.position(t)

            # 交差位置が開口半径以上なら交差しない
            if p[0] ** 2 + p[1] ** 2 > self.h ** 2:
                return None, None

            # 法線
            n = flip_normal(ray.direction, normalize(p - center))

            return p, n
        else:
            # 平面との交差

            # 交差位置
            t = -(ray.origin[2] - self.z) / ray.direction[2]
            p = ray.position(t)

            # 交差位置が開口半径以上なら交差しない
            if p[0] ** 2 + p[1] ** 2 > self.h ** 2:
                return None, None

            # 法線
            n = flip_normal(ray.direction, np.array([0, 0, -1]))

            return p, n


class LensSystem:
    def __init__(self, filepath: str):
        # レンズデータの読み込み
        self.df = df = pd.read_csv(filepath)

        # レンズ面の生成
        self.lenses = []
        for i in range(len(df)):
            self.lenses.append(LensSurface(
                df.iloc[i]["r"],
                df.iloc[i]["h"],
                df.iloc[i]["d"],
                df.iloc[i]["ior"]
            ))

        # 各レンズ面の位置を計算
        z = 0
        for i in reversed(range(len(df))):
            z -= self.lenses[i].d
            self.lenses[i].z = z

    def __repr__(self):
        return str(self.df)

    def raytrace_from_object(self, ray_in: Ray) -> Ray:
        n1 = 1
        ray = ray_in
        rays = [ray]
        for i in range(len(self.lenses)):
            lens = self.lenses[i]

            # レンズとの交差位置, 法線を計算
            p, n = lens.intersect(ray)
            if p is None or n is None:
                return None

            # 屈折方向を計算
            n2 = lens.ior
            t = refract(-ray.direction, n, n1, n2)
            if t is None:
                return None

            # レイを更新
            ray = Ray(p, t)
            rays.append(ray)

            # 屈折率を更新
            n1 = n2

        return rays

    def plot_spherical_aberration(self):
        graph_x = []
        graph_y = []

        for i in range(50):
            # 入射高
            u = (i / 50)
            height = 0.5 * self.lenses[0].h * u
            graph_y.append(height)

            # レイトレ
            rays = self.raytrace_from_object(Ray(
                np.array([0, height, -1000]),
                np.array([0, 0, 1])
            ))

            # 像面との交点
            t = -rays[-1].origin[2] / rays[-1].direction[2]
            p = rays[-1].position(t)
            graph_x.append(p[1])

        ax = plt.plot(graph_x, graph_y)
        plt.grid()
        plt.title('Spherical Aberration')
        plt.xlabel('$y\mathrm{[mm]}$')
        plt.ylabel('Height$\mathrm{[mm]}$')

        return ax
