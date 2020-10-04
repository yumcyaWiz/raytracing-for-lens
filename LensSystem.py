import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Optional


# ベクトルを正規化する
def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)

# レイの方向に応じて法線を逆転させる
# v: レイの方向ベクトル
# n: 法線


def flip_normal(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    if np.dot(v, n) < 0:
        return n
    else:
        return -n


class Ray:
    # レイを表現する
    # origin: レイの始点
    # direction: レイの方向ベクトル
    def __init__(self, origin: np.ndarray, direction: np.ndarray):
        self.origin = origin
        self.direction = direction

    def __repr__(self):
        return "origin: {0}, direction: {1}".format(self.origin, self.direction)

    # レイ上の位置を計算する
    # t: 始点からの距離
    def position(self, t: float) -> np.ndarray:
        p = self.origin + t * self.direction
        return p


# 屈折ベクトルを計算する
# 全反射の場合はNoneを返す
# v: 入射ベクトル
# n: 法線
# n1: 入射側媒質の屈折率
# n2: 出射側媒質の屈折率
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
    # レンズ面を表現する
    # r=0で平面を表現する
    # r: 曲率半径
    # h: 開口半径
    # d: 次の面までの距離
    # ior: 屈折率
    def __init__(self, r: float, h: float, d: float, ior: float):
        self.z = 0
        self.r = r
        self.h = h
        self.d = d
        self.ior = ior

    # レイとの交差位置, 法線を計算する
    # 交差しない場合はNoneを返す
    # ray: レイ
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
    # レンズ系を表現する
    # filepath: csvファイルのファイルパス
    def __init__(self, filepath: str):
        # レンズデータの読み込み
        self.df = pd.read_csv(filepath)

        # レンズ面の生成
        self.lenses = []
        for i in range(len(self.df)):
            self.lenses.append(LensSurface(
                self.df.iloc[i]["r"],
                self.df.iloc[i]["h"],
                self.df.iloc[i]["d"],
                self.df.iloc[i]["ior"]
            ))

        # 各レンズ面の位置を計算
        z = 0
        for i in reversed(range(len(self.df))):
            z -= self.lenses[i].d
            self.lenses[i].z = z

    def __repr__(self):
        return str(self.df)

    # 物体側から像側に向かってレイトレーシングを行い、光線経路を返す
    # ray_in: 入射レイ
    def raytrace_from_object(self, ray_in: Ray) -> List[Ray]:
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

    # 球面収差をプロットする
    def plot_spherical_aberration(self):
        graph_x = []
        graph_y = []

        for i in range(50):
            # 入射高
            u = (i / 50)
            height = self.lenses[0].h * u
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

    # レンズ系をプロットする
    def plot(self):
        fig, ax = plt.subplots()
        for i in range(len(self.lenses)):
            lens = self.lenses[i]

            # レンズ面のプロット
            # 絞りの場合
            if lens.r == 0:
                ax.plot([lens.z, lens.z], [lens.h, 1.2*lens.h], c='blue')
                ax.plot([lens.z, lens.z], [-lens.h, -1.2*lens.h], c='blue')
            # 球面レンズの場合
            else:
                theta = abs(np.degrees(np.arcsin(lens.h / lens.r)))
                angle = 180 if lens.r > 0 else 0
                arc = patches.Arc((lens.z + lens.r, 0), 2 * abs(lens.r),
                                  2 * abs(lens.r), angle=angle, theta1=-theta, theta2=theta)
                ax.add_patch(arc)

            # レンズ枠のプロット
            if i > 0:
                lens_prev = self.lenses[i - 1]

                # 前面 or 現在の面が絞りならスキップ
                if lens.r == 0 or lens_prev.r == 0:
                    continue

                # 前面が空気ならスキップ
                if lens_prev.ior == 1:
                    continue

                def compute_l(z, r, h):
                    return r - np.sqrt(r**2 - h**2) if r > 0 else -(np.abs(r) - np.sqrt(r**2 - h**2))

                zp = lens_prev.z
                hp = lens_prev.h
                lp = compute_l(lens_prev.z, lens_prev.r, lens_prev.h)
                z = lens.z
                h = lens.h
                l = compute_l(lens.z, lens.r, lens.h)

                if lens.h > lens_prev.h:
                    ax.plot([zp + lp, z + l], [h, h], c='black')
                    ax.plot([zp + lp, z + l], [-h, -h], c='black')
                    ax.plot([zp + lp, zp + lp], [hp, h], c='black')
                    ax.plot([zp + lp, zp + lp], [-hp, -h], c='black')
                else:
                    ax.plot([zp + lp, z + l], [hp, hp], c='black')
                    ax.plot([zp + lp, z + l], [-hp, -hp], c='black')
                    ax.plot([z + l, z + l], [hp, h], c='black')
                    ax.plot([z + l, z + l], [-h, -hp], c='black')

        z_list = [lens.z for lens in self.lenses]
        length = max(z_list) - min(z_list)
        max_h = max([lens.h for lens in self.lenses])
        ax.set_xlim([min(z_list) - 0.3*length, 0.3*length])
        ax.set_ylim([-1.1*max_h, 1.1*max_h])
        ax.set_aspect('equal')
        ax.grid('on')
        plt.xlabel('$z \mathrm{[mm]}$')
        plt.xlabel('$y \mathrm{[mm]}$')

        return ax
