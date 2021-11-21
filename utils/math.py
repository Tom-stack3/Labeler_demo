import math
import numpy as np

def calc_angle(a, b, c):
    """
    :param a: a[x, y]
    :param b: b[x, y]
    :param c: c[x, y]
    :return: angle between ab and bc
    """
    if b in (a, c):
        raise ValueError("Undefined angle, two identical points", (a, b, c))

    ang = math.degrees(
        math.atan2(a[1] - b[1], a[0] - b[0]) - math.atan2(c[1] - b[1], c[0] - b[0]))
    return ang + 360 if ang < 0 else ang



def calc_m_and_b(point1, point2):
    """
    calculate the slope intercept form of the line from Point 1 to Point 2.
    meaning, finding the m and b, in y=mx+b.
    :param point1: point 1
    :param point2: point 2
    :return: m, b
    """
    points = [point1, point2]
    x_coords, y_coords = zip(*points)
    A = np.vstack([x_coords, np.ones(len(x_coords))]).T

    return np.linalg.lstsq(A, y_coords, rcond=None)[0]


def y_from_m_b_x(m, b, x):
    """
    get y from y=mx+b
    :param m: slope (m)
    :param b: b
    :param x: x
    :return: y from y=mx+b
    """
    return m * x + b


def x_from_m_b_y(m, b, y):
    """
    get x from y=mx+b
    :param m: slope (m)
    :param b: b
    :param y: y
    :return: get x from y=mx+b
    """
    return (y - b) / m
