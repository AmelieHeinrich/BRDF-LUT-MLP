# brdf_math.py : BRDF math used for the CPU generator

import numpy as np


def radical_inverse_vdc(bits):
    bits = (bits << 16) | (bits >> 16)
    bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1)
    bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2)
    bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4)
    bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8)
    return float(bits) * 2.3283064365386963e-10


def hammersley(i, n):
    return (float(i) / float(n), radical_inverse_vdc(i))


def importance_sample_ggx(xi, N, roughness):
    a = roughness * roughness
    phi = 2.0 * np.pi * xi[0]
    cos_theta = np.sqrt((1.0 - xi[1]) / (1.0 + (a * a - 1.0) * xi[1]))
    sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)
    H = np.array([np.cos(phi) * sin_theta, np.sin(phi) * sin_theta, cos_theta])
    up = np.array([0.0, 0.0, 1.0]) if abs(N[2]) < 0.999 else np.array([1.0, 0.0, 0.0])
    tangent = np.cross(up, N)
    tangent = tangent / np.linalg.norm(tangent)
    bitangent = np.cross(N, tangent)
    sample_vec = tangent * H[0] + bitangent * H[1] + N * H[2]
    return sample_vec / np.linalg.norm(sample_vec)


def geometry_schlick_ggx(ndotv, roughness):
    a = roughness
    k = (a * a) / 2.0
    nom = ndotv
    denom = ndotv * (1.0 - k) + k
    return nom / denom


def geometry_smith(N, V, L, roughness):
    ndotv = max(N @ V, 0.0)
    ndotl = max(N @ L, 0.0)
    ggx2 = geometry_schlick_ggx(ndotv, roughness)
    ggx1 = geometry_schlick_ggx(ndotl, roughness)
    return ggx1 * ggx2
