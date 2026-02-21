# brdf_ground_truth_cpu.py : Generates the BRDF LUT on the CPU

import numpy as np
from PIL import Image

from brdf_math import geometry_smith, hammersley, importance_sample_ggx


class BRDFGroundTruthCPU:
    def __init__(self):
        pass

    def sample(self, u, v, spp):
        V = np.array([np.sqrt(1.0 - u * u), 0.0, u])
        N = np.array([0.0, 0.0, 1.0])
        A = 0.0
        B = 0.0

        for i in range(spp):
            xi = hammersley(i, spp)
            H = importance_sample_ggx(xi, N, v)
            L = 2.0 * np.dot(V, H) * H - V
            L = L / np.linalg.norm(L)
            ndotl = max(L[2], 0.0)
            ndoth = max(H[2], 0.0)
            vdoth = max(np.dot(V, H), 0.0)
            if ndotl > 0.0 and ndoth > 1e-7 and u > 1e-7:
                g = geometry_smith(N, V, L, v)
                g_vis = (g * vdoth) / (ndoth * u)
                fc = (1.0 - vdoth) ** 5.0
                A += (1.0 - fc) * g_vis
                B += fc * g_vis
        A /= spp
        B /= spp
        return (A, B)

    def generate(self, width, height, spp):
        self.width = width
        self.height = height
        self.data = np.zeros((height, width, 2), dtype=np.float32)
        for y in range(height):
            for x in range(width):
                u = x / (width - 1)
                v = y / (height - 1)
                a, b = self.sample(u, v, spp)
                self.data[y, x, 0] = a
                self.data[y, x, 1] = b
            print(f"Row {y + 1}/{height}")

    def save(self, path):
        r = (np.clip(self.data[:, :, 0], 0, 1) * 255).astype(np.uint8)
        g = (np.clip(self.data[:, :, 1], 0, 1) * 255).astype(np.uint8)
        b = np.zeros_like(r)

        img = Image.fromarray(np.stack([r, g, b], axis=-1))
        img.save(path)
