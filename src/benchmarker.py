import os
import time

import torch

from brdf_ground_truth_cpu import BRDFGroundTruthCPU
from brdf_nn_training import BRDFModel, BRDFNeuralTraining


class Benchmarks:
    def __init__(self, dimension: int = 512, spp: int = 1024):
        self.dimension = dimension
        self.spp = spp

    def benchmark_gt_generation(self) -> dict:
        print(
            f"Benchmarking GT generation ({self.dimension}x{self.dimension}, {self.spp} spp)..."
        )
        gt = BRDFGroundTruthCPU()
        start = time.perf_counter()
        gt.generate(self.dimension, self.dimension, self.spp)
        elapsed = time.perf_counter() - start
        samples = self.dimension * self.dimension
        return {
            "time_s": round(elapsed, 3),
            "samples": samples,
            "samples_per_s": round(samples / elapsed),
        }

    def benchmark_training(
        self, lr: float = 0.01, epochs: int = 100, batch_size: int = 4096
    ) -> dict:
        print(f"Benchmarking training ({epochs} epochs)...")
        model = BRDFModel()
        trainer = BRDFNeuralTraining()
        start = time.perf_counter()
        trainer.train(model, lr=lr, epochs=epochs, batch_size=batch_size)
        elapsed = time.perf_counter() - start
        return {
            "time_s": round(elapsed, 3),
            "epochs": epochs,
            "time_per_epoch_ms": round((elapsed / epochs) * 1000, 2),
        }

    def benchmark_inference(self) -> dict:
        print("Benchmarking inference...")
        model = BRDFModel()
        model.load("assets/brdf_nn.pth")
        trainer = BRDFNeuralTraining()

        # Warmup
        with torch.no_grad():
            _ = model(trainer.X)

        runs = 100
        start = time.perf_counter()
        for _ in range(runs):
            with torch.no_grad():
                _ = model(trainer.X)
        elapsed = (time.perf_counter() - start) / runs

        return {
            "time_ms": round(elapsed * 1000, 3),
            "samples": trainer.X.shape[0],
            "samples_per_s": round(trainer.X.shape[0] / elapsed),
        }

    def model_size(self) -> dict:
        path = "assets/brdf_nn.pth"
        size_bytes = os.path.getsize(path)
        lut_bytes = self.dimension * self.dimension * 2 * 4  # float32, 2 channels
        return {
            "model_size_kb": round(size_bytes / 1024, 2),
            "lut_size_kb": round(lut_bytes / 1024, 2),
            "compression_ratio": round(lut_bytes / size_bytes, 2),
        }

    def run_all(self):
        results = {}
        # Not benchmarking GT generation on CPU as it is extremely slow
        # results["gt_generation"] = self.benchmark_gt_generation()
        results["training"] = self.benchmark_training()
        results["inference"] = self.benchmark_inference()
        results["model_size"] = self.model_size()

        print("\n=== Benchmark Results ===")
        for section, metrics in results.items():
            print(f"\n[{section}]")
            for k, v in metrics.items():
                print(f"  {k}: {v}")
        return results
