import flip_evaluator as flip
import matplotlib.pyplot as plt
from PIL import Image

from benchmarker import Benchmarks


def showcase_images():
    ref = "assets/brdf_ground_truth.png"
    test = "assets/brdf_nn.png"

    flipErrorMap, meanFLIPError, _ = flip.evaluate(ref, test, "LDR")
    ref_img = Image.open(ref)
    test_img = Image.open(test)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(ref_img)
    axes[0].set_title("Ground Truth")
    axes[1].imshow(test_img)
    axes[1].set_title("Neural Output")
    axes[2].imshow(flipErrorMap)
    axes[2].set_title(f"FLIP Error (mean: {round(meanFLIPError, 6)})")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def benchmark():
    benchmark = Benchmarks()
    benchmark.run_all()


def main():
    showcase_images()


if __name__ == "__main__":
    main()
