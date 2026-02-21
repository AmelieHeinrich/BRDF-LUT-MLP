import pygame

from brdf_ground_truth_cpu import BRDFGroundTruthCPU


def main():
    pygame.init()
    width, height = 512, 512
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("LUT Preview")

    image = pygame.image.load("models/brdf_ground_truth.png")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        lut_scaled = pygame.transform.scale(image, (width, height))
        screen.blit(lut_scaled, (0, 0))
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
