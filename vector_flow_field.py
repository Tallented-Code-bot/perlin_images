import cv2
import numpy as np
from perlin_numpy import generate_perlin_noise_2d, generate_perlin_noise_3d
import tqdm
import math


def create_image(
    filename: str,
    w: int,
    h: int,
    base_color: tuple[int, int, int],
    seed: int = None,
    iterations: int = 100,
):
    """Create an image based on perlin noise flow fields."""
    # Set numpy seed if it is specified.
    if seed is not None:
        np.random.seed(seed)

    # Generate a 2d field of perlin noise.  The range
    # is -1 to 1.
    angle_field = generate_perlin_noise_2d((h, w), (10, 10)).astype(np.float64)
    angle_field = (
        angle_field + 1
    ) * math.pi  # convert noise to angles between 0 and 2PI

    # Turn the angles into a field of vectors
    field_x = np.cos(angle_field)
    field_y = np.sin(angle_field)
    field = np.stack((field_y, field_x), axis=2)

    # Create an array of particles, one for every pixel in the image.
    xarr = np.arange(h)
    yarr = np.arange(w)
    particles = np.array(np.meshgrid(xarr, yarr)).T.reshape(-1, 2)

    # Initialize the particle velocities.
    particleVelocities = np.zeros(particles.shape)

    brightness = np.zeros((h, w))

    # Iterate for the number of iterations (tqdm is a progress bar).
    for i in tqdm.tqdm(range(iterations), desc="Loading..."):
        # Transpose the particles array, then use it as an index
        # to update the velocity with the vectors corresponding to the position of each particle.
        transposedParticles = np.floor(particles).astype(np.int64).transpose()
        particleVelocities = (
            particleVelocities + field[transposedParticles[0], transposedParticles[1]]
        )

        # Update the particle positions.
        newTransposedParticles = np.transpose(particles + particleVelocities)
        particles = np.array(
            [
                np.clip(newTransposedParticles[0], 0, h - 1),
                np.clip(newTransposedParticles[1], 0, w - 1),
            ]
        ).transpose()

        # Add 1 to each position where there is a particle.
        brightness_to_add = np.zeros((h, w))
        brightness_to_add[transposedParticles[0], transposedParticles[1]] = 1
        brightness += brightness_to_add

    # Normalize the brightness array to a range of 0-1
    brightness = brightness / np.max(brightness)

    # Apply a coloring function to create red, green, and blue values.
    r = np.sqrt(brightness) * base_color[0]
    g = np.full(brightness.shape, base_color[1])
    b = np.sqrt(brightness) * base_color[2]

    # Write the rgb file to disk.
    # rgb = np.transpose([r, g, b], (1, 2, 0)).astype(np.int32)
    rgb = np.dstack((b, g, r))

    cv2.imwrite(filename, rgb)


def create_images(
    base_filename: str,
    w: int,
    h: int,
    base_color: tuple[int, int, int],
    seed: int = None,
    frames: int = 100,
):
    """Create a series of images based on perlin noise flow fields."""
    # Set numpy seed if it is specified.
    if seed is not None:
        np.random.seed(seed)

    # Generate a 2d field of perlin noise.  The range
    # is -1 to 1.
    angle_field = generate_perlin_noise_2d((h, w), (10, 10)).astype(np.float64)
    angle_field = (
        angle_field + 1
    ) * math.pi  # convert noise to angles between 0 and 2PI

    # Turn the angles into a field of vectors
    field_x = np.cos(angle_field)
    field_y = np.sin(angle_field)
    field = np.stack((field_y, field_x), axis=2)

    # Create an array of particles, one for every pixel in the image.
    xarr = np.arange(h)
    yarr = np.arange(w)
    particles = np.array(np.meshgrid(xarr, yarr)).T.reshape(-1, 2)

    # Initialize the particle velocities.
    particleVelocities = np.zeros(particles.shape)

    brightness = np.zeros((h, w))

    # Iterate for the number of iterations (tqdm is a progress bar).
    for i in tqdm.tqdm(range(frames), desc="Loading..."):
        filename = f"{base_filename}{str(i).zfill(len(str(frames)))}.png"

        # Transpose the particles array, then use it as an index
        # to update the velocity with the vectors corresponding to the position of each particle.
        transposedParticles = np.floor(particles).astype(np.int64).transpose()
        particleVelocities = (
            particleVelocities + field[transposedParticles[0], transposedParticles[1]]
        )

        # Update the particle positions.
        newTransposedParticles = np.transpose(particles + particleVelocities)
        particles = np.array(
            [
                np.clip(newTransposedParticles[0], 0, h - 1),
                np.clip(newTransposedParticles[1], 0, w - 1),
            ]
        ).transpose()

        # Add 1 to each position where there is a particle.
        brightness_to_add = np.zeros((h, w))
        brightness_to_add[transposedParticles[0], transposedParticles[1]] = 1
        brightness += brightness_to_add

        # Normalize the brightness array to a range of 0-1
        new_brightness = brightness / np.max(brightness)

        # Apply a coloring function to create red, green, and blue values.
        r = np.sqrt(new_brightness) * base_color[0]
        g = np.full(new_brightness.shape, base_color[1])
        b = np.sqrt(new_brightness) * base_color[2]

        # Write the rgb file to disk.
        # rgb = np.transpose([r, g, b], (1, 2, 0)).astype(np.int32)
        rgb = np.dstack((r, g, b))

        cv2.imwrite(filename, rgb)
