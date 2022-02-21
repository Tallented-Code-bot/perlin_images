import cv2
import numpy as np
from perlin_numpy import generate_perlin_noise_2d, generate_perlin_noise_3d
import tqdm
import math


# Intitialize constants.
w, h = 1920, 1920
shape = (h, w, 2)
num_particles = 1000
iterations = 100

# Generate a 2d field of perlin noise.  The range
# is -1 to 1.
angle_field = generate_perlin_noise_2d((h, w), (16, 16)).astype(np.float64)
angle_field = (angle_field + 1) * math.pi  # convert noise to angles between 0 and 2PI

# Turn the angles into a field of vectors
field_x = np.cos(angle_field)
field_y = np.sin(angle_field)
field = np.stack((field_x, field_y), axis=2)


# Create an array of particles, one for every pixel in the image.
xarr = np.arange(w)
yarr = np.arange(h)
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
    particles = np.clip(particles + particleVelocities, 0, h - 1)

    # Add 1 to each position where there is a particle.
    brightness_to_add = np.zeros((h, w))
    brightness_to_add[transposedParticles[0], transposedParticles[1]] = 1
    brightness += brightness_to_add


# Normalize the brightness array to a range of 0-1
brightness = brightness / np.max(brightness)

# Apply a coloring function to create red, green, and blue values.
r = np.sqrt(brightness) * 180
g = np.full(brightness.shape, 20)
b = np.sqrt(brightness) * 225

# Write the rgb file to disk.
rgb = np.transpose([r, g, b], (1, 2, 0)).astype(np.float32)
cv2.imwrite("image.png", rgb)
