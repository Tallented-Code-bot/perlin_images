import cv2
import time
import numpy as np
from perlin_numpy import generate_perlin_noise_2d, generate_perlin_noise_3d
import matplotlib.pyplot as plt
import matplotlib
import tqdm

# from Generative_Art.examples import draw_vectors

matplotlib.use("TkAgg")

# np.random.seed(0)

w, h = 1920, 1920
# w, h = 1080, 1080

# w, h = 4000, 4000

shape = (h, w, 2)


num_particles = 1000
iterations = 1000
to_add = 1


# seed = 5234
# noise = PerlinNoise(octaves=10, seed=seed)


angle_field = generate_perlin_noise_2d((h, w), (10, 10))
angle_field = angle_field.astype(np.float64)
field_x = np.cos(angle_field)
field_y = np.sin(angle_field)
field = np.stack((field_x, field_y), axis=2)
print(field_x.shape)

x = np.arange(w)
y = np.arange(h)
X = np.meshgrid(x)
Y = np.meshgrid(y)
plt.streamplot(y, x, field_y, field_x)
plt.show()


# field = cv2.normalize(
# field, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
# )

# x = np.arange(w)
# y = np.arange(h)

# vectors = field.reshape(-1, 2).transpose()
# print(vectors)

# plt.quiver(x, y, vectors[0], vectors[1])
# plt.show()


particles = np.random.rand(num_particles, 2) * min(w, h)
particleVelocities = np.zeros((num_particles, 2))
print(particles.shape)
time.sleep(0.1)

# particle = particles + 960
# print(particles)
# input()
# print(particles)

# for i in range(iterations):
# particles += field[np.floor(particles).astype(int)]
brightness = np.zeros((h, w))


for i in tqdm.tqdm(range(iterations), desc="Loading..."):
    tParticles = np.floor(particles).astype(np.int64).transpose()
    # particlesDiff = particles - np.clip(
    # particles + field[tParticles[0], tParticles[1]], 0, h - 1
    # )
    # print(tParticles)

    # print(field[tParticles[0], tParticles[1]])
    # particleVelocities = np.clip(
    # particles + field[tParticles[0], tParticles[1]], 0, h - 1
    # )
    # print(np.max(tParticles[0]), np.max(tParticles[1]))
    particleVelocities = particleVelocities + field[tParticles[0], tParticles[1]]
    # particles = np.clip(particles + field[tParticles[0], tParticles[1]], 0, h - 1)
    particles = np.clip(particles + particleVelocities, 0, h - 1)

    # particles = np.clip(particles + field[tParticles[0], tParticles[1]], 0, h - 1)

    # print(particlesDiff[0])

    brightness_to_add = np.zeros((h, w))
    brightness_to_add[tParticles[0], tParticles[1]] = to_add
    # print(brightness_to_add)
    brightness += brightness_to_add
plt.imshow(brightness, vmin=0, vmax=255)
plt.show()


# brightness = cv2.normalize(
# brightness, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
# )


print()
print()
print()
# np.savetxt("array.txt", brightness)


# for i in range(500):
# pass


# hue = np.full(brightness.shape, 266)
# saturation = np.full(brightness.shape, 100)
brightness = np.clip(brightness, 0, 255)

print(brightness)
print(brightness.shape)

r = np.sqrt(brightness) * 180
g = np.full(brightness.shape, 20)
b = np.sqrt(brightness) * 225

rgb = np.transpose([r, g, b], (1, 2, 0)).astype(np.float32)
print(rgb)


# hsv = np.transpose([hue, saturation, brightness], (1, 2, 0)).astype(np.float32)

cv2.imwrite("image.png", rgb)
# cv2.imwrite("image.png", cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))


"""
Procedure:
- [x] Generate perlin noise grid(not normalized)
- [ ] Create 

for i in range(500):
    for each particle:
        particle+=field[floor(particle[0])][floor(particle[1])]

"""


# cv2.imwrite("image.png", field)
