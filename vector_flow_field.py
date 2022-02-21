import cv2
import time
import numpy as np
from perlin_numpy import generate_perlin_noise_2d, generate_perlin_noise_3d
import matplotlib.pyplot as plt
import matplotlib
import tqdm
import math


matplotlib.use("TkAgg")


w, h = 1920, 1920

shape = (h, w, 2)


num_particles = 1000
iterations = 100
to_add = 1


angle_field = generate_perlin_noise_2d((h, w), (16, 16)).astype(np.float64)
angle_field = (angle_field + 1) * math.pi
field_x = np.cos(angle_field) * 0.05
field_y = np.sin(angle_field) * 0.05
field = np.stack((field_x, field_y), axis=2)
print(field_x.shape)


x = np.arange(w)
y = np.arange(h)
X = np.meshgrid(x)
Y = np.meshgrid(y)
plt.streamplot(y, x, field_y, field_x)
plt.show()


xarr = np.arange(w)
yarr = np.arange(h)
particles = np.array(np.meshgrid(xarr, yarr)).T.reshape(-1, 2)


particleVelocities = np.zeros(particles.shape)
print(particles.shape)
time.sleep(0.1)

brightness = np.zeros((h, w))


for i in tqdm.tqdm(range(iterations), desc="Loading..."):
    tParticles = np.floor(particles).astype(np.int64).transpose()
    particleVelocities = particleVelocities + field[tParticles[0], tParticles[1]]
    particles = np.clip(particles + particleVelocities, 0, h - 1)

    brightness_to_add = np.zeros((h, w))
    brightness_to_add[tParticles[0], tParticles[1]] = to_add
    brightness += brightness_to_add


print("max is ", np.max(brightness))
brightness = brightness / np.max(brightness)  # Normalize it to a range of 0-1

r = np.sqrt(brightness) * 180
g = np.full(brightness.shape, 20)
b = np.sqrt(brightness) * 225


rgb = np.transpose([r, g, b], (1, 2, 0)).astype(np.float32)


cv2.imwrite("image.png", rgb)


"""
Procedure:
- [x] Generate perlin noise grid(not normalized)
- [ ] Create 

for i in range(500):
    for each particle:
        particle+=field[floor(particle[0])][floor(particle[1])]

"""


# cv2.imwrite("image.png", field)
