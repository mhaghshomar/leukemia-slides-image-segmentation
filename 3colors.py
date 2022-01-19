import numpy as np
import cv2
from math import log
import random


colors = np.arange(9).reshape(3, 3)
colors[0][0] = random.randint(0, 255)
colors[0][1] = random.randint(0, 255)
colors[0][2] = random.randint(0, 255)
colors[1][0] = random.randint(0, 255)
colors[1][1] = random.randint(0, 255)
colors[1][2] = random.randint(0, 255)
colors[2][0] = random.randint(0, 255)
colors[2][1] = random.randint(0, 255)
colors[2][2] = random.randint(0, 255)

totalSteps = 20


def getNeighbours(image, colors, step):
    neighbours = []
    fitnesses = []
    y = 25 - totalSteps
    for x in range(10):
        while True:
            a, b, c, d, e, f, g, h, i = random.randint(-y, y), random.randint(-y, y), random.randint(-y, y), \
                                        random.randint(-y, y), random.randint(-y, y), random.randint(-y, y), \
                                        random.randint(-y, y), random.randint(-y, y), random.randint(-y, y)
            if (0 <= colors[0][0] + a <= 255) and (0 <= colors[0][1] + b <= 255) and (0 <= colors[0][2] + c <= 255)\
                    and (0 <= colors[1][0] + d <= 255) and (0 <= colors[1][1] + e <= 255) and (0 <= colors[1][2] + f <= 255) \
                    and (0 <= colors[2][0] + g <= 255) and (0 <= colors[2][1] + h <= 255) and (0 <= colors[2][2] + i <= 255):
                colors[0][0] += a
                colors[0][1] += b
                colors[0][2] += c
                colors[1][0] += d
                colors[1][1] += e
                colors[1][2] += f
                colors[2][0] += g
                colors[2][1] += h
                colors[2][2] += i
                break
        neighbours.append(colors)
        print("currentColors:\n",  colors)
        fitnesses.append(fitness(image, colors))


    return neighbours, fitnesses


def getBestNeighbour(neighbours, fitnesses):
    best = 10000000
    for i in fitnesses:
        if i < best:
            best = i
            position = fitnesses.index(i)
    return neighbours[position], fitnesses[position]


def hillClimbing(image, colors):
    step = 0
    bestFinalFitness = 10000000
    while step < totalSteps:
        neighbours, fitnesses = getNeighbours(image, colors, step)
        bestNeighbour, bestFitness = getBestNeighbour(neighbours, fitnesses)
        if bestFitness < bestFinalFitness:
            bestFinalFitness = bestFitness
            x = np.copy(bestNeighbour)
            bestFinalColor = x
            print("step: ", step, "***", "bestFitness: ", bestFitness, "***", "bestNeighbour:\n", bestNeighbour)
            print("******************************************************************************")
        step += 1
    return (bestFinalFitness, bestFinalColor)


def fitness(image, colors):
    assign_func = lambda pixel: min(np.linalg.norm(pixel / 255 - color / 255) for color in colors)
    distances = np.array(list(map(assign_func, image)))
    return np.sum(distances)/100


def reconstruct_image(image, bestcolor):
    size = image.shape[0]
    target = image.reshape(-1, image.shape[-1])
    assign_func = lambda pixel: bestcolor[np.argmin([np.linalg.norm(pixel - color) for color in bestcolor])]
    result = np.array(list(map(assign_func, target)), dtype=np.uint8)
    return result.reshape(size, size, 3)


def PSNR(original, reconstructed):
    MSE = np.mean((original - reconstructed) ** 2)
    if(MSE == 0):
        return 100
    psnr = 10*log(255*255/MSE)
    return psnr


img = cv2.imread('Im047_1.tif')
resized_img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
linear_image = resized_img.reshape(-1, img.shape[-1])


bestFinalFitness, bestFinalColor = hillClimbing(linear_image, colors)
print("\n\nbestFinalFitness: ", bestFinalFitness, "***", "bestFinalColor:\n", bestFinalColor)

reconstructed = reconstruct_image(img, bestFinalColor)

psnr = PSNR(img, reconstructed)
print("\n\n psnr: ", psnr)

cv2.imshow("maryam", reconstructed)
cv2.waitKey(0)


