import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy
import math
from skimage.metrics import structural_similarity, mean_squared_error

image = cv2.imread('assets/sar_1.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.ion()

plt.figure(1)
plt.imshow(image_gray, cmap="gray")
plt.title('Исходное изображение')
plt.show()

# Шум гаусса
mean = 0
stddev = 100
noise_gauss = np.zeros(image_gray.shape, np.uint8)
cv2.randn(noise_gauss, mean, stddev)

plt.figure(2)
plt.imshow(noise_gauss, cmap="gray")
plt.title('Гауссовский шум')
plt.show()

# Шум соль и перец
noise = np.random.randint(0, 101, size=(image_gray.shape[0], image_gray.shape[1]), dtype=int)
zeros_pixel = np.where(noise == 0)
ones_pixel = np.where(noise == 100)

bg_image = np.ones(image_gray.shape, np.uint8) * 128
bg_image[zeros_pixel] = 0
bg_image[ones_pixel] = 255

plt.figure(3)
plt.imshow(bg_image, cmap="gray")
plt.title('Шум "соль и перец"')
plt.show()

# Шум гаусса на картинке
image_noise_gauss = cv2.add(image_gray, noise_gauss)
plt.figure(4)
plt.imshow(image_noise_gauss, cmap="gray")
plt.title('Изображение с гауссовским шумом')
plt.show()

mse_gauss = mean_squared_error(image_gray, image_noise_gauss)
(ssim, diff) = structural_similarity(image_gray, image_noise_gauss, full=True)
print(f"Гауссовский шум: MSE = {mse_gauss:.4f}, SSIM = {ssim:.4f}")

# Шум соль и перец на картинке
image_sp = copy.deepcopy(image_gray)
image_sp[zeros_pixel] = 0
image_sp[ones_pixel] = 255

plt.figure(6)
plt.imshow(image_sp, cmap="gray")
plt.title('Изображение с шумом "соль и перец"')
plt.show()

mse_sp = mean_squared_error(image_gray, image_sp)
(ssim_sp, diff) = structural_similarity(image_gray, image_sp, full=True)
print(f"Шум соль и перец: MSE = {mse_sp:.4f}, SSIM = {ssim_sp:.4f}")

# Фильтр Гаусса
image_gauss_gauss = cv2.GaussianBlur(image_noise_gauss, (5,5), 0)
plt.figure(8)
plt.imshow(image_gauss_gauss, cmap="gray")
plt.title('Фильтр Гаусса 5x5')
plt.show()

mse_gauss_gauss = mean_squared_error(image_gray, image_gauss_gauss)
ssim_gauss_gauss = structural_similarity(image_gray, image_gauss_gauss)
print(f"Фильтр Гаусса: MSE = {mse_gauss_gauss:.4f}, SSIM = {ssim_gauss_gauss:.4f}")

# Билатеральный фильтр
image_gauss_bilat = cv2.bilateralFilter(image_noise_gauss, 9, 75, 75)
plt.figure(9)
plt.imshow(image_gauss_bilat, cmap="gray")
plt.title('Билатеральный фильтр')
plt.show()

mse_bilat = mean_squared_error(image_gray, image_gauss_bilat)
ssim_bilat = structural_similarity(image_gray, image_gauss_bilat)
print(f"Билатеральный фильтр: MSE = {mse_bilat:.4f}, SSIM = {ssim_bilat:.4f}")

# Фильтр средних
image_gauss_nlm = cv2.fastNlMeansDenoising(image_noise_gauss, h=20)
plt.figure(10)
plt.imshow(image_gauss_nlm, cmap="gray")
plt.title('Фильтр средних (h=20)')
plt.show()

mse_nlm = mean_squared_error(image_gray, image_gauss_nlm)
ssim_nlm = structural_similarity(image_gray, image_gauss_nlm)
print(f"NLM h=20: MSE = {mse_nlm:.4f}, SSIM = {ssim_nlm:.4f}")

# Геометрический фильтр
def geom(a):
    prod = 1
    for i in range(a.shape[0]):
        prod1 = 1
        for j in range(a.shape[1]):
            prod1 *= a[i,j]
        prod1 = math.pow(prod1, 1.0/9.0)
        prod *= prod1
    return prod

def proc(img, filter):
    img_res = copy.deepcopy(img)
    for i in range(0, img.shape[0] - 2):
        for j in range(0, img.shape[1] - 2):
            img_res[i:i+3, j:j+3] = filter(img[i:i+3, j:j+3])
    return img_res

res = proc(image_noise_gauss, geom)

plt.figure(11)
plt.imshow(res, cmap="gray")
plt.title('Геометрический фильтр')
plt.show()

mse_geom = mean_squared_error(image_gray, res)
ssim_geom = structural_similarity(image_gray, res)
print(f"Геометрический фильтр: MSE = {mse_geom:.4f}, SSIM = {ssim_geom:.4f}")

# 2D свертка, averaging filter
kernel_5 = np.ones((5,5), np.float32)/25
image_k5 = cv2.filter2D(image_gray, -1, kernel_5)
image_b5 = cv2.blur(image_gray, (5,5))

plt.figure(12, figsize=(10, 5))
plt.subplot(121)
plt.imshow(image_k5, cmap="gray")
plt.title('averaging filter')
plt.subplot(122)
plt.imshow(image_b5, cmap="gray")
plt.title('cv2.blur')
plt.show()

mse_kb = mean_squared_error(image_k5, image_b5)
ssim_kb = structural_similarity(image_k5, image_b5)
print(f"Сравнение: MSE = {mse_kb:.4f}, SSIM = {ssim_kb:.4f}")

# Laplasian
kernel_lapl = np.array([[0,-10,0],
                        [-10,40,-10],
                        [0,-10,0]], np.float32)

image_lapl = cv2.filter2D(image_gray, -1, kernel_lapl)

plt.figure(13)
plt.imshow(image_lapl, cmap="gray")
plt.title('Laplasian фильтр')
plt.show()

# Сравнение фильтров
print("\nСРАВНЕНИЕ ФИЛЬТРАЦИЙ ШУМА:")

results = [
    ("Исходный шум", mse_gauss, ssim),
    ("Фильтр Гаусса 5x5", mse_gauss_gauss, ssim_gauss_gauss),
    ("Билатеральный фильтр", mse_bilat, ssim_bilat),
    ("NLM h=20", mse_nlm, ssim_nlm),
    ("Геометрический фильтр", mse_geom, ssim_geom)
]

results_sorted_ssim = sorted(results, key=lambda x: x[2], reverse=True)
results_sorted_mse = sorted(results, key=lambda x: x[1])

best_filter_ssim = results_sorted_ssim[0]
best_filter_mse = results_sorted_mse[0]

print(f"\nЛУЧШИЙ ФИЛЬТР ПО SSIM: {best_filter_ssim[0]} (SSIM = {best_filter_ssim[2]:.4f})")
print(f"ЛУЧШИЙ ФИЛЬТР ПО MSE: {best_filter_mse[0]} (MSE = {best_filter_mse[1]:.4f})")


plt.ioff()
plt.show()