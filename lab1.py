import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error, structural_similarity

plt.ion()

image = cv2.imread('assets/sar_1_gray.jpg')

if image is None:
    print("Ошибка: не удалось загрузить изображение!")
else:
    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image

    # Гамма-коррекцция
    def gamma_correction(image, gamma):
        image_normalized = image.astype(np.float32) / 255.0
        corrected = np.power(image_normalized, gamma)
        return (corrected * 255).astype(np.uint8)

    # Сравнение MSE и SSIM
    def compare_images(img1, img2, title):
        mse = mean_squared_error(img1, img2)
        ssim = structural_similarity(img1, img2)
        print(f"{title}: MSE = {mse:.4f}, SSIM = {ssim:.4f}")
        return mse, ssim

    gamma_low = 0.5
    image_gamma_low = gamma_correction(image_gray, gamma_low)

    gamma_high = 2.0
    image_gamma_high = gamma_correction(image_gray, gamma_high)

    plt.figure(figsize=(15, 10))

    # Исходное изображение
    plt.subplot(231)
    plt.imshow(image_gray, cmap='gray')
    plt.title('Исходное изображение')
    plt.axis('off')

    # Гамма < 1
    plt.subplot(232)
    plt.imshow(image_gamma_low, cmap='gray')
    plt.title(f'Гамма-коррекция (γ={gamma_low} < 1) - осветление')
    plt.axis('off')

    # Гамма > 1
    plt.subplot(233)
    plt.imshow(image_gamma_high, cmap='gray')
    plt.title(f'Гамма-коррекция (γ={gamma_high} > 1) - затемнение')
    plt.axis('off')

    # Гистограмма исходного
    plt.subplot(234)
    hist_orig = cv2.calcHist([image_gray], [0], None, [256], [0, 256])
    plt.plot(hist_orig, color='black')
    plt.title('Гистограмма исходного')
    plt.grid(True)

    # гистограмма гамма < 1
    plt.subplot(235)
    hist_low = cv2.calcHist([image_gamma_low], [0], None, [256], [0, 256])
    plt.plot(hist_low, color='blue')
    plt.title(f'Гистограмма (γ={gamma_low})')
    plt.grid(True)

    # гистограмма гамма > 1
    plt.subplot(236)
    hist_high = cv2.calcHist([image_gamma_high], [0], None, [256], [0, 256])
    plt.plot(hist_high, color='red')
    plt.title(f'Гистограмма (γ={gamma_high})')
    plt.grid(True)

    # Сравнение для гамма < 1
    (ssim_low, diff_low) = structural_similarity(image_gray, image_gamma_low, full=True)
    diff_low = (diff_low * 255).astype("uint8")
    mse_low = mean_squared_error(image_gray, image_gamma_low)
    print(f"Для гамма <1: MSE = {mse_low:.4f}, SSIM = {ssim_low:.4f}")

    # Сравнение для гамма > 1
    (ssim_high, diff_high) = structural_similarity(image_gray, image_gamma_high, full=True)
    diff_high = (diff_high * 255).astype("uint8")
    mse_high = mean_squared_error(image_gray, image_gamma_high)
    print(f"Для гамма >1: MSE = {mse_high:.4f}, SSIM = {ssim_high:.4f}")

    # Статистическая коррекция
    mean = image_gray.mean()
    std = image_gray.std()
    print(f"Исходное: mean={mean:.2f}, std={std:.2f}")

    eq_gray = cv2.equalizeHist(image_gray)
    mean_eq = eq_gray.mean()
    std_eq = eq_gray.std()
    print(f"Выравненное: mean={mean_eq:.2f}, std={std_eq:.2f}")

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(image_gray, cmap="gray")
    plt.title('Исходное')
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(eq_gray, cmap="gray")
    plt.title('Выравненная гистограмма')
    plt.axis('off')
    plt.show()

    # Пороговая фильтрация
    thresholds = [100, 150, 200]
    methods = [
        ('THRESH_BINARY', cv2.THRESH_BINARY),
        ('THRESH_BINARY_INV', cv2.THRESH_BINARY_INV),
        ('THRESH_TRUNC', cv2.THRESH_TRUNC),
        ('THRESH_TOZERO', cv2.THRESH_TOZERO),
        ('THRESH_TOZERO_INV', cv2.THRESH_TOZERO_INV)
    ]

    for threshold in thresholds:
        print(f"\nПорог = {threshold}:")
        plt.figure(figsize=(15, 3))

        for i, (method_name, method) in enumerate(methods, 1):
            _, thresh = cv2.threshold(image_gray, threshold, 255, method)

            above_threshold = np.sum(thresh > 0)
            total_pixels = thresh.size
            percentage = (above_threshold / total_pixels) * 100

            print(f"  {method_name}: {above_threshold} пикселей выше порога ({percentage:.1f}%)")

            plt.subplot(1, 5, i)
            plt.imshow(thresh, cmap='gray')
            plt.title(f'{method_name}\nПорог={threshold}')
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    plt.tight_layout()
    plt.show()

plt.ioff()
plt.show()