#Задание 1.
#Добавление шума

#Импорт необходимых библиотек
import cv2
import numpy as np


image = cv2.imread("variant-5.jpg")

noi = np.zeros_like(image, np.uint8)
cv2.randn(noi, 0, 2500)
noisy = cv2.addWeighted(image, 1, noi, 1000, 0)

#Уменьшение изображения, чтобы не было на весь компьютер
res_image = cv2.resize(image,(0,0), fx = 0.3, fy = 0.3)
res_noisy = cv2.resize(noisy,(0,0), fx = 0.3, fy = 0.3)

#Вывод изображений и последующее их сохранение в дереве
cv2.imshow("original", res_image)
cv2.imshow("shoom", res_noisy)
cv2.imwrite('original.jpg', res_image)
cv2.imwrite('shoom.jpg', res_noisy)
cv2.waitKey(0)
cv2.destroyAllWindows()






#Задание 2-3.
#Распечатайте изображение метки на листе бумаги и расместите его на поверхности.
#Используя камеру, захватите поверхность с меткой и реализуйте алгоритм её отслеживания.

#Измените цвет обводки метки, на синий, когда она попадает в левый верхний угол (область 50 на 50)
# и на красный, когда она попадает в правый нижний угол

# Функция, которая будет отслеживать метку на изображении и изменять цвета в зависимости от её положения
def track_marker(image):
    # Преобразование изображения в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Применение алгоритма детекции объектов,преобразование круга Хафа
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=10,
                               maxRadius=50)

    if circles is not None:
        circles = np.uint16(np.around(circles))

        # Определение центра и радиуса метки
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            radius = circle[2]

            # Проверяем положение метки и изменяем её цвет
            if center[0] < 50 and center[1] < 50:  # Левый верхний угол (x < 50, y < 50)
                color = (255, 0, 0)  # Синий цвет
            elif center[0] > image.shape[1] - 50 and center[1] > image.shape[0] - 50:  # Правый нижний угол
                color = (0, 0, 255)  # Красный цвет
            else:
                color = (200, 152, 147)

            # Рисуем круг вокруг метки с определенным цветом
            cv2.circle(image, center, radius, color, 3)

    return image


# Открываем камеру
cap = cv2.VideoCapture(0)

while (True):
    # Захватываем кадр с камеры
    ret, frame = cap.read()

    # Отслеживаем метку на кадре и изменяем её цвет в зависимости от положения
    tracked_frame = track_marker(frame)

    # Отображаем отслеженный кадр
    cv2.imshow('Tracked Marker', tracked_frame)

    # Выход из цикла
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы камеры и закрываем окна OpenCV
cap.release()
cv2.destroyAllWindows()
