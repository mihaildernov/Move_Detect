import cv2

cap = cv2.VideoCapture(0)

cap.set(3, 1280)
cap.set(4, 700)

ret, frame1 = cap.read()
ret, frame2 = cap.read()

while cap.isOpened():

    diff = cv2.absdiff(frame1, frame2)  # нахождение разницы двух кадров

    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)  # перевод кадров в черно-белую градацию

    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # фильтрация лишних контуров

    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)  # метод для выделения кромки объекта белым цветом

    dilated = cv2.dilate(thresh, None, iterations=3)  # расширяет выделенную на предыдущем этапе область

    сontours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # нахождение массива контурных точек

    for contour in сontours:
        (x, y, w, h) = cv2.boundingRect(contour)  # преобразование массива из предыдущего этапа в кортеж из четырех координат

        # площадь зафиксированного объекта
        # print(cv2.contourArea(contour))

        if cv2.contourArea(contour) < 700:  # условие, при котором площадь выделенного объекта меньше 700 px
            continue

        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)  # прорисовка прямоугольника
        cv2.putText(frame1, "Status: {}".format("Movement"), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)  # текст

    # cv2.drawContours(frame1, сontours, -1, (0, 255, 0), 2) - нарисовать контур объекта

    cv2.imshow("frame1", frame1)
    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(40) == 27:
        break

cap.release()
cv2.destroyAllWindows()
