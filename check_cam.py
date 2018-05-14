
""" Для проверки работоспособности камеры!!! """

import cv2
import numpy as np
import config
import face, colors

cap = config.capturing()

while True:
    ret, frame = cap.read()

    if not ret:
        print('camera error!')

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face.detect_faces(gray)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), colors.yellow, 2)


    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()