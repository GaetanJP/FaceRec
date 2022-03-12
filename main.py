import cv2


trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#yeji_img = cv2.imread('yeji.jpg')
#chae_img = cv2.imread('chae.jpg')

webcam = cv2.VideoCapture(0)

while True:
    successful_frame_rad, frame = webcam.read()

    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coord = trained_face_data.detectMultiScale(grayscaled_img)
    for (x,y,w,h) in face_coord:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0))
    cv2.imshow("Face detector", frame)
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break




