import cv2
import numpy as np
import dlib
from math import hypot



#sayac=0
cap = cv2.VideoCapture(0)
sayac=0


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def midpoint(p1, p2):
    return int((p1.x + p2.x)/2 ), int((p1.y + p2.y)/2 )

font=cv2.FONT_HERSHEY_COMPLEX

def get_blinking(goz_hatlari,yuz_kismi):
    left_point = (yuz_kismi.part(goz_hatlari[0]).x, yuz_kismi.part(goz_hatlari[0]).y)
    right_point = (yuz_kismi.part(goz_hatlari[3]).x, yuz_kismi.part(goz_hatlari[3]).y)
    center_top = midpoint(yuz_kismi.part(goz_hatlari[1]), yuz_kismi.part(goz_hatlari[2]))
    center_bottom = midpoint(yuz_kismi.part(goz_hatlari[5]), yuz_kismi.part(goz_hatlari[4]))

   # hor_line = cv2.line(frame, left_point, right_point, (255, 0, 0), 2)
   # ver_line = cv2.line(frame, center_top, center_bottom, (255, 0, 0), 2)

    her_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
    ortalama = her_line_lenght / ver_line_lenght
    return ortalama
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        # x, y = face.left(), face.top()
        # x1, y1 = face.right(), face.bottom()
        # cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

        landmarks = predictor(gray, face)

        sol_goz_oran=get_blinking([36,37,38,39,40,41],landmarks)
        sag_goz_oran = get_blinking([42,43, 44,45, 46, 47], landmarks)
        pythondabunadasahitoldum=(sol_goz_oran+sag_goz_oran)/2

        if pythondabunadasahitoldum > 4:


         cv2.putText(frame, 'gozkirp' , (20, 100), font, 1, (255, 0, 0))



        sol_goz_bolge=np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                (landmarks.part(37).x, landmarks.part(37).y),
                                (landmarks.part(38).x, landmarks.part(38).y),
                                (landmarks.part(39).x, landmarks.part(39).y),
                                (landmarks.part(40).x, landmarks.part(40).y),
                                (landmarks.part(41).x, landmarks.part(41).y)],np.int32)
       # cv2.polylines(sol_goz_bolge,True,(0,0,255),2)

       # print(sol_goz_bolge)

        #sag_goz_bolge = np.array([(landmarks.part(42).x, landmarks.part(42).y),
                          #        (landmarks.part(43).x, landmarks.part(43).y),
                             #     (landmarks.part(44).x, landmarks.part(44).y),
                           #       (landmarks.part(45).x, landmarks.part(45).y),
                            #      (landmarks.part(46).x, landmarks.part(46).y),
                             #     (landmarks.part(47).x, landmarks.part(47).y)], np.int32)

        #cv2.polylines(frame, [sol_goz_bolge], True, (0,0,255), 2)
        # cv2.polylines(frame, [sag_goz_bolge], True, (0,0,255), 2)
        height, weight, _ = frame.shape
        mask = np.zeros((height,weight),np.uint8)
        cv2.polylines(mask, [sol_goz_bolge], True, 255 , 2)
        cv2.fillPoly(mask,[sol_goz_bolge],255)


        min_x = np.min(sol_goz_bolge[:, 0])
        max_x = np.max(sol_goz_bolge[:, 0])
        min_y = np.min(sol_goz_bolge[:, 1])
        max_y = np.max(sol_goz_bolge[:, 1])

       # sagmin_x = np.min(sol_goz_bolge[:, 0])
      #  sagmax_x = np.max(sol_goz_bolge[:, 0])
      #  solmin_y = np.min(sol_goz_bolge[:, 1])
       # solmax_y = np.max(sol_goz_bolge[:, 1])

       # eye2 = frame[solmin_y: solmax_y, sagmin_x: sagmax_x]
       # eye2 = cv2.resize(eye2, None, ix=5, iy=5)

        solgoz=frame[min_y : max_y, min_x : max_x ]
        gray_solgoz=cv2.cvtColor(solgoz,cv2.COLOR_BGR2GRAY)
        _ , sol_threshold= cv2.threshold(gray_solgoz , 70 , 255, cv2.THRESH_BINARY)

        sol_threshold = cv2.resize(sol_threshold, None, fx=5, fy=5)
        solgoz = cv2.resize(solgoz, None, fx=5, fy=5)
        cv2.imshow("eye",solgoz)
        cv2.imshow("threshold",sol_threshold)
        cv2.imshow("mask",mask)


        #cv2.imshow("eye",eye)

    cv2.imshow("Frame", frame)


    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()