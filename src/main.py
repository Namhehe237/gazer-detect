import cv2
import numpy as np
import dlib
from math import hypot

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def midpoint(p1,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def get_blink_ratio(eye_points, facial_landmarks):

        left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
        right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
        center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
        center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

        hor_line = cv2.line(frame, left_point, right_point, (0,255,0),2)
        ver_line = cv2.line(frame, center_top, center_bottom, (0,255,0),2)
        
        hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
        ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
        
        # sqrt((x2 -x1)*2, (y2-y1))
        ratio = hor_line_length/ ver_line_length
        
        return ratio

def get_gaze_ratio(eye_points, facial_landmarks) :
        left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                     (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                     (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                     (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                     (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                     (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], dtype=np.int32)

        # right_eye_region = np.array([(facial_landmarks.part(eye_points[6]).x, facial_landmarks.part(eye_points[6]).y),
        #                               (facial_landmarks.part(eye_points[7]).x, facial_landmarks.part(eye_points[7]).y),
        #                               (facial_landmarks.part(eye_points[8]).x, facial_landmarks.part(eye_points[8]).y),
        #                               (facial_landmarks.part(eye_points[9]).x, facial_landmarks.part(eye_points[9]).y),
        #                               (facial_landmarks.part(eye_points[10]).x, facial_landmarks.part(eye_points[10]).y),
        #                               (facial_landmarks.part(eye_points[11]).x, facial_landmarks.part(eye_points[11]).y)], dtype=np.int32)

        # cv2.polylines(frame, [right_eye_region], True, (0,255,0), 2)
        
        height, width, _ = frame.shape
        mask = np.zeros((height,width), np.uint8)
        
        cv2.polylines(frame, [left_eye_region], True, 255, 0)   
        cv2.fillPoly(mask,[left_eye_region],255) 
        left_eye = cv2.bitwise_and(gray,gray,mask=mask)
        
        min_x = np.min(left_eye_region[:, 0])
        min_y = np.min(left_eye_region[:, 1])
        max_x = np.max(left_eye_region[:, 0])
        max_y = np.max(left_eye_region[:, 1])

        gray_eye = left_eye[min_y: max_y, min_x: max_x]
        _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
        
        height, width = threshold_eye.shape
        left_side_threshold = threshold_eye[0: height, 0 : int(width/2)]
        left_side_white = cv2.countNonZero(left_side_threshold)
        
        right_side_threshold = threshold_eye[0: height,int(width/2): width]
        right_side_white = cv2.countNonZero(right_side_threshold)
        
        
        if right_side_white == 0:
            gaze_ratio = 1  # hoặc một giá trị mặc định phù hợp
        else:
            gaze_ratio = left_side_white / right_side_white
    
        return gaze_ratio

font = cv2.FONT_HERSHEY_PLAIN

while True:
    _, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = detector(gray)
    
    for face in faces:
        x,y = face.left(), face.top()
        x1,y1 = face.right(), face.bottom()

        landmarks = predictor(gray, face)

        left_eye_ratio = get_blink_ratio([36,37,38,39,40,41], landmarks)
        right_eye_ratio = get_blink_ratio([42,43,44,45,46,47], landmarks)
        
        blinking_ratio = (left_eye_ratio + right_eye_ratio)/2
        if blinking_ratio > 5.7:
            cv2.putText(frame, "BLINKING",(50,150),font ,7,(255,0,0))
    
        gaze_ratio_left_eye = get_gaze_ratio([36,37,38,39,40,41], landmarks)
        gaze_ratio_right_eye = get_gaze_ratio([42,43,44,45,46,47], landmarks)
        gaze_ratio = (gaze_ratio_left_eye + gaze_ratio_right_eye) / 2
        if gaze_ratio <= 0.9 :
            cv2.putText(frame, "LEFT",(50,50),font ,2,(0,0,255),3)
        elif 0.9 < gaze_ratio < 1.7 :
            cv2.putText(frame, "CENTER",(50,50),font ,2,(0,0,255),3)
        else :
            cv2.putText(frame, "RIGHT",(50,50),font ,2,(0,0,255),3)
            
        # cv2.putText(frame, str(gaze_ratio_left_eye), (50,100), font, 2, (0,0,255), 3)
        # cv2.putText(frame, str(gaze_ratio_right_eye), (50,200), font, 2, (0,0,255), 3)

    cv2.imshow("VIDEO", frame)
    
    key = cv2.waitKey(1)
    
    if key == 27: break
    
cap.release()
cv2.destroyAllWindows()