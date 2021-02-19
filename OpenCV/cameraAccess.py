from cv2 import cv2
def cameraAccess():
    
    cap = cv2.VideoCapture(0)   #default id of webcam,1 for another camera
    cap.set(3,400)     #width id
    cap.set(4,400)     #height id
    cap.set(10,400)   #brightness id

    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        success, frame  = cap.read()
        cv2.imshow("video", frame)

        if cv2.waitKeyEx(1) & 0xFF == ord('q'):
            break
    
    cap.release()
cameraAccess()