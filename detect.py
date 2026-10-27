import cv2
from tensorflow.keras.models import load_model

model=load_model("mask_model.keras")
model.trainable=False


face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
if face_cascade.empty():
    raise RuntimeError("Could not load Haar Cascade XML file")



def draw_label(img,text,pos,bg_color):
    text_size=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1,cv2.FILLED)

    end_x=pos[0] + text_size[0][0] + 2
    end_y=pos[1] + text_size[0][1] - 2

    cv2.rectangle(img,pos,(end_x,end_y),bg_color,cv2.FILLED)
    cv2.putText(img,text,pos,cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2,cv2.LINE_AA)



def detect(img):

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(50, 50)
    )


    for (x,y,w,h) in faces:
        face = img[y:y + h, x:x + w]

        img_resize=cv2.resize(face,(256,256))
        img_reshape=img_resize.reshape(1,256,256,3)/255.0

        y_pred=model.predict(img_reshape,verbose=0)[0][0]



        if y_pred >=0.5 :
            label=f"Mask: {y_pred*100:.2f}%"
            color=(0,255,0)
        else:
            label=f"No Mask: {(1-y_pred)*100:.2f}%"
            color=(0,0,255)



        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        draw_label(img, label, (x, y), color)

    return img
    


cap=cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open webcam. Check camera permissions or camera index."
    )

while(True):
    ret,frame=cap.read()

    if not ret:
        print("Could not read from webcam.")
        break
    
    frame=detect(frame)


    if frame is None or frame.size == 0:
        print("The processed frame is empty.")
        break


    cv2.imshow("Face Mask Detection",frame)

    
    

    if cv2.waitKey(1) & 0xFF==ord('x'):
        break

cap.release()
cv2.destroyAllWindows()
