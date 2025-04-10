import cv2
from keras.models import load_model
import numpy as np
facemodel=cv2.CascadeClassifier("face.xml")
drowsinessmodel = load_model("model.h5",compile=False)
vid=cv2.VideoCapture("video.mp4")
i=1
while(vid.isOpened()):
   flag,frame=vid.read()
   if(flag):
       pred=facemodel.detectMultiScale(frame)
       for (x,y,l,w) in pred:
           face_img=frame[y:y+w,x:x+l]
           face_img=cv2.resize(face_img,(224,224), interpolation=cv2.INTER_AREA)
           face_img=np.asarray(face_img, dtype=np.float32).reshape(1, 224, 224, 3)
           face_img=(face_img / 127.5) -1
           pred=drowsinessmodel.predict(face_img)[0][0]
           if(pred>0.9):
               cv2.rectangle(frame,(x,y),(x+l,y+w),(0.255,0),4)
           else:
               cv2.rectangle(frame,(x,y),(x+1,y+w),(0,0,255),4)
       cv2.namedWindow("my window",cv2.WINDOW_NORMAL)
       cv2.imshow("my window",frame)
       k=cv2.waitKey(0)
       if(k==ord('x')):
           break
   else:
       break
cv2.destroyAllWindows()