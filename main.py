import streamlit as st
import cv2
from keras.models import load_model
import numpy as np
import playsound
st.set_page_config(page_title="Drowsiness Detection System",page_icon="https://thumbs.dreamstime.com/b/person-drowsiness-covid-symptom-line-style-icon-vector-illustration-design-179927281.jpg")
st.title("DROWSINESS DETECTION SYSTEM")
st.sidebar.header('Dashboard')
st.sidebar.image("https://thumbs.dreamstime.com/t/people-question-marks-vector-illustration-people-question-marks-vector-illustration-man-woman-question-thinking-164636031.jpg")
selected_option = st.sidebar.selectbox('Select an option',
                                       ['Click Here','About Drowsiness','App Features','App Uses','Feedback'])

if selected_option == ('About Drowsiness'):
   st.sidebar.write("Drowsiness is a feeling of sleepiness or fatigue that can occur due to various factors such as lack of sleep, medication side effects, certain medical conditions, or boredom. It can lead to difficulty concentrating, slower reaction times, impaired judgment, and decreased awareness of surroundings. Drowsy driving is a significant risk factor for accidents on the road, causing an estimated 100,000 crashes, 71,000 injuries, and 1,550 fatalities in the US each year. To combat drowsiness, it's important to get enough sleep, practice good sleep hygiene, and recognize the signs of drowsiness and take appropriate action. Technologies and methods such as sensors, wearable devices, and apps can help detect and prevent drowsiness in transportation contexts.")
   st.sidebar.image("https://www.fundacionmapfre.org/media/educacion-divulgacion/seguridad-vial/sistemas-adas/sistema-deteccion-fatiga-1194x585-1.jpg")

if selected_option==('App Features'):
   st.sidebar.write("A drowsiness app can be used to help detect and prevent drowsiness, particularly in transportation contexts. These apps use various sensors and algorithms to monitor the user's behavior and alert them if they appear to be getting drowsy. They can also provide recommendations for improving sleep quality, such as creating a comfortable sleep environment, establishing a consistent sleep schedule, and avoiding caffeine and alcohol before bedtime. By using a drowsiness app, users can reduce the risk of accidents on the road caused by drowsy driving and improve their overall sleep health.")   
   st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/4/4f/Possible_features_in_web_cooperation_platforms_like_wikis.png")

if selected_option==('App Uses'):
   st.sidebar.write("A drowsiness app can be used to help detect and prevent drowsiness, particularly in transportation contexts. These apps use various sensors and algorithms to monitor the user's behavior and alert them if they appear to be getting drowsy. They can also provide recommendations for improving sleep quality, such as creating a comfortable sleep environment, establishing a consistent sleep schedule, and avoiding caffeine and alcohol before bedtime. By using a drowsiness app, users can reduce the risk of accidents on the road caused by drowsy driving and improve their overall sleep health")
   st.sidebar.image("https://us.123rf.com/450wm/spicytruffel/spicytruffel2102/spicytruffel210200082/163752152-sleepy-people-drowsy-characters-in-transport-and-public-places-tired-office-workers-with.jpg?ver=6")

if selected_option == ('Feedback'):
   st.sidebar.write("Share you Opinion")
   st.sidebar.write("https://docs.google.com/forms/d/e/1FAIpQLSch-3QfM_ZwIX-HMF6Oazouqlp8vbGdT1G-pyI6kAtfIYRHig/viewform?usp=sf_link")

choice=st.selectbox("My Menu",("Home","URL","CAMERA"))
if(choice=="Home"):
   st.image("https://www.thewindscreenco.co.uk/wp-content/uploads/2020/03/06-Drowsiness-Detection-1000x477.png")
   st.markdown("<center><h1 style='color: Green;'>WELCOME</h1></center>",unsafe_allow_html=True)
   st.markdown('<hr style="border: 2px solid #000000;">', unsafe_allow_html=True)  
   st.write("<p style='text-align:center; color:Green;'>Drowsiness detection system is an advanced technology that uses machine learning algorithms to analyze driver behavior and detect signs of drowsiness or fatigue, helping to prevent accidents on the road.</p>", unsafe_allow_html=True)
elif(choice=="URL"):
   url=st.text_input("Enter Video URL Here")
   btn=st.button("Start Detection")
   window=st.empty()
   if btn:
       facemodel=cv2.CascadeClassifier("face.xml")
       drowsinessmodel = load_model("model.h5",compile=False)
       vid=cv2.VideoCapture(url)
       i=1
       btn2=st.button("Stop Detection")
       if btn2:
           vid.release()
           st.experimental_rerun()
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
                       cv2.rectangle(frame,(x,y),(x+l,y+w),(0,255,0),4)
                   else:
                       cv2.rectangle(frame,(x,y),(x+1,y+w),(0,0,255),4)
                       playsound.playsound('alarm.mp3')
               window.image(frame,channels="BGR")
   st.markdown('<hr style="border: 2px solid #000000;">', unsafe_allow_html=True)
elif(choice=="CAMERA"):
   cam=st.selectbox("Select 0 for Primary Camera and 1 For Secondary Camera",("None",0,1))
   btn=st.button("Start Detection")
   window=st.empty()
   if btn:
       facemodel=cv2.CascadeClassifier("face.xml")
       drowsinessmodel = load_model("model.h5",compile=False)
       vid=cv2.VideoCapture(cam)
       i=1
       btn2=st.button("Stop Detection")
       if btn2:
           vid.release()
           st.experimental_rerun()
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
                       cv2.rectangle(frame,(x,y),(x+l,y+w),(0,255,0),4)
                   else:
                       cv2.rectangle(frame,(x,y),(x+1,y+w),(0,0,255),4)
                       playsound.playsound('alarm.mp3')
               window.image(frame,channels="BGR")
   st.markdown('<hr style="border: 2px solid #000000;">', unsafe_allow_html=True)
st.markdown("<style>{visibility: hidden;}footer {visibility: hidden;}</style>",unsafe_allow_html=True)