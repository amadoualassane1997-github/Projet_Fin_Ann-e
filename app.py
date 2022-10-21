from keras.models import load_model
from PIL import Image
import numpy as np
import cv2
import os
import uuid
import flask
import urllib
import pylab # this allows you to control figure size
from flask import Flask , render_template  , request , send_file 
import matplotlib.pyplot as plt
pylab.rcParams['figure.figsize'] = (10.0, 8.0) # this controls figure size in the notebook

port = int(os.environ.get('PORT', 5000))

export_dir='age_model_pretrained.h5'
age_model = load_model(export_dir)

export_dir='gender_model_pretrained.h5'
gender_model = load_model(export_dir)

export_dir='emotion_model_pretrained.h5'
emotion_model = load_model(export_dir)

age_ranges = ['1-2', '3-9', '10-20', '21-27', '28-45', '46-65', '66-116']
gender_ranges = ['male', 'female']
emotion_ranges= ['positive','negative','neutral']


app = Flask(__name__)



ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png' , 'jfif'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT

def predict(img_path):
    path=img_path.split("\\")
    test_image = cv2.imread(img_path)
    gray = cv2.cvtColor(test_image,cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('1.4_test_input/cv2_cascade_classifier/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    myDict = {}
    pred=[]
    
    i = 0
    
    for (x,y,w,h) in faces:
      i = i+1
      cv2.rectangle(test_image,(x,y),(x+w,y+h),(203,12,255),2)
    
      img_gray=gray[y:y+h,x:x+w]
    
      emotion_img = cv2.resize(img_gray, (48, 48), interpolation = cv2.INTER_AREA)
      emotion_image_array = np.array(emotion_img)
      emotion_input = np.expand_dims(emotion_image_array, axis=0)
      output_emotion= emotion_ranges[np.argmax(emotion_model.predict(emotion_input))]
      
      gender_img = cv2.resize(img_gray, (100, 100), interpolation = cv2.INTER_AREA)
      gender_image_array = np.array(gender_img)
      gender_input = np.expand_dims(gender_image_array, axis=0)
      output_gender=gender_ranges[np.argmax(gender_model.predict(gender_input))]
    
      age_image=cv2.resize(img_gray, (200, 200), interpolation = cv2.INTER_AREA)
      age_input = age_image.reshape(-1, 200, 200, 1)
      output_age = age_ranges[np.argmax(age_model.predict(age_input))]
      pred=[output_gender,output_age,output_emotion]
      myDict[i]=pred
    
      
      col = (0,255,0)
    
      cv2.putText(test_image, str(i),(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,col,2)
    cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite('C:/Users/Amadou/Desktop/PFA/static/images/'+path[-1], test_image)
    return myDict
  
@app.route('/')
def home():
        return render_template("index.html")

@app.route('/success' , methods = ['GET' , 'POST'])
def success():
    error = ''
    target_img = os.path.join(os.getcwd() , 'static/images')
    if request.method == 'POST':
        if(request.form):
            link = request.form.get('link')
            try :
                resource = urllib.request.urlopen(link)
                unique_filename = str(uuid.uuid4())
                filename = unique_filename+".jpg"
                img_path = os.path.join(target_img , filename)
                output = open(img_path , "wb")
                output.write(resource.read())
                output.close()
                img = filename

                result =predict(img_path)


            except Exception as e : 
                print(str(e))
                error = 'This image from this site is not accesible or inappropriate input'

            if(len(error) == 0):
                return  render_template('success.html' , img  = img , predictions = result)
            else:
                return render_template('index.html' , error = error) 

            
        elif (request.files):
            file = request.files['file']
            if file and allowed_file(file.filename):
                file.save(os.path.join(target_img , file.filename))
                img_path = os.path.join(target_img , file.filename)
                img=file.filename
                
        

                result = predict(img_path)
                

               
    

            else:
                error = "Please upload images of jpg , jpeg and png extension only"

            if(len(error) == 0):
                return  render_template('success.html' ,img=img,len=len(result)  , predictions =result)
            else:
                return render_template('index.html' , error = error)

    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=port, debug=True)