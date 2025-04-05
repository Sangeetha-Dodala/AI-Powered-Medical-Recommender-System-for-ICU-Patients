from flask import Flask, render_template, request, flash, redirect
##from tensorflow.keras.models import load_model
import sqlite3
import pickle
import numpy as np
import random
import requests
import warnings
import google.generativeai as genai
import nltk
from nltk.stem.lancaster import LancasterStemmer
import pickle
import numpy as np

import json
import random
import sqlite3

# Load prediction model
stemmer = LancasterStemmer()

intents = json.loads(open('intents2.json').read())
words = pickle.load(open('words.pkl', 'rb'))
labels = pickle.load(open('labels.pkl', 'rb'))


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


def predict_class(sentence):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": labels[r[0]], "probability": str(r[1])})
    return return_list


# Load Gemini AI model
genai.configure(api_key='AIzaSyB2zvehKi1RBJ4dyWvg1i1-YiG_RAopFtg')
gemini_model = genai.GenerativeModel('gemini-pro')
chat = gemini_model.start_chat(history=[])

warnings.filterwarnings('ignore')
model = pickle.load(open('LL.pkl', 'rb'))
import telepot
bot=telepot.Bot("8006509116:AAF_yezrU92XfyiJ1VVRkWxofwX_YZXMoz0")
chatid="5032969785"

connection = sqlite3.connect('user_data.db')
cursor = connection.cursor()
command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
cursor.execute(command)

command = """CREATE TABLE IF NOT EXISTS sessions(name TEXT, password TEXT, timestamp TEXT)"""
cursor.execute(command)

app = Flask(__name__)
chat_history = []

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/class1')
def class1():
   return render_template('class1.html')

@app.route('/class2')
def class2():
   return render_template('class2.html')

@app.route('/class3')
def class3():
   return render_template('class3.html')

@app.route('/class4')
def class4():
   return render_template('class4.html')


@app.route('/uhome')
def uhome():
    return render_template('index.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT * FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchone()

        if result:
            
            from datetime import datetime
            now = datetime.now()
            date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
            cursor.execute("INSERT INTO sessions VALUES ('"+name+"', '"+password+"', '"+str(date_time)+"')")
            connection.commit()

            data=requests.get("https://api.thingspeak.com/channels/2778154/feeds.json?results=2")
            hb=float(data.json()['feeds'][-1]['field1'])
            bp=float(data.json()['feeds'][-1]['field2'])
            ecg=float(data.json()['feeds'][-1]['field3'])
            temp=float(data.json()['feeds'][-1]['field4'])
            oxy=float(data.json()['feeds'][-1]['field5'])
           
            return render_template('logged.html',hb=hb,temp=temp,oxy=oxy,ecg=ecg,bp=bp)
        else:
            return render_template('index.html', msg='Sorry , Incorrect Credentials Provided,  Try Again')

    return render_template('index.html')


@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        query = "SELECT * FROM user WHERE mobile = '"+mobile+"'"
        cursor.execute(query)

        result = cursor.fetchone()
        if result:
            return render_template('index.html', msg='Phone number already exists')
        else:

            cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
            connection.commit()

            return render_template('index.html', msg='Successfully Registered')
    
    return render_template('index.html')

@app.route('/logout')
def logout():
    return render_template('index.html')


@app.route("/kidneyPage")
def kidneyPage():
    data=requests.get("https://api.thingspeak.com/channels/2778154/feeds.json?results=2")
    hb=float(data.json()['feeds'][-1]['field1'])
    bp=float(data.json()['feeds'][-1]['field2'])
    ecg=float(data.json()['feeds'][-1]['field3'])
    temp=float(data.json()['feeds'][-1]['field4'])
    oxy=float(data.json()['feeds'][-1]['field5'])

    return render_template('logged.html',hb=hb,temp=temp,oxy=oxy,ecg=ecg,bp=bp)


@app.route("/predictPage", methods = ['POST', 'GET'])
def predictPage():
    
 
    name = request.form['name']
    Age = request.form['age']
    sex = request.form['sex']
    if sex==1:
        sex="MALE"
    else:
        sex="FEMALE"
    bp = request.form['bp']
    oxy = request.form['oxy']
    print(oxy)
    hb = request.form['heart']
    ecg = request.form['ecg']
    Temperature = request.form['Temperature']
    to_predict_list = np.array([[bp,oxy,hb,ecg,Temperature]])
    print(to_predict_list)
    prediction = model.predict(to_predict_list)
    output = prediction[0]
    print("Prediction is {}  :  ".format(output))
    print(output)
    
    # Check the output values and retrive the result with html tag based on the value
    
    if output == 1:
        result="Healthy   !!!!!!" 
        med=""
        bot.sendMessage(chatid,str("Patient  :  "+str(name)+ "\n Age  :  "+str(Age)+ "\n Gender  :  "+str(sex)+ "\n Status  :  "+str(result)))
    if output == 2:
        result="Fever" 
        med="Diagnosis Drugs for Fever  \n  Paracetamol \n acetaminophen \n Tylenol  \n aspirin \n  Acephen  \n Ecpirin \n"
        bot.sendMessage(chatid,str("Patient  :  "+str(name)+ "\n Age  :  "+str(Age)+ "\n Gender  :  "+str(sex)+ "\n Status  :  "+str(result)+" \n  Medicine Provided  :  "+str(med)+ "LOCATION:  https://maps.app.goo.gl/XacX6FLK686rg6uc6"))
    if output == 3:
        result="Chest Pain" 
        med="Diagnosis Drugs for chest_pain \n Exercise ,\n Avoid carbonated beverages , \n Hydration ,\n Acupuncture or Acupressure, \n Herbal Remedies ,\n Diltiazem \n"
        bot.sendMessage(chatid,str("Patient  :  "+str(name)+ "\n Age  :  "+str(Age)+ "\n Gender  :  "+str(sex)+ "\n Status  :  "+str(result)+" \n  Medicine Provided  :  "+str(med)+ "LOCATION https://maps.app.goo.gl/XacX6FLK686rg6uc6"))
    if output == 4:
        result="Critical" 
        med="You are critical \n concern the doctor nearby"
        print("Patient  :  "+str(name)+ "\n Age  :  "+str(Age)+ "\n Gender  :  "+str(sex)+ "\n Status  :  "+str(result)+" \n  Medicine Provided  :  "+str(med)+ "LOCATION https://maps.app.goo.gl/XacX6FLK686rg6uc6")
        bot.sendMessage(chatid,str("Patient  :  "+str(name)+ "\n Age  :  "+str(Age)+ "\n Gender  :  "+str(sex)+ "\n Status  :  "+str(result)+" \n  Medicine Provided  :  "+str(med)+ "LOCATION https://maps.app.goo.gl/XacX6FLK686rg6uc6"))
    
    
    # out=output
    print(result,output)
    return render_template('predict.html', result = result,out=output,name =name,med=med )  #,out=out)

    # return render_template('logged.html')

@app.route('/msg',methods = ['POST', 'GET'])
def msg():
    if request.method == 'POST':
        fileName=request.form['filename']
        img='dataset/'+fileName
        bot.sendPhoto(chatid, photo=open(img,'rb'))

    return render_template('logged.html')

@app.route('/analyse', methods=['GET', 'POST'])
def analyse():
    if request.method == 'POST':
        user_input = request.form['query']
        # Get response from Gemini AI model
        gemini_response = chat.send_message(user_input)

        data = gemini_response.text
        print(data)
        result = []
        for row in data.split('*'):
            if row.strip() != '':
                result.append(row)
        print(result)

        import csv
        f = open('HOSPITAL.csv', 'r')
        data = csv.reader(f)
        header = next(data)
        # Update chat history with both responses
        chat_history.append([user_input, result])
        try:
            for row in data:
                if row[1] in user_input:
                    name = row[0]
                    Link = row[2]
                    break
            return render_template('chatbot.html', status="Sever", chat_history=chat_history, name=name, Link=Link)
        except:
            return render_template('chatbot.html', chat_history=chat_history)
    return render_template('chatbot.html')

if __name__ == '__main__':
	app.run(debug = True, use_reloader=False)
