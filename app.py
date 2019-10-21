
import os
import io 
import base64

from flask import Flask, flash, request, redirect, url_for,render_template
from werkzeug import secure_filename
from wordcloud import WordCloud

UPLOAD_FOLDER = 'NlpReview'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle



# load the model from disk



clf = pickle.load(open('NlpReview/nlp_model.pkl', 'rb'))
cv=pickle.load(open('NlpReview/tranform.pkl','rb'))


@app.route('/')
def home():
    return render_template("ReviewAnalysis.html")

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html',prediction = my_prediction)


@app.route('/uploader', methods = ['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        df = pd.read_csv(app.config['UPLOAD_FOLDER']+"/"+filename, encoding="latin-1")
        data= df['Comment']
        vect = cv.transform(data)        
        df.insert (1, "target", clf.predict(vect))
        os.remove(app.config['UPLOAD_FOLDER']+"/"+filename)
        #df.to_csv("com4.csv")
        swarm_plot =  sns.countplot(x='target',data=df)
        fig = swarm_plot.get_figure()     
        figfile = io.BytesIO()
        fig.savefig(figfile, format='png')     
        figfile.seek(0)    
        data_uri = base64.b64encode(figfile.read()).decode('ascii') 
        text = df.Comment[0]

        # Create and generate a word cloud image:
        filename = "wine3.png"
        wordcloud = WordCloud().generate(text)
        wordcloud.to_file(app.config['UPLOAD_FOLDER']+"/"+filename)
  
        
        with open(app.config['UPLOAD_FOLDER']+"/"+filename, "rb") as f:
             worldcloudimg=base64.b64encode(f.read()).decode('ascii') 
#         os.remove(app.config['UPLOAD_FOLDER']+"/"+filename)
        
#         message = app.config['UPLOAD_FOLDER']+"/"+filename
    return render_template('ReviewAnalysis.html', plot_img =data_uri,wordcloud_img=worldcloudimg) 
        
  #  return str(df.shape[0])
#     return render_template('result.html',prediction = my_prediction)








if __name__ == '__main__':
    app.run()