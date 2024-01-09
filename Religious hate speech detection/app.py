
from flask import Flask, request,jsonify,json,render_template,Markup
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)

model = pickle.load(open('svc.pkl','rb'))
# model = pickle.load(open('mlp_model.pkl','rb'))
# model = pickle.load(open('mlp_model.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    sentence = request.form.get('sentence')
    # sentence = cv.fit_transform(sentence)
    res = {'sentence':sentence}
    input = [sentence]
    res = model.predict(input)
    result = res.item()

    # if result == 1:
    #     result = 'Religious Hate Speech'
    # else:
    #     result = "<h1>I'm bolded!<h1>"

    return render_template('index.html',result=result)

    # return result

    # return jsonify({'hate speech':result.item()})

if __name__ == '__main__':
    app.run(debug = True)
