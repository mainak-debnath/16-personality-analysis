from nltk.stem.porter import PorterStemmer
from flask import Flask, render_template, request
import pickle
import nltk
from nltk.corpus import stopwords
import string
nltk.download('stopwords')
ps = PorterStemmer()

app = Flask(__name__)

EorI_model = pickle.load(open('model_ie.pickle', 'rb'))
SorN_model = pickle.load(open("model_ns.pickle", 'rb'))
TorF_model = pickle.load(open("model_tf.pickle", 'rb'))
JorP_model = pickle.load(open("model_jp.pickle", 'rb'))
tfidf = pickle.load(open('vectorizer.pickle', 'rb'))


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/base')
def base():
    return render_template('base.html')


@app.route('/predict', methods=['POST'])
def predict():
    string = 'Your Personality Type Is '
    pType = ''
    if request.method == 'POST':
        text = request.form['message']
        transformed_text = transform_text(text)
        vector_input = tfidf.transform([transformed_text])
        result1 = EorI_model.predict(vector_input)[0]
        result2 = SorN_model.predict(vector_input)[0]
        result3 = TorF_model.predict(vector_input)[0]
        result4 = JorP_model.predict(vector_input)[0]
        if result1 == 0:
            pType += 'I'
        else:
            pType += 'E'
        if result2 == 0:
            pType += 'N'
        else:
            pType += 'S'
        if result3 == 0:
            pType += 'T'
        else:
            pType += 'F'
        if result4 == 0:
            pType += 'J'
        else:
            pType += 'P'
        if pType == '':
            return render_template('base.html', prediction_text='Please enter some text!!!')
        else:
            return render_template('base.html', prediction_text=string + str(pType))


if __name__ == "__main__":
    app.run(debug=True, port=8080)


if __name__ == "__main__":
    app.run(debug=True, port=8080)
