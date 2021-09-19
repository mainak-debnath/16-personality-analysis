from flask import Flask, render_template, url_for, request

app = Flask(__name__)


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
    if request.method == 'POST':
        value = request.form['message']
        return render_template('base.html', prediction_text=string + str(value))


if __name__ == "__main__":
    app.run(debug=True, port=8080)


if __name__ == "__main__":
    app.run(debug=True, port=8080)
