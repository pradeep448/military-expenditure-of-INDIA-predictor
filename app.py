from flask import Flask, render_template,request
import numpy as np
import joblib

app = Flask(__name__)
model=joblib.load('mil_exp_PLR.pkl')
@app.route("/")
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0]/1E9, 2)
    return render_template('index.html', 
                            prediction_text=f'Predicted Militray Expenditure of INDIA is USD {output} billion for year {final_features[0][0]}')

if __name__ == "__main__":
    app.run()
