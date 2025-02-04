from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
import joblib
import dill
import joblib

app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")


@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!" 


@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            reviewText =str(request.form['review_text'])



            with open('SentimentalAnalysisPrrojectModelFinal.dill', 'rb') as f:
                model = dill.load(f)

            vectorizer_reviewText = joblib.load('SentimentalAnalysisPrrojectVectorizervectorReviewText.joblib') 

            # Transform the text data using the saved TF-IDF vectorizer
            review_sample_scaled = reviewText
            X_reviewText = [review_sample_scaled]

            # Transform the text using the vectorizer
            X_reviewText_scaled = vectorizer_reviewText.transform(X_reviewText)

            #  Make predictions using the trained model
            y_pred_new = model.predict(review_sample_scaled)

            # Output the result (the predicted sentiment)
            print(f"Predicted Sentiment: {y_pred_new}")

            # Display sentiment labels
            sentiment_labels = {0: 'Negative', 1: 'Positive'}
            print(f"The given sentence is: {sentiment_labels.get(y_pred_new, 'Unknown')}")
    
            # model = joblib.load("SentimentalAnalysisPrrojectModelFinal.dill")
            # predict = model(reviewText)

            return render_template('results.html', prediction = str(y_pred_new))

        except Exception as e:
            print('The Exception message is ghg: ',e)
            return 'something is wrong'

    else:
        return render_template('index.html')


if __name__ == "__main__":
	# app.run(host="0.0.0.0", port = 8080, debug=True)
	app.run(host="0.0.0.0", port = 8080)