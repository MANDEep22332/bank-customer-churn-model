# -*- coding: utf-8 -*-
"""
Created on Fri May 22 12:09:07 2026

@author: mandeep
"""
import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, render_template

app = Flask(__name__)

# --- 1. SETUP: Download, Prepare, and Train ONCE at startup ---
print("Downloading dataset and training model...")
path = kagglehub.dataset_download("radheshyamkollipara/bank-customer-churn")
df = pd.read_csv(f"{path}/Customer-Churn-Records.csv")

# Preprocessing
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
df['Geography'] = df['Geography'].map({'France': 0, 'Germany': 1, 'Spain': 2})
df['Card Type'] = df['Card Type'].map({'SILVER': 0, 'GOLD': 1, 'PLATINUM': 2, 'DIAMOND': 3})

features = ['Complain', 'Age', 'NumOfProducts', 'Balance', 'IsActiveMember']
X = df[features]
y = df['Exited']

# Train the model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
print("Model trained successfully.")

# --- 2. ROUTES ---
@app.route("/", methods=['GET', 'POST'])
def CustomerchurnPrediction():
    if request.method == 'POST':
        # Get input from form
        q1 = float(request.form['query1'])
        q2 = float(request.form['query2'])
        q3 = float(request.form['query3'])
        q4 = float(request.form['query4'])
        q5 = float(request.form['query5'])
        
        # Make Prediction
        new_df = pd.DataFrame([[q1, q2, q3, q4, q5]], columns=features)
        prediction = rf.predict(new_df)
        proba = rf.predict_proba(new_df)
        
        if prediction[0] == 1:
            output1 = "The customer is likely to CHURN."
            output2 = "Confidence: {:.2f}%".format(proba[0][1] * 100)
        else:
            output1 = "The customer is likely to STAY."
            output2 = "Confidence: {:.2f}%".format(proba[0][0] * 100)
            
        return render_template('home.html', output1=output1, output2=output2, 
                               query1=q1, query2=q2, query3=q3, query4=q4, query5=q5)
    
    return render_template('home.html', query="")

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, use_reloader=False)