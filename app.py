from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

#Data Preparation
df= pd.read_csv('experience_salary.csv')
X = df[["Experience", "Education_Level", "Projects", "Certifications"]]
y= df["Salary"]


#Splitting Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Model training
model = LinearRegression()
model.fit(X_train, y_train)

#Saving Trained model
joblib.dump(model, 'salary_model.pkl')

#Load the saved Model
model = joblib.load('salary_model.pkl')


#CReate App
app = Flask(__name__)

#Starting App
@app.route('/',methods=['GET','POST'])
def home():
    return render_template('form1.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        exp = float(request.form.get('years_of_experience'))
        edu = float(request.form.get('education_level'))
        proj = float(request.form.get('projects'))
        cert = float(request.form.get('certifications'))

        predicted_salary = float(model.predict([[exp, edu, proj, cert]])[0])
        predicted_salary = round(predicted_salary, 2)
        # Create Plot
        plt.figure(figsize=(7, 5))
        plt.scatter(df['Experience'], df['Salary'], s=50)

        # Mark the predicted point
        plt.scatter(exp, predicted_salary, color='red', s=200)

        plt.xlabel("Experience (Years)")
        plt.ylabel("Salary")
        plt.title("Experience vs Salary (Prediction Highlight)")


        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()
        # convert to base 4 string
        plot_data = base64.b64encode(img.getvalue()).decode('utf8')
        return render_template('result.html', predicted_salary=predicted_salary, plot_url=plot_data)

    except:
        return "Invalid input! Please enter a number"


if __name__ == '__main__':
    app.run(debug=True)


