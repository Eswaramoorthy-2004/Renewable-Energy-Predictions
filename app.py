from flask import Flask, render_template, request
import joblib
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# Load the trained models
solar_model = joblib.load('solar_model.pkl')
wind_model = joblib.load('wind_model.pkl')

# Route for the index page
@app.route('/')
def index():
    return render_template('index.html')

# Route for solar energy prediction index
@app.route('/predict/solar')
def solar_index():
    return render_template('solar_index.html')

# Route for wind energy prediction index
@app.route('/predict/wind')
def wind_index():
    return render_template('wind_index.html')

# Route for solar energy prediction result
@app.route('/predict/solar/result', methods=['POST'])
def predict_solar():
    if request.method == 'POST':
        # Get input data from the form
        cloud_coverage = float(request.form['cloud_coverage'])
        visibility = float(request.form['visibility'])
        temperature = float(request.form['temperature'])
        dew_point = float(request.form['dew_point'])
        relative_humidity = float(request.form['relative_humidity'])
        wind_speed = float(request.form['wind_speed'])
        station_pressure = float(request.form['station_pressure'])

        # Impute missing values
        imputer = SimpleImputer(strategy='mean')
        X_input = [[cloud_coverage, visibility, temperature, dew_point, relative_humidity, wind_speed, station_pressure]]
        X_input_imputed = imputer.fit_transform(X_input)

        # Make prediction
        prediction = solar_model.predict(X_input_imputed)[0]

        return render_template('solar_result.html', prediction=prediction)

# Route for wind energy prediction result
@app.route('/predict/wind/result', methods=['POST'])
def predict_wind():
    if request.method == 'POST':
        # Get the input values from the form
        pressure = float(request.form['pressure'])
        wind_direction = float(request.form['wind_direction'])
        wind_speed = float(request.form['wind_speed'])

        # Make a prediction using the trained model
        prediction = wind_model.predict([[pressure, wind_direction, wind_speed]])

        return render_template('wind_result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
