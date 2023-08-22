from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from joblib import load
import plotly.express as px
import plotly.graph_objects as go
import uuid

# Initialize Flask app
app = Flask(__name__)

# Define route for both GET and POST methods
@app.route("/", methods = ['GET', 'POST'])
def hello_world():
    try:
        # Check if the request is GET or POST
        request_str = request.method
        if request_str == 'GET':
            # If GET, render the base page
            return render_template('index.html', href = 'static/base_pic.svg')
        else:
            # If POST, get the text from the form
            text = request.form['text']
            # Generate a random string for the filename
            random_str = uuid.uuid4().hex
            # Define the path for the new file
            path = './static/' + random_str + '.svg'
            # Create the graph
            make_graph('./AgesAndHeights.pkl', load('./model.joblib'), float_to_numpy(text), path)
            # Render the page with the new graph
            return render_template('index.html', href = path)
    except Exception as e:
        # If an error occurs, return the error message
        return f"An error occurred: {e}"

# Function to create the graph
def make_graph(training_data_filename, model, new_input, output_file = None):
    # Load the training data
    data = pd.read_pickle(training_data_filename)
    # Filter out invalid ages
    ages = data['Age']
    data = data[ages > 0]
    # Convert height to meters and rename columns
    data['Height'] = data['Height'] * 0.0254
    data = data.rename(columns = {'Age': 'Age(y)', 'Height': 'Height(m)'})  
    ages = data['Age(y)']
    heights = data['Height(m)']
    # Predict the heights for ages 0-18
    x_try = np.array(list(range(19))).reshape(19, 1)
    preds = model.predict(x_try)
    # Create the scatter plot
    fig = px.scatter(x = ages, y = heights, title = 'Height vs. Age', labels = {'x': 'Age (years)', 'y': 'Height (meters)'})
    # Add the model's predictions to the plot
    fig.add_trace(go.Scatter(x = x_try.reshape(19), y = preds, mode = 'lines', name = 'model'))
    # Predict the heights for the new input
    new_preds = model.predict(new_input)
    # Add the new predictions to the plot
    fig.add_trace(go.Scatter(x = new_input.reshape(len(new_input)), y = new_preds, name = 'Output', 
                                                    mode = 'markers', 
                                                    marker = dict(color = 'purple', size = 10,
                                                                line = dict(color = 'purple', width = 2))))
    # Save the plot as an image
    fig.write_image(output_file, width = 800, engine = 'kaleido')
    # Show the plot
    fig.show()

# Function to convert a string of floats to a numpy array
def float_to_numpy(float_str):
    # Function to check if a string can be converted to a float
    def is_float(x):
        try:
            float(x)
            return True
        except:
            return False
    # Convert the string to a numpy array of floats
    floats = np.array([float(x) for x in float_str.split(',') if is_float(x)])
    # Reshape the array to be 2D
    return floats.reshape(len(floats), 1)