import json
import requests

# Send a GET request
url_get = "http://127.0.0.1:8000"
r_get = requests.get(url_get)

# Print the status code
print("GET Request Status Code:", r_get.status_code)

# Print the welcome message
print("Welcome Message from GET Request:", r_get.text)

# Data for the POST request
data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# Send a POST request
url_post = "http://127.0.0.1:8000/data/"
r_post = requests.post(url_post, json=data)

# Print the status code
print("POST Request Status Code:", r_post.status_code)

# Print the result
print("Result from POST Request:", r_post.json()["result"])

