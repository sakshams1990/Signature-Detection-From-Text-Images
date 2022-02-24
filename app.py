from flask import Flask, request
from flasgger import Swagger
from detecto import core
from predict import test_single_file

app = Flask(__name__)
Swagger(app)
model = core.Model.load('Trained_Models/model_weights.pth', ['Signature'])


@app.route('/predict', methods=["POST"])
def identify_signature():
    """ Identify signature from the document
    This is using docstrings for specifications.
    ---
    parameters:
      - name: filename
        in: path
        type: string
        required: true

    responses:
        200:
            description: The output values

    """
    filename = request.args.get("filename")
    response = test_single_file(model, filename, 0.75)
    return response


if __name__ == '__main__':
    app.run(debug=True)
