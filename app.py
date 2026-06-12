from flask import Flask, request, jsonify, render_template
from predict import predict_image
import os

app = Flask(__name__, static_folder=os.path.dirname(os.path.abspath(__file__)))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/guide')
def guide():
    return render_template('guide.html')

@app.route('/classify', methods=['POST'])
def classify():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    result = predict_image(image_file)
    return jsonify(result)

if __name__ == '__main__':
    print("Starting Galaxy Classifier server...")
    app.run(debug=True, host='localhost', port=5000)
