# Galaxy Classifier Web Application

A web application for classifying galaxy images using a deep learning model.

## Setup Instructions

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Flask Server

```bash
python app.py
```

The server will start on `http://localhost:5000`

You should see output like:
```
Starting Galaxy Classifier server...
Model loaded from: [...]/galaxy_classifier.keras
Classes: ['features_or_disk_bulge', 'features_or_disk_no_bulge', 'features_or_disk_odd', 'smooth_no_odd', 'smooth_odd']
 * Serving Flask app 'app'
 * Running on http://localhost:5000
```

### 3. Open the Web Interface

Open `index.html` in your web browser. You can:
- Simply double-click the file to open it
- Or use a local HTTP server (e.g., Python's built-in server):
  ```bash
  python -m http.server 8000
  ```
  Then visit `http://localhost:8000`

## How to Use

1. **Upload an Image**: Click on the file input to select a galaxy image
2. **Preview**: The image will display in the page
3. **Classify**: Click the "Classify Image" button
4. **View Results**: The classification and confidence percentage will appear below the image

## File Structure

- `index.html` - Web interface with image upload and results display
- `app.py` - Flask backend server
- `predict.py` - File containing prediction logic
- `galaxy_classifier.keras` - Pre-trained Keras model
- `class_names.json` - Galaxy classification categories
- `requirements.txt` - Python dependencies

## Classification Categories

The model can classify images into 5 galaxy types:
- features_or_disk_bulge
- features_or_disk_no_bulge
- features_or_disk_odd
- smooth_no_odd
- smooth_odd

## Troubleshooting

### "Make sure the Flask server is running"
- Check that `app.py` is running in a terminal
- Verify the server is accessible at `http://localhost:5000`

### Image processing errors
- Ensure the image is a valid image file (JPG, PNG, etc.)
- Try with a different image
- Check the console output in the Flask terminal for specific errors

### Port already in use
- Change the port in `app.py` (last line):
  ```python
  app.run(debug=True, host='localhost', port=5001)  # Use 5001 instead
  ```
- Update the URL in `index.html` accordingly

## API Endpoint

### POST /classify
Upload an image for classification

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Parameter: `image` (file)

**Response:**
```json
{
    "classification": "smooth_no_odd",
    "probability": 0.9234,
    "all_predictions": {
        "features_or_disk_bulge": 0.001,
        "features_or_disk_no_bulge": 0.005,
        "features_or_disk_odd": 0.002,
        "smooth_no_odd": 0.9234,
        "smooth_odd": 0.0644
    }
}
```
