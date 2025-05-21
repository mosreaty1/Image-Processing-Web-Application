# Image-Processing-Web-Application
A Flask-based web application for applying various image processing techniques and filters to uploaded images.

✨ Features
Image Processing (app.py)
Brightness enhancement

Contrast adjustment

Gamma transformation

Histogram equalization

Image compression (JPEG and PNG)

Image segmentation (Thresholding, Watershed)

Gray level slicing

Image Filters (filter.py)
Grayscale conversion

Standard deviation filter

Gaussian blur

Image complement (inversion)

Salt & pepper noise addition and denoising

Parallel processing with ThreadPoolExecutor

🛠️ Technologies Used
Backend: Python, Flask

Image Processing: OpenCV, Pillow, NumPy, scikit-image

Frontend: HTML, CSS

Concurrency: ThreadPoolExecutor

🚀 Installation & Setup
Clone the repository:

bash
git clone https://github.com/yourusername/image-processing-app.git
cd image-processing-app
Create a virtual environment (recommended):

bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install dependencies:

bash
pip install -r requirements.txt
Run the application:

For image processing:

bash
python app.py
For image filters:

bash
python filter.py
Access the application:
Open your browser and navigate to:

Image Processing: http://localhost:5000

Image Filters: http://localhost:5000

📂 Project Structure
image-processing-app/
├── static/
│   ├── uploads/          # Uploaded and processed images
│   ├── style.css         # CSS for filter.html
│   └── styles.css        # CSS for upload.html
├── templates/
│   ├── filter.html       # Image filters interface
│   └── upload.html       # Image processing interface
├── app.py                # Main image processing application
├── filter.py             # Image filters application
└── README.md
📝 Requirements
Python 3.7+

Required packages (install via pip install -r requirements.txt):

flask
pillow
opencv-python
numpy
scikit-image
scipy
👥 Contributors
Mohamed Alsariti

Mariam Alrafaei

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

Note: Make sure to create a requirements.txt file with all the necessary dependencies if you haven't already. You can generate one using:

bash
pip freeze > requirements.txt
