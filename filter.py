import os
from flask import Flask, request, render_template, send_from_directory
from PIL import Image, ImageEnhance
import cv2
import numpy as np
from scipy.ndimage import generic_filter, gaussian_filter
from skimage.util import random_noise
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create a thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=4)

@app.route('/')
def index():
    return render_template('filter.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return "No file part", 400
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400
    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Resize the image to reduce processing time
        resize_filepath = resize_image(filepath)

        # Process filters concurrently
        futures = [
            executor.submit(apply_filters_concurrently, resize_filepath)
        ]
        
        results = [future.result() for future in futures]
        
        # Assuming the results are returned in the order we submitted the tasks
        gray_img_name, std_dev_img_name, gaussian_img_name, complement_img_name, salt_pepper_img_name, denoised_img_name = results[0]

        return render_template(
            'filter.html',
            original=filename,
            gray=gray_img_name,
            std_dev=std_dev_img_name,
            gaussian=gaussian_img_name,
            complement=complement_img_name,
            salt_pepper=salt_pepper_img_name,
            denoised=denoised_img_name
        )

def resize_image(image_path, size=(500, 500)):
    """Resize image to reduce processing time."""
    img = Image.open(image_path)
    img = img.resize(size, Image.Resampling.LANCZOS)  # Use LANCZOS instead of ANTIALIAS
    resized_path = os.path.join(UPLOAD_FOLDER, 'resized_' + os.path.basename(image_path))
    img.save(resized_path)
    return resized_path

def apply_filters_concurrently(filepath):
    """Apply all filters concurrently to the resized image."""
    gray_img_name = rgb_to_gray(filepath)
    std_dev_img_name = apply_std_dev_filter(filepath)
    gaussian_img_name = apply_gaussian_filter(filepath, sigma=0.9)
    complement_img_name = complement_image(filepath)
    salt_pepper_img_name, denoised_img_name = add_and_denoise_salt_pepper(filepath)

    return gray_img_name, std_dev_img_name, gaussian_img_name, complement_img_name, salt_pepper_img_name, denoised_img_name

def rgb_to_gray(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img_name = 'gray_' + os.path.basename(image_path)
    gray_img_path = os.path.join(UPLOAD_FOLDER, gray_img_name)
    cv2.imwrite(gray_img_path, gray)
    return gray_img_name

def apply_std_dev_filter(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    filtered_img = generic_filter(img, np.std, size=(3, 3))
    std_dev_img_name = 'std_dev_' + os.path.basename(image_path)
    std_dev_img_path = os.path.join(UPLOAD_FOLDER, std_dev_img_name)
    cv2.imwrite(std_dev_img_path, filtered_img)
    return std_dev_img_name

def apply_gaussian_filter(image_path, sigma=0.9):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    smoothed_img = gaussian_filter(img, sigma=sigma)
    gaussian_img_name = 'gaussian_' + os.path.basename(image_path)
    gaussian_img_path = os.path.join(UPLOAD_FOLDER, gaussian_img_name)
    cv2.imwrite(gaussian_img_path, smoothed_img)
    return gaussian_img_name

def complement_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    complemented_img = np.invert(img)
    complement_img_name = 'complement_' + os.path.basename(image_path)
    complement_img_path = os.path.join(UPLOAD_FOLDER, complement_img_name)
    cv2.imwrite(complement_img_path, complemented_img)
    return complement_img_name

def add_and_denoise_salt_pepper(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    noisy_img = random_noise(img, mode='s&p')
    noisy_img = np.array(255 * noisy_img, dtype='uint8')  # Convert to 8-bit format
    denoised_img = cv2.medianBlur(noisy_img, 3)  # Denoise using median filter

    salt_pepper_img_name = 'salt_pepper_' + os.path.basename(image_path)
    denoised_img_name = 'denoised_' + os.path.basename(image_path)
    salt_pepper_img_path = os.path.join(UPLOAD_FOLDER, salt_pepper_img_name)
    denoised_img_path = os.path.join(UPLOAD_FOLDER, denoised_img_name)

    cv2.imwrite(salt_pepper_img_path, noisy_img)
    cv2.imwrite(denoised_img_path, denoised_img)

    return salt_pepper_img_name, denoised_img_name

@app.route('/uploads/<filename>')
def serve_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
