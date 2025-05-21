import os
from flask import Flask, request, render_template, send_from_directory
from PIL import Image, ImageEnhance
import cv2
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('upload.html')

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

        # Apply various transformations and processing
        bright_img_name, contrast_img_name = enhance_image(filepath)
        gamma_img_name = gamma_transformation(filepath)
        histogram_img_name = histogram_equalization(filepath)
        compressed_img_name = compress_image(filepath)
        lossless_compressed_img_name = lossless_compression(filepath)
        segmented_img_name = segment_image(filepath)
        watershed_img_name = watershed_segmentation(filepath)
        threshold_img_name = thresholding(filepath)
        gray_slice_img_name = gray_level_slicing(filepath)

        return render_template(
            'upload.html',
            original=filename,
            bright=bright_img_name,
            contrast=contrast_img_name,
            gamma=gamma_img_name,
            histogram=histogram_img_name,
            compressed=compressed_img_name,
            lossless=lossless_compressed_img_name,
            segmented=segmented_img_name,
            watershed=watershed_img_name,
            threshold=threshold_img_name,
            gray_slice=gray_slice_img_name
        )

def enhance_image(image_path):
    img = Image.open(image_path)

    enhancer = ImageEnhance.Brightness(img)
    bright_img = enhancer.enhance(1.5)
    bright_img_name = 'bright_' + os.path.basename(image_path)
    bright_img_path = os.path.join(UPLOAD_FOLDER, bright_img_name)
    bright_img.save(bright_img_path)

    enhancer = ImageEnhance.Contrast(img)
    contrast_img = enhancer.enhance(1.5)
    contrast_img_name = 'contrast_' + os.path.basename(image_path)
    contrast_img_path = os.path.join(UPLOAD_FOLDER, contrast_img_name)
    contrast_img.save(contrast_img_path)

    return bright_img_name, contrast_img_name

def gamma_transformation(image_path, gamma=0.5):
    img = cv2.imread(image_path)
    gamma_corrected = np.array(255 * (img / 255) ** gamma, dtype='uint8')
    gamma_img_name = 'gamma_' + os.path.basename(image_path)
    gamma_img_path = os.path.join(UPLOAD_FOLDER, gamma_img_name)
    cv2.imwrite(gamma_img_path, gamma_corrected)
    return gamma_img_name

def histogram_equalization(image_path):
    img = cv2.imread(image_path, 0)
    equalized_img = cv2.equalizeHist(img)
    histogram_img_name = 'histogram_' + os.path.basename(image_path)
    histogram_img_path = os.path.join(UPLOAD_FOLDER, histogram_img_name)
    cv2.imwrite(histogram_img_path, equalized_img)
    return histogram_img_name

def compress_image(image_path):
    img = cv2.imread(image_path)
    compressed_img_name = 'compressed_' + os.path.basename(image_path)
    compressed_img_path = os.path.join(UPLOAD_FOLDER, compressed_img_name)
    cv2.imwrite(compressed_img_path, img, [cv2.IMWRITE_JPEG_QUALITY, 50])
    return compressed_img_name

def lossless_compression(image_path):
    img = Image.open(image_path)
    lossless_img_name = 'lossless_' + os.path.basename(image_path).replace('.jpg', '.png')
    lossless_img_path = os.path.join(UPLOAD_FOLDER, lossless_img_name)
    img.save(lossless_img_path, format='PNG', compress_level=9)
    return lossless_img_name

def segment_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, segmented_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    segmented_img_name = 'segmented_' + os.path.basename(image_path)
    segmented_img_path = os.path.join(UPLOAD_FOLDER, segmented_img_name)
    cv2.imwrite(segmented_img_path, segmented_img)
    return segmented_img_name

def watershed_segmentation(image_path):
    # Load the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(thresh, kernel, iterations=3)  # Sure background

    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)

    markers = markers + 1
    markers[unknown == 255] = 0
    cv2.watershed(img, markers)

    img[markers == -1] = [255, 0, 0]

    watershed_img_name = 'watershed_' + os.path.basename(image_path)
    watershed_img_path = os.path.join(UPLOAD_FOLDER, watershed_img_name)
    cv2.imwrite(watershed_img_path, img)

    return watershed_img_name

def thresholding(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    threshold_img_name = 'threshold_' + os.path.basename(image_path)
    threshold_img_path = os.path.join(UPLOAD_FOLDER, threshold_img_name)
    cv2.imwrite(threshold_img_path, binary)
    return threshold_img_name

def gray_level_slicing(image_path, min_gray=100, max_gray=200):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sliced_img = np.zeros_like(img)
    sliced_img[(img >= min_gray) & (img <= max_gray)] = 255
    gray_slice_img_name = 'gray_slice_' + os.path.basename(image_path)
    gray_slice_img_path = os.path.join(UPLOAD_FOLDER, gray_slice_img_name)
    cv2.imwrite(gray_slice_img_path, sliced_img)
    return gray_slice_img_name

@app.route('/uploads/<filename>')
def serve_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
