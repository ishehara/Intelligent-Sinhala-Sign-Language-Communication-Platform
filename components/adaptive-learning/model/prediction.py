import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import ImageFont, ImageDraw, Image

# Resolve paths relative to this script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define class labels (corresponding to your 25 classes)
# Mapping: 1->අ, 2->ආ, 3->ඇ, 4->ඉ, 5->උ, 6->එ, 7->ක, 8->ග, 9->ට, 10->ද, 11->ත, 12->ඩ, 13->න, 14->ප, 15->බ, 16->ම, 17->ය, 18->ර, 19->ල, 20->ව, 21->ස, 22->හ, 23->ං, 24->ච, 25->ෆ
class_labels = ['අ', 'ද', 'ත', 'ඩ', 'න', 'ප', 'බ', 'ම', 'ය', 'ර', 'ල', 'ආ', 'ව', 'ස', 'හ', 'ං', 'ච', 'ෆ', 'ඇ', 'ඉ', 'උ', 'එ', 'ක', 'ග', 'ට']

# Parameters matching dataset creation
offset = 20
imgSize = 300  # Intermediate size for white background image
target_size = (224, 224)  # Model input size

def put_unicode_text(img, text, position, font_size=32, color=(0, 255, 0)):
    """
    Put Unicode text (like Sinhala) on OpenCV image using PIL
    """
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # Try to use fonts that support Sinhala (in order of preference)
    font_paths = [
        os.path.join(SCRIPT_DIR, "NotoSansSinhala-VariableFont_wdth,wght.ttf"),
        os.path.join(SCRIPT_DIR, "NotoSansSinhala-Regular.ttf"),
        os.path.join(SCRIPT_DIR, "fonts", "NotoSansSinhala-Regular.ttf"),
        os.path.join(os.environ.get('WINDIR', 'C:/Windows'), 'Fonts', 'iskpota.ttf'),
        os.path.join(os.environ.get('WINDIR', 'C:/Windows'), 'Fonts', 'iskpotab.ttf'),
        os.path.join(os.environ.get('WINDIR', 'C:/Windows'), 'Fonts', 'NotoSansSinhala-Regular.ttf'),
        os.path.join(os.environ.get('WINDIR', 'C:/Windows'), 'Fonts', 'segoeui.ttf'),
    ]
    
    font = None
    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, font_size)
            break
        except Exception:
            continue
    
    if font is None:
        font = ImageFont.load_default()
    
    draw.text(position, text, font=font, fill=color)
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img_bgr

def preprocess_image(imgWhite):
    # Resize to model input size
    img_resized = cv2.resize(imgWhite, target_size)
    img_normalized = img_resized / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0)
    return img_expanded


def main():
    # Load the trained model
    model_path = os.path.join(SCRIPT_DIR, 'sinhala_sign_language_classifier.keras')
    model = load_model(model_path)

    # Initialize video capture
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    # Hand detector
    detector = HandDetector(maxHands=1, detectionCon=0.8)

    while True:
        success, img = cap.read()
        if not success:
            break

        hands, img = detector.findHands(img)
        predicted_class = "No hand detected"
        confidence = 0.0

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            # Create white background image
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            # Crop the hand region with offset (handle boundaries)
            y1 = max(0, y - offset)
            y2 = min(img.shape[0], y + h + offset)
            x1 = max(0, x - offset)
            x2 = min(img.shape[1], x + w + offset)
            imageCrop = img[y1:y2, x1:x2]

            # Calculate aspect ratio and resize to fit white image
            imgCropShape = imageCrop.shape
            h_crop, w_crop = imgCropShape[0], imgCropShape[1]

            if h_crop == 0 or w_crop == 0:
                continue  # Skip if invalid crop

            aspectRatio = h_crop / w_crop

            if aspectRatio > 1:
                k = imgSize / h_crop
                wCal = math.ceil(k * w_crop)
                if wCal > 0:
                    imageResize = cv2.resize(imageCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imageResize
            else:
                k = imgSize / w_crop
                hCal = math.ceil(k * h_crop)
                if hCal > 0:
                    imageResize = cv2.resize(imageCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imageResize

            # Preprocess for model
            processed_img = preprocess_image(imgWhite)

            # Predict
            prediction = model.predict(processed_img)
            predicted_index = np.argmax(prediction)
            confidence = np.max(prediction)
            predicted_class = class_labels[predicted_index]

        # Display prediction on the main image using PIL for Unicode support
        display_text = f'Predicted: {predicted_class} ({confidence*100:.2f}%)'
        img = put_unicode_text(img, display_text, (10, 30), font_size=32, color=(0, 255, 0))

        cv2.imshow("Image", img)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()