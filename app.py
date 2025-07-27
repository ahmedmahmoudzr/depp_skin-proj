import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# تحميل الموديل المحفوظ
model = load_model("skin_cancer_model.h5")

# القاموس الخاص بالأصناف (هتكون موجودة بالفعل في مشروعك)
classes = {
    4: ('nv', 'melanocytic nevi'),
    6: ('mel', 'melanoma'),
    2: ('bkl', 'benign keratosis-like lesions'),
    1: ('bcc', 'basal cell carcinoma'),
    5: ('vasc', 'pyogenic granulomas and hemorrhage'),
    0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'),
    3: ('df', 'dermatofibroma')
}

st.title("Skin Cancer Detection")
image_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if image_file is not None:
    img = Image.open(image_file)

    img_rgb = img.convert("RGB")

    st.image(img_rgb, caption="Uploaded Image", use_column_width=True)

    # تصغير الصورة لـ 28x28 إذا كان الموديل مدرب على هذا الحجم
    img_resized = img.resize((28, 28))
    img_input = np.array(img_resized) / 255.0
    img_input = img_input.reshape(1, 28, 28, 3)  # Resize to match model input shape

    prediction = model.predict(img_input)
    class_index = np.argmax(prediction[0])
    class_name = classes[class_index][1]  
    confidence = max(prediction[0]) * 100  

    st.write(f"Predicted Class: {class_name}")
    st.write(f"Confidence: {confidence:.2f}%")
