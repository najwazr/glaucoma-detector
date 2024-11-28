import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Fungsi untuk gaya visual
def apply_custom_css():
    st.markdown("""
    <style>
    body {
        background: #f0f4f8; /* Warna latar belakang lembut */
        font-family: 'Arial', sans-serif;
    }
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f4e79; /* Warna biru tua */
        text-align: center;
        margin-bottom: 20px;
    }
    .subheader {
        font-size: 1.2rem;
        color: #495057; /* Warna abu-abu tua */
        text-align: center;
        margin-bottom: 30px;
    }
    .footer {
        text-align: center;
        font-size: 0.8rem;
        color: #6c757d; /* Warna abu-abu */
        margin-top: 50px;
    }
    .stButton>button {
        background: #1f4e79; /* Tombol warna biru tua */
        color: white;
        font-size: 1rem;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
    }
    .stButton>button:hover {
        background: #145374; /* Biru lebih gelap saat di-hover */
    }
    </style>
    """, unsafe_allow_html=True)

# Fungsi untuk memproses gambar dan prediksi
def import_and_predict(image_data, model):
    image = ImageOps.fit(image_data, (100, 100), Image.LANCZOS)
    image = image.convert('RGB')
    image = np.asarray(image)
    st.image(image, caption="Uploaded Fundus Image", channels='RGB', use_container_width=True)
    image = (image.astype(np.float32) / 255.0)
    img_reshape = image[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

# Load model
model = tf.keras.models.load_model('my_model2.h5')

# Aplikasi utama
def main():
    # Terapkan CSS
    apply_custom_css()

    # Header utama
    st.markdown("<div class='main-header'>GLAUCOLens</div>", unsafe_allow_html=True)
    st.markdown("<div class='subheader'>See the World Clearly, Detect Glaucoma Early</div>", unsafe_allow_html=True)

    # Sidebar Navigation
    menu = ["Home", "Vision Simulator", "Detection", "Learn About Glaucoma"]
    choice = st.sidebar.selectbox("Navigation", menu)

    if choice == "Home":
        st.subheader("👁👁 About GLAUCOLens")
        st.write("""
        Welcome to *GLAUCOLens*, a platform dedicated to raising awareness about glaucoma 
        and helping users detect the disease early through image-based analysis.
        
        Why it matters:
        - Glaucoma is a leading cause of irreversible blindness.
        - Early detection can slow down or prevent vision loss.
        
        Explore our features:
        - Vision Simulator: Visualize the impact of glaucoma on vision.
        - Detection: Upload retina images to analyze glaucoma risks.
        """)
        st.image("glaucoma-web.jpeg", 
                 caption="Join us in preventing glaucoma blindness.", use_container_width=True)

    elif choice == "Vision Simulator":
        st.subheader("🔍 Vision Simulator")
        st.write("Adjust the slider to simulate how glaucoma affects vision.")

        # Slider untuk memilih tingkat keparahan
        severity = st.slider("Select Glaucoma Severity Level", 0, 100, 25)

        # Menampilkan gambar berdasarkan tingkat keparahan
        if severity < 25:
            img = Image.open("Normal.jpeg")
            st.image(img, caption="Normal Vision", use_container_width=True)
        elif severity < 50:
            img = Image.open("Mild.jpeg")
            st.image(img, caption="Early Glaucoma", use_container_width=True)
        elif severity < 75:
            img = Image.open("Medium.jpeg")
            st.image(img, caption="Advanced Glaucoma", use_container_width=True)
        else:
            img = Image.open("Severe.jpeg")
            st.image(img, caption="Severe Glaucoma", use_container_width=True)

    elif choice == "Detection":
        st.markdown("<div class='main-header'>Glaucoma Detection</div>", unsafe_allow_html=True)
        st.markdown("<div class='subheader'>Could you upload a picture of the inside of your eye (fundus)? It's the photo that shows the retina and blood vessels.</div>", unsafe_allow_html=True)
        
        # Kotak unggah gambar
        st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
        file = st.file_uploader("Upload Fundus Image", type=["jpg", "jpeg", "png"])
        st.markdown("</div>", unsafe_allow_html=True)
        if file is None:
            st.info("Please upload an image to proceed.")
        else:
            st.success("Image uploaded successfully! Analysis results will be displayed below.")
            
            imageI = Image.open(file)
            prediction = import_and_predict(imageI, model)
            pred = prediction[0][0]
            if pred > 0.5:
                st.write("""## **Prediction:** Your eye is healthy!😀""")
                st.balloons()
            else:
                st.write("""## **Prediction:** You are affected by Glaucoma😔 Please consult an ophthalmologist as soon as possible.""")

        st.markdown("<div style='text-align:center; margin-top:20px;'>", unsafe_allow_html=True)
        if st.button("Analyze Fundus"):
            st.warning("Analysis model integration is in progress. Please check back soon.")
        st.markdown("</div>", unsafe_allow_html=True)

    elif choice == "Learn About Glaucoma":
        st.markdown("<div class='main-header'>Learn About Glaucoma</div>", unsafe_allow_html=True)

        st.write("""
        Glaucoma is a group of eye diseases that damage the optic nerve, often caused by abnormally high pressure in the eye. 
        Left untreated, glaucoma can lead to irreversible blindness. Here's what you need to know:
        """)

        # Fakta menarik tentang glaukoma
        st.subheader("👁️ Fast Facts About Glaucoma")
        st.write("""
        - *Silent Thief of Sight*: Glaucoma often has no symptoms until significant vision loss occurs.
        - *Global Impact*: Over 70 million people worldwide are affected by glaucoma.
        - *Prevention*: Early detection is key to preventing vision loss.
        - *At Risk*: People over 60, those with a family history of glaucoma, and individuals with diabetes or high blood pressure.
        """)

        # Interaktif: Tes risiko
        st.subheader("🧐 Are You at Risk?")
        st.write("Answer these quick questions to learn about your glaucoma risk:")
        age = st.slider("Your Age", 10, 100, 30)
        family_history = st.selectbox("Do you have a family history of glaucoma?", ["No", "Yes"])
        diabetes = st.selectbox("Do you have diabetes?", ["No", "Yes"])
        high_bp = st.selectbox("Do you have high blood pressure?", ["No", "Yes"])

        if st.button("Check My Risk"):
            risk_score = 0
            if age > 60: risk_score += 2
            if family_history == "Yes": risk_score += 2
            if diabetes == "Yes": risk_score += 1
            if high_bp == "Yes": risk_score += 1

            if risk_score >= 4:
                st.error("High Risk: You should consult an eye specialist immediately.")
            elif risk_score >= 2:
                st.warning("Moderate Risk: Consider scheduling an eye check-up soon.")
            else:
                st.success("Low Risk: Keep up with routine eye check-ups.")

        # Edukasi mendalam
        st.subheader("📚 How to Protect Your Vision")
        st.write("""
        - *Routine Eye Exams*: Early detection through regular check-ups is crucial.
        - *Healthy Lifestyle*: Maintain a balanced diet, exercise regularly, and manage systemic health conditions like diabetes.
        - *Medication Adherence*: If diagnosed, follow prescribed treatments to control eye pressure.
        - *Awareness*: Share knowledge with friends and family about the importance of eye health.
        """)

        # Fakta unik tentang pengobatan dan inovasi teknologi
        st.subheader("🔬 Cutting-Edge Innovations in Glaucoma Care")
        st.write("""
        - *Laser Therapy*: Advanced laser treatments like SLT (Selective Laser Trabeculoplasty) are minimally invasive and effective.
        - *Smart Contact Lenses*: Emerging technology enables real-time monitoring of eye pressure.
        - *AI Diagnosis*: Artificial intelligence assists in early detection by analyzing retinal images with high accuracy.
        - *Gene Therapy*: Research is underway to develop gene-based solutions for hereditary glaucoma.
        """)

        # Inspirasi: Kisah Nyata
        st.subheader("💡 Inspiring Stories")
        st.write("""
        - "I was diagnosed early, and it saved my vision." - Maria, 45, a teacher who advocates for regular eye exams.
        - "Technology gave me hope." - James, 60, uses smart lenses to monitor his glaucoma daily.
        """)

        # Ajakan bertindak
        st.markdown("""
        <div style='background: #1f4e79; color: white; padding: 15px; text-align: center; border-radius: 10px;'>
        <h3>Protect Your Vision Today!</h3>
        <p>Schedule a comprehensive eye exam and spread the word about glaucoma prevention.</p>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("<div class='footer'>© 2024 GLAUCOLens. All Rights Reserved.</div>", unsafe_allow_html=True)
    
if __name__ == "__main__":
    main()
    else:
        st.write("""
                 ## **Prediction:** You are affected by Glaucoma. Please consult an ophthalmologist as soon as possible.
                 """
                 )
