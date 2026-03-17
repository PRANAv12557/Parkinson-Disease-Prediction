import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import pickle

st.set_page_config(
    page_title="Parkinson's Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI improvement
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Headers */
    h1 {
        color: #1e3d59;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
    }
    h2, h3 {
        color: #2b5876;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    
    /* Buttons */
    .stButton>button, .stFormSubmitButton>button {
        background-color: #ff6e40;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover, .stFormSubmitButton>button:hover {
        background-color: #ff521b;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(255, 110, 64, 0.3);
    }
    
    /* Metrics/Cards */
    div[data-testid="stMetricValue"] {
        color: #1e3d59;
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #ffffff;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
    
    /* Inputs */
    .stNumberInput>div>div>input {
        border-radius: 6px;
    }
    
    /* Success/Error boxes */
    .stSuccess {
        background-color: #d4edda;
        color: #155724;
        border-color: #c3e6cb;
        border-radius: 8px;
    }
    .stError {
        background-color: #f8d7da;
        color: #721c24;
        border-color: #f5c6cb;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = pickle.load(open('parkinson_model.pkl', 'rb'))
    pca = pickle.load(open('pca.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    return model, pca, scaler

model, pca, scaler = load_model()

# Sidebar navigation
with st.sidebar:
    st.image('parkinson2.jpg', use_container_width=True)
    st.markdown("## Navigation")
    page = st.radio(
        "Menu", ["Home", "Parkinson Prediction", "FAQs"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.info("Early detection and precise measurements of acoustic features can help predict Parkinson's Disease effectively.")

# Features List globally defined for consistency
features_list = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
            'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
            'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3',
            'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA',
            'NHR', 'HNR', 'RPDE', 'DFA',
            'spread1', 'spread2', 'D2', 'PPE']

# Home Page
if page == "Home":
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.title("Parkinson's Disease Prediction Platform")
        st.markdown("#### *Empowering Early Detection through Voice Analysis*")
        st.write("""
        Parkinson's disease is a progressive movement disorder of the nervous system. It causes nerve cells in parts of the brain to weaken and die, leading to problems with movement, tremor, stiffness, and impaired balance. 
                 
        This application utilizes a comprehensive machine learning model to predict the likelihood of Parkinson's Disease (PD) based on various vocal and acoustic measurements. In India, estimates suggest around 1.5 million people may be living with the disease. Early diagnosis is crucial for improving treatment outcomes.
        """)
        
        st.info("Did you know? Parkinson's Disease is a significant health concern globally, affecting millions worldwide. Early detection opens doors for better management.")
        
    with col2:
        image = Image.open('parkinson2.jpg')
        st.image(image, use_container_width=True, caption="Prioritize neurological health")

    st.markdown("---")
    
    # Key Statistics in cards
    st.subheader("Key Statistics")
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    with stat_col1:
        st.metric(label="Global Impact", value="10M+", delta="People worldwide", delta_color="off")
    with stat_col2:
        st.metric(label="Cases in India", value="1.5M+", delta="Estimated", delta_color="off")
    with stat_col3:
        st.metric(label="New Cases/Year", value="60K+", delta="In India alone", delta_color="off")

    st.markdown("---")

    col_vid, col_links = st.columns([2, 1])
    with col_vid:
        st.subheader("Educational Resource")
        st.video("https://www.youtube.com/watch?v=u_tozEV7f4k")
    with col_links:
        st.subheader("Dataset Source")
        st.write("The underlying model is trained on a robust dataset sourced from the UCI Machine Learning Repository.")
        st.markdown("[Explore Official Dataset](https://archive.ics.uci.edu/dataset/174/parkinsons)")

# Parkinson Prediction Page
elif page == "Parkinson Prediction":
    st.title("🧠 Parkinson's Disease Prediction")
    st.markdown("""
    Select an input method below to provide the necessary acoustic features. The model analyzes vocal frequency, jitter, shimmer, and other voice measures.
    """)

    # Create tabs for input methods
    tab1, tab2, tab3 = st.tabs(["✍️ Manual Input", "📄 CSV Upload", "🎤 Audio Upload"])

    with tab1: # Manual Input
        st.subheader("Manual Feature Entry")
        st.write("Please enter the 22 required acoustic features precisely, or upload a patient report to auto-fill the values.")
        
        # Optional Single Patient CSV Upload
        uploaded_report = st.file_uploader("Upload Single Patient Report (CSV) to Auto-fill", type=["csv"], key="single_upload")
        
        # Initialize default values
        default_vals = {feat: 0.0 for feat in features_list}
        
        if uploaded_report is not None:
            try:
                single_df = pd.read_csv(uploaded_report)
                if len(single_df) > 0:
                    # Take the first row if multiple exist
                    for feat in features_list:
                        if feat in single_df.columns:
                            default_vals[feat] = float(single_df[feat].iloc[0])
                    st.success("Values auto-filled from report! You can edit them below.")
                else:
                    st.warning("The uploaded CSV is empty.")
            except Exception as e:
                st.error(f"Error reading the CSV file: {e}")
        
        with st.form("prediction_form"):
            freq_features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)']
            jitter_features = ['MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP']
            shimmer_features = ['MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA']
            other_features = ['NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']
            
            features_dict = {}
            
            st.markdown("##### Fundamental Frequency Settings")
            cols = st.columns(3)
            for i, feat in enumerate(freq_features):
                with cols[i]:
                    features_dict[feat] = st.number_input(feat, value=default_vals[feat], format="%.5f", step=0.00001)
                    
            st.markdown("##### Jitter (Frequency Variation)")
            cols = st.columns(5)
            for i, feat in enumerate(jitter_features):
                with cols[i]:
                    features_dict[feat] = st.number_input(feat, value=default_vals[feat], format="%.5f", step=0.00001)
                    
            st.markdown("##### Shimmer (Amplitude Variation)")
            cols = st.columns(3)
            for i, feat in enumerate(shimmer_features[:3]):
                with cols[i]:
                    features_dict[feat] = st.number_input(feat, value=default_vals[feat], format="%.5f", step=0.00001)
            cols = st.columns(3)
            for i, feat in enumerate(shimmer_features[3:]):
                with cols[i]:
                    features_dict[feat] = st.number_input(feat, value=default_vals[feat], format="%.5f", step=0.00001)
                    
            st.markdown("##### Additional Parameters")
            cols = st.columns(4)
            for i, feat in enumerate(other_features[:4]):
                with cols[i]:
                    features_dict[feat] = st.number_input(feat, value=default_vals[feat], format="%.5f", step=0.00001)
            cols = st.columns(4)
            for i, feat in enumerate(other_features[4:]):
                with cols[i]:
                    features_dict[feat] = st.number_input(feat, value=default_vals[feat], format="%.5f", step=0.00001)

            submit_button = st.form_submit_button(label='Run Diagnostics')

        if submit_button:
            features = freq_features + jitter_features + shimmer_features + other_features
            input_data_ordered = [features_dict[f] for f in features_list]
            input_df = pd.DataFrame([input_data_ordered], columns=features_list)

            with st.spinner('Analyzing acoustic features...'):
                scaled_data = scaler.transform(input_df)
                pca_data = pca.transform(scaled_data)
                prediction = model.predict(pca_data)
                probability = model.predict_proba(pca_data)[0][1]

            st.markdown("---")
            st.subheader("Diagnostic Results")
            
            res_col1, res_col2 = st.columns([1, 1.5])
            with res_col1:
                if prediction[0] == 0:
                    st.success("#### Negative \nThe model indicates NO presence of Parkinson's Disease.")
                    st.metric("Confidence", f"{1-probability:.2%}")
                else:
                    st.error("#### Positive \nThe model indicates PRESENCE of Parkinson's Disease.")
                    st.metric("Confidence", f"{probability:.2%}")
                    
            with res_col2:
                feature_importance = abs(pca.components_[0])
                fig = go.Figure(data=[go.Bar(
                    x=features_list, 
                    y=feature_importance,
                    marker_color='#2b5876'
                )])
                fig.update_layout(
                    title="Feature Contribution to Outcome",
                    xaxis_title="", 
                    yaxis_title="Importance",
                    margin=dict(l=0, r=0, t=30, b=0),
                    height=250
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab2: # CSV Upload
        st.subheader("Batch Prediction via CSV")
        st.write("Upload a dataset containing patient records to process multiple predictions at once.")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            input_df = pd.read_csv(uploaded_file)
            st.dataframe(input_df, use_container_width=True)

            if st.button('Analyze Batch Data'):
                with st.spinner('Processing...'):
                    # Ensure columns match
                    try:
                        ordered_df = input_df[features_list]
                        scaled_data = scaler.transform(ordered_df)
                        pca_data = pca.transform(scaled_data)
                        predictions = model.predict(pca_data)
                        probabilities = model.predict_proba(pca_data)[:, 1]

                        # Add predictions to the dataframe
                        result_df = input_df.copy()
                        result_df['Prediction'] = ['Positive' if p == 1 else 'Negative' for p in predictions]
                        result_df['Probability'] = [f"{prob:.2%}" if p == 1 else f"{1-prob:.2%}" for p, prob in zip(predictions, probabilities)]

                        st.subheader("Batch Prediction Results")
                        st.dataframe(result_df, use_container_width=True)

                        # Download results
                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download results as CSV",
                            data=csv,
                            file_name="parkinson_batch_predictions.csv",
                            mime="text/csv",
                        )
                    except KeyError as e:
                        st.error(f"Missing required columns in CSV: {e}")

    with tab3: # Audio Upload
        st.info("💡 **Note on Audio Uploads**: This feature extracts acoustic features directly from your voice using `praat-parselmouth`. However, because some of the non-linear features (like DFA and RPDE) used proprietary algorithms, approximations may lead to slightly different predictions. **For best results, upload a clear, noise-free recording of a sustained 'Ahhhh' vowel sound.**")
        st.write("Upload a `.wav` audio recording of your voice.")
        
        uploaded_audio = st.file_uploader("Choose a WAV file", type=["wav"])
        
        if uploaded_audio is not None:
            st.audio(uploaded_audio, format='audio/wav')
            
            if st.button('Extract Features & Predict'):
                with st.spinner('Extracting audio features... (This may take a moment)'):
                    import tempfile
                    import os
                    from audio_processing import extract_features
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                        tmp_file.write(uploaded_audio.getvalue())
                        temp_path = tmp_file.name
                        
                    features_dict, error = extract_features(temp_path)
                    
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                    
                    if error:
                        st.error(f"Error during extraction: {error}")
                    else:
                        st.success("Features successfully extracted!")
                        
                        input_data = [features_dict[f] for f in features_list]
                        input_df = pd.DataFrame([input_data], columns=features_list)
                        
                        with st.expander("View Extracted Features"):
                            st.dataframe(input_df, use_container_width=True)
                        
                        scaled_data = scaler.transform(input_df)
                        pca_data = pca.transform(scaled_data)
                        prediction = model.predict(pca_data)
                        probability = model.predict_proba(pca_data)[0][1]

                        st.markdown("---")
                        st.subheader("Diagnostic Results")
                        
                        res_col1, res_col2 = st.columns([1, 1.5])
                        with res_col1:
                            if prediction[0] == 0:
                                st.success("#### Negative \nThe model indicates NO presence of Parkinson's Disease.")
                                st.metric("Confidence Level", f"{1-probability:.2%}")
                            else:
                                st.error("#### Positive \nThe model indicates the PRESENCE of Parkinson's Disease.")
                                st.metric("Confidence Level", f"{probability:.2%}")

                        with res_col2:
                            feature_importance = abs(pca.components_[0])
                            fig = go.Figure(data=[go.Bar(
                                x=features_list, 
                                y=feature_importance,
                                marker_color='#ff6e40'
                            )])
                            fig.update_layout(
                                title="Feature Importance Analysis",
                                xaxis_title="", 
                                yaxis_title="Impact",
                                margin=dict(l=0, r=0, t=30, b=0),
                                height=250
                            )
                            st.plotly_chart(fig, use_container_width=True)

# FAQs Page
elif page == "FAQs":
    st.title("Frequently Asked Questions & Facts")
    
    # Facts in cards first
    st.subheader("Key Demographic Facts")
    col1, col2 = st.columns(2)
    with col1:
        st.info("**Male to Female ratio**  \n2.66 : 1")
    with col2:
        st.info("**Average age of onset**  \n57.73 years")

    st.markdown("---")
    st.subheader("FAQs")

    faqs = [
        ("What is Parkinson's Disease?",
         "Parkinson's Disease is a neurodegenerative disorder that primarily affects movement. It can lead to tremors, stiffness, and difficulty with balance and coordination."),
        ("How can I use this app?",
         "Navigate to the prediction page using the sidebar. You can manually input features, upload a CSV file with batch data, or upload an audio file to extract acoustic features instantly."),
        ("What are the core symptoms?",
         "Common symptoms include: tremors or shaking, stiffness in limbs, slowness of movement, balance problems, and changes in speech or writing."),
        ("Is there a definitive cure?",
         "Currently, there is no cure for Parkinson's Disease, but numerous treatments are available to manage symptoms effectively. Early diagnosis significantly improves the quality of life."),
        ("Where can I learn more?",
         "For more information, consider visiting: \n- [Parkinson's Foundation](https://www.parkinson.org) \n- [Mayo Clinic](https://www.mayoclinic.org/diseases-conditions/parkinsons-disease) \n- [National Institute of Neurological Disorders](https://www.ninds.nih.gov)")
    ]

    for question, answer in faqs:
        with st.expander(f"**{question}**"):
            st.write(answer)
