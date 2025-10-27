# crop_yield_predictor_25RP18655.py (Simplified version - No Matplotlib)
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page configuration
st.set_page_config(
    page_title="Crop Yield Predictor",
    page_icon="🌱",
    layout="centered"
)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load("Linear_Regression_25RP18655.pkl")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# App title and description
st.title("🌱 Crop Yield Prediction App")
st.markdown("**Developer:** IRAKOZE Chrispin | **Registration Number:** 25RP18655")
st.markdown("---")

# Sidebar for navigation and info
st.sidebar.header("ℹ️ App Information")
st.sidebar.markdown("""
**Machine Learning Exam Project**
- **Model:** Linear Regression
- **Input:** Temperature Data
- **Output:** Crop Yield Prediction
- **Purpose:** Smart Agriculture Decision Support
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**Built with:** Python, Scikit-learn, Streamlit")

# Main content
st.header("🎯 Make Predictions")

# Input method selection
input_method = st.radio(
    "Select input method:",
    ["Single Temperature", "Multiple Temperatures"],
    horizontal=True
)

if input_method == "Single Temperature":
    st.subheader("Single Prediction")
    temperature = st.slider(
        "Select Temperature (°C)",
        min_value=10.0,
        max_value=40.0,
        value=25.0,
        step=0.5,
        help="Adjust the temperature to get crop yield prediction"
    )
    
    if st.button("Predict Crop Yield", type="primary"):
        if model is not None:
            # Create input DataFrame
            input_data = pd.DataFrame({"Temperature": [temperature]})
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            
            # Display result
            st.success(f"**Predicted Crop Yield:** `{prediction:.2f} units`")
            
            # Additional context
            st.info(f"At **{temperature}°C**, the expected crop yield is approximately **{prediction:.2f} units**.")
            
else:
    st.subheader("Multiple Predictions")
    st.write("Enter temperature values separated by commas:")
    
    temp_input = st.text_input(
        "Temperatures",
        "20.0, 25.0, 30.0, 35.0",
        help="Example: 20, 25, 30, 35"
    )
    
    if st.button("Predict Multiple Yields", type="primary"):
        if model is not None:
            try:
                # Parse input temperatures
                temperatures = [float(x.strip()) for x in temp_input.split(",")]
                
                # Create input DataFrame
                input_data = pd.DataFrame({"Temperature": temperatures})
                
                # Make predictions
                predictions = model.predict(input_data)
                
                # Create results DataFrame
                results_df = pd.DataFrame({
                    "Temperature (°C)": temperatures,
                    "Predicted Yield": predictions
                })
                
                # Display results
                st.subheader("📊 Prediction Results")
                st.dataframe(
                    results_df.style.format({"Predicted Yield": "{:.2f}"}),
                    use_container_width=True
                )
                
                # Simple text-based visualization
                st.subheader("📈 Results Summary")
                max_yield_idx = np.argmax(predictions)
                min_yield_idx = np.argmin(predictions)
                
                st.write(f"**Highest yield:** {predictions[max_yield_idx]:.2f} units at {temperatures[max_yield_idx]}°C")
                st.write(f"**Lowest yield:** {predictions[min_yield_idx]:.2f} units at {temperatures[min_yield_idx]}°C")
                st.write(f"**Average yield:** {np.mean(predictions):.2f} units")
                st.write(f"**Yield range:** {np.max(predictions) - np.min(predictions):.2f} units")
                
            except ValueError:
                st.error("❌ Please enter valid numbers separated by commas.")

# Model Information Section
st.header("📋 Model Information")

if model is not None:
    st.success("✅ Model loaded successfully!")
    
    # Display model details
    st.subheader("Model Specifications")
    st.markdown(f"""
    - **Algorithm:** Linear Regression
    - **Input Feature:** Temperature
    - **Target Variable:** Crop Yield
    - **Model File:** `Linear_Regression_25RP18655.pkl`
    """)
    
    # Display sample coefficients (if available)
    try:
        if hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
            st.subheader("Model Parameters")
            st.markdown(f"""
            - **Coefficient:** `{model.coef_[0]:.4f}`
            - **Intercept:** `{model.intercept_:.4f}`
            - **Equation:** `Yield = {model.coef_[0]:.4f} × Temperature + {model.intercept_:.4f}`
            """)
    except:
        pass
    
    # Quick predictions showcase
    st.subheader("🚀 Quick Predictions")
    sample_temps = [15, 20, 25, 30, 35]
    for temp in sample_temps:
        pred = model.predict(pd.DataFrame({"Temperature": [temp]}))[0]
        st.write(f"**{temp}°C** → **{pred:.2f} units**")

# Sample predictions for demonstration
st.header("🔍 Sample Predictions")
if model is not None:
    # Create a sample table
    sample_data = {
        "Temperature (°C)": [15, 20, 25, 30, 35],
        "Predicted Yield": [model.predict(pd.DataFrame({"Temperature": [x]}))[0] for x in [15, 20, 25, 30, 35]]
    }
    sample_df = pd.DataFrame(sample_data)
    sample_df["Predicted Yield"] = sample_df["Predicted Yield"].round(2)
    
    st.table(sample_df)

# Footer
st.markdown("---")
st.markdown(
    "**Machine Learning Practical Exam | Semester I | October 2025** | "
    "**University of Rwanda**"
)

# Add some sample data statistics in expander
with st.expander("📊 Sample Data Statistics"):
    st.markdown("""
    **Typical Data Ranges from Training:**
    - **Temperature:** 11.9°C to 37.3°C
    - **Crop Yield:** 24.4 to 74.4 units
    
    **Model Purpose:**
    This linear regression model helps farmers predict crop yield based on temperature
    variations, enabling better agricultural planning and resource allocation.
    """)

# Success message when everything works
if model is not None:
    st.balloons()
    st.success("🎉 App is running successfully! Take a screenshot for your submission.")