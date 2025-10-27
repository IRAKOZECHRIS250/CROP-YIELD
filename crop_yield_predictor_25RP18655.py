# Create the CORRECT Streamlit app file - this should be the content of crop_yield_predictor_25RP18655.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Crop Yield Predictor",
    page_icon="🌱",
    layout="wide"
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
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🎯 Make Predictions")
    
    # Input method selection
    input_method = st.radio(
        "Select input method:",
        ["Single Temperature", "Multiple Temperatures", "Temperature Range"],
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
                
    elif input_method == "Multiple Temperatures":
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
                    
                    # Visualization
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(temperatures, predictions, color='green', s=100, alpha=0.7, label='Predictions')
                    ax.plot(temperatures, predictions, 'r--', alpha=0.5, linewidth=2)
                    ax.set_xlabel("Temperature (°C)", fontsize=12)
                    ax.set_ylabel("Predicted Crop Yield", fontsize=12)
                    ax.set_title("Crop Yield vs Temperature Predictions", fontsize=14)
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    
                    st.pyplot(fig)
                    
                except ValueError:
                    st.error("❌ Please enter valid numbers separated by commas.")
    
    else:  # Temperature Range
        st.subheader("Temperature Range Analysis")
        
        col_a, col_b = st.columns(2)
        with col_a:
            min_temp = st.number_input("Minimum Temperature (°C)", value=15.0, min_value=10.0, max_value=35.0)
        with col_b:
            max_temp = st.number_input("Maximum Temperature (°C)", value=30.0, min_value=15.0, max_value=40.0)
        
        num_points = st.slider("Number of points", 5, 20, 10)
        
        if st.button("Generate Predictions", type="primary"):
            if model is not None and min_temp < max_temp:
                # Generate temperature range
                temperatures = np.linspace(min_temp, max_temp, num_points)
                
                # Create predictions
                input_data = pd.DataFrame({"Temperature": temperatures})
                predictions = model.predict(input_data)
                
                # Create results
                results_df = pd.DataFrame({
                    "Temperature (°C)": temperatures,
                    "Predicted Yield": predictions
                })
                
                # Display results
                st.subheader("📈 Range Analysis Results")
                st.dataframe(
                    results_df.style.format({"Temperature (°C)": "{:.1f}", "Predicted Yield": "{:.2f}"}),
                    use_container_width=True
                )
                
                # Enhanced visualization
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                
                # Line plot
                ax1.plot(temperatures, predictions, 'b-o', linewidth=2, markersize=6)
                ax1.set_xlabel("Temperature (°C)")
                ax1.set_ylabel("Predicted Crop Yield")
                ax1.set_title("Crop Yield vs Temperature")
                ax1.grid(True, alpha=0.3)
                
                # Bar plot
                ax2.bar(range(len(temperatures)), predictions, color='lightgreen', alpha=0.7)
                ax2.set_xlabel("Data Point Index")
                ax2.set_ylabel("Predicted Crop Yield")
                ax2.set_title("Yield Distribution")
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
            else:
                st.error("❌ Minimum temperature must be less than maximum temperature.")

with col2:
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
        sample_temps = [20, 25, 30]
        for temp in sample_temps:
            pred = model.predict(pd.DataFrame({"Temperature": [temp]}))[0]
            st.write(f"**{temp}°C** → {pred:.2f} units")

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