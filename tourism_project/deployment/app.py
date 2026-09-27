"""
Streamlit app: Wellness Tourism Package purchase predictor
----------------------------------------------------------
Loads the trained model that the GitHub Actions pipeline committed to this
folder, collects customer details from the sales agent, puts them into a
single-row DataFrame and shows the purchase probability.
"""

# ============================== NOTES ======================================
# - Loads best_tourism_model.joblib, the model the GitHub Actions pipeline
#   committed to tourism_project/deployment/.
# - Every input is collected into a one-row pandas DataFrame with exactly the
#   same columns as the training data, then passed to model.predict_proba().
# - Tested headlessly with streamlit.testing (AppTest): the app loads, builds
#   the DataFrame and returns a prediction without errors.
# ===========================================================================

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "best_tourism_model.joblib"
METADATA_PATH = APP_DIR / "model_metadata.json"

st.set_page_config(page_title="Wellness Package Predictor", page_icon="🌿", layout="wide")


@st.cache_resource
def load_model():
    """Load the model committed to the repo (cached across reruns)."""
    model = joblib.load(MODEL_PATH)
    metadata = {}
    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text())
    return model, metadata


try:
    model, metadata = load_model()
except FileNotFoundError:
    st.error(
        "Model file not found. Run the GitHub Actions pipeline so it commits "
        "`best_tourism_model.joblib` to tourism_project/deployment/."
    )
    st.stop()

threshold = float(metadata.get("classification_threshold", 0.5))

st.title("🌿 Wellness Tourism Package: Purchase Predictor")
st.write(
    "Visit with Us: enter a customer's profile and pitch details to estimate "
    "how likely they are to buy the Wellness Tourism Package **before** the "
    "sales team contacts them."
)

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Customer profile")
    age = st.number_input("Age", min_value=18, max_value=90, value=35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital status", ["Married", "Single", "Unmarried", "Divorced"])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly income", min_value=0.0, max_value=200000.0, value=23000.0, step=500.0)

with col2:
    st.subheader("Travel details")
    city_tier = st.selectbox("City tier", [1, 2, 3], help="Tier 1 > Tier 2 > Tier 3")
    num_persons = st.number_input("Number of persons visiting", min_value=1, max_value=10, value=3)
    num_children = st.number_input("Children under 5 visiting", min_value=0, max_value=5, value=1)
    preferred_star = st.selectbox("Preferred property star", [3, 4, 5])
    num_trips = st.number_input("Trips per year (average)", min_value=0, max_value=30, value=3)
    passport = st.radio("Holds a valid passport?", ["Yes", "No"], horizontal=True)
    own_car = st.radio("Owns a car?", ["Yes", "No"], horizontal=True)

with col3:
    st.subheader("Sales interaction")
    type_of_contact = st.selectbox("Type of contact", ["Self Enquiry", "Company Invited"])
    product_pitched = st.selectbox("Product pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
    duration_of_pitch = st.number_input("Duration of pitch (minutes)", min_value=0, max_value=180, value=15)
    num_followups = st.number_input("Number of follow-ups", min_value=0, max_value=10, value=4)
    pitch_score = st.slider("Pitch satisfaction score", min_value=1, max_value=5, value=3)

# ---------------------------------------------------------------------------
# Save the inputs into a DataFrame (same columns as the training data)
# ---------------------------------------------------------------------------
input_df = pd.DataFrame([{
    "Age": float(age),
    "TypeofContact": type_of_contact,
    "CityTier": int(city_tier),
    "DurationOfPitch": float(duration_of_pitch),
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": int(num_persons),
    "NumberOfFollowups": float(num_followups),
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": float(preferred_star),
    "MaritalStatus": marital_status,
    "NumberOfTrips": float(num_trips),
    "Passport": 1 if passport == "Yes" else 0,
    "PitchSatisfactionScore": int(pitch_score),
    "OwnCar": 1 if own_car == "Yes" else 0,
    "NumberOfChildrenVisiting": float(num_children),
    "Designation": designation,
    "MonthlyIncome": float(monthly_income),
}])

with st.expander("Input data sent to the model"):
    st.dataframe(input_df, use_container_width=True)

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
if st.button("Predict purchase likelihood", type="primary"):
    probability = float(model.predict_proba(input_df)[0, 1])
    will_buy = probability >= threshold

    st.metric("Purchase probability", f"{probability:.1%}")
    st.progress(min(max(probability, 0.0), 1.0))
    if will_buy:
        st.success("Likely buyer: prioritise this customer for the Wellness package pitch.")
    else:
        st.warning("Unlikely buyer: consider a different package or a lower-cost contact channel.")
    st.caption(f"Decision threshold: {threshold:.2f}")

if metadata.get("test_metrics"):
    m = metadata["test_metrics"]
    st.divider()
    st.caption(
        f"Model test performance: F1 {m['f1']:.3f} · Recall {m['recall']:.3f} · "
        f"Precision {m['precision']:.3f} · ROC-AUC {m['roc_auc']:.3f}"
    )
