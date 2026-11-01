import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Page configuration must come before other Streamlit commands.
st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="📊",
    layout="wide",
)

# Load the trained model.
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "employee_attrition_logistic_regression.joblib"
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "Trained model not found. Check that the file exists at "
        "models/employee_attrition_logistic_regression.joblib."
    )
    st.stop()
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()


# Application heading.

st.sidebar.title("About This Project")
st.sidebar.write(
"This application uses a Machine Learning model "
"to estimate the likelihood of employee attrition."
)
st.sidebar.markdown("---")
st.sidebar.subheader("Model Information")
st.sidebar.write("Algorithm: Logistic Regression")
st.sidebar.write("Dataset: IBM HR Analytics Employee Attrition")
st.sidebar.info(
"This project is for educational and demonstration purposes."
)

st.title("📊 Employee Attrition Prediction")
st.write(
"Enter employee details below to estimate the likelihood "
"of attrition using a trained Machine Learning model."
)
st.markdown("---")

st.write(
    "Enter employee information to estimate attrition likelihood "
    "using the trained machine-learning model."
)
st.info(
    "This is a demonstration using the IBM HR Analytics dataset. "
    "It should not be used as the sole basis for employment decisions."
)

st.header("Employee Information")

# All input widgets and the submit button belong inside this form.
with st.form("employee_details_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=65, value=30)
        daily_rate = st.number_input(
            "Daily Rate",
            min_value=100,
            max_value=1500,
            value=802,
            step=1
        )

        business_travel = st.selectbox(
            "Business Travel",
            ["Travel_Rarely", "Travel_Frequently", "Non-Travel"],
        )
        department = st.selectbox(
            "Department",
            ["Research & Development", "Sales", "Human Resources"],
        )
        distance_from_home = st.number_input(
            "Distance From Home", min_value=1, max_value=29, value=5
        )
        education = st.selectbox("Education Level", [1, 2, 3, 4, 5])
        education_field = st.selectbox(
            "Education Field",
            [
                "Life Sciences",
                "Medical",
                "Marketing",
                "Technical Degree",
                "Other",
                "Human Resources",
            ],
        )
        environment_satisfaction = st.selectbox(
            "Environment Satisfaction", [1, 2, 3, 4]
        )
        gender = st.selectbox("Gender", ["Female", "Male"])

    with col2:
        hourly_rate = st.number_input(
            "Hourly Rate", min_value=30, max_value=100, value=65
        )
        job_involvement = st.selectbox("Job Involvement", [1, 2, 3, 4])
        job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
        job_role = st.selectbox(
            "Job Role",
            [
                "Healthcare Representative",
                "Human Resources",
                "Laboratory Technician",
                "Manager",
                "Manufacturing Director",
                "Research Director",
                "Research Scientist",
                "Sales Executive",
                "Sales Representative",
            ],
        )
        job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
        marital_status = st.selectbox(
            "Marital Status", ["Single", "Married", "Divorced"]
        )
        monthly_income = st.number_input(
            "Monthly Income", min_value=1000, max_value=20000,
            value=5000, step=100
        )
        monthly_rate = st.number_input(
            "Monthly Rate", min_value=2000, max_value=27000,
            value=14000, step=100
        )

    with col3:
        num_companies_worked = st.number_input(
            "Number of Companies Worked", min_value=0, max_value=9, value=2
        )
        overtime = st.selectbox("OverTime", ["No", "Yes"])
        percent_salary_hike = st.number_input(
            "Percent Salary Hike", min_value=11, max_value=25, value=15
        )
        performance_rating = st.selectbox("Performance Rating", [3, 4])
        relationship_satisfaction = st.selectbox(
            "Relationship Satisfaction", [1, 2, 3, 4]
        )
        stock_option_level = st.selectbox("Stock Option Level", [0, 1, 2, 3])
        total_working_years = st.number_input(
            "Total Working Years", min_value=0, max_value=40, value=8
        )
        training_times_last_year = st.number_input(
            "Training Times Last Year", min_value=0, max_value=6, value=3
        )

    st.subheader("Work Experience and Balance")
    col4, col5, col6 = st.columns(3)

    with col4:
        work_life_balance = st.selectbox("Work-Life Balance", [1, 2, 3, 4])
        years_at_company = st.number_input(
            "Years at Company", min_value=0, max_value=40, value=5
        )

    with col5:
        years_in_current_role = st.number_input(
            "Years in Current Role", min_value=0, max_value=18, value=3
        )
        years_since_last_promotion = st.number_input(
            "Years Since Last Promotion", min_value=0, max_value=15, value=1
        )

    with col6:
        years_with_curr_manager = st.number_input(
            "Years With Current Manager", min_value=0, max_value=17, value=3
        )

    submitted = st.form_submit_button(
        "Predict Employee Attrition",
        type="primary",
        use_container_width=True,
    )


# Generate a prediction only after the form is submitted.
if submitted:
    employee_data = pd.DataFrame(
        [
            {
                "Age": age,
                "BusinessTravel": business_travel,
                "DailyRate": daily_rate,
                "Department": department,
                "DistanceFromHome": distance_from_home,
                "Education": education,
                "EducationField": education_field,
                "EnvironmentSatisfaction": environment_satisfaction,
                "Gender": gender,
                "HourlyRate": hourly_rate,
                "JobInvolvement": job_involvement,
                "JobLevel": job_level,
                "JobRole": job_role,
                "JobSatisfaction": job_satisfaction,
                "MaritalStatus": marital_status,
                "MonthlyIncome": monthly_income,
                "MonthlyRate": monthly_rate,
                "NumCompaniesWorked": num_companies_worked,
                "PercentSalaryHike": percent_salary_hike,
                "PerformanceRating": performance_rating,
                "RelationshipSatisfaction": relationship_satisfaction,
                "StockOptionLevel": stock_option_level,
                "TotalWorkingYears": total_working_years,
                "TrainingTimesLastYear": training_times_last_year,
                "WorkLifeBalance": work_life_balance,
                "YearsAtCompany": years_at_company,
                "YearsInCurrentRole": years_in_current_role,
                "YearsSinceLastPromotion": years_since_last_promotion,
                "YearsWithCurrManager": years_with_curr_manager,
                "OverTime": overtime,
            }
        ]
    )

    try:
        expected_columns = (
            model.named_steps["preprocessor"].feature_names_in_
        )
        employee_data = employee_data.loc[:, list(expected_columns)]

        prediction = int(model.predict(employee_data)[0])
        probabilities = model.predict_proba(employee_data)[0]
        attrition_probability = float(probabilities[1])


        st.divider()
        st.header("📊 Prediction Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            if prediction == 1:
                st.error("⚠️ Higher likelihood of attrition")
                st.write(
                    "The model estimates that this employee may have "
                    "a higher likelihood of leaving."
                )
            else:
                st.success("✅ Lower likelihood of attrition")
                st.write(
                    "The model estimates a lower likelihood of this "
                    "employee leaving."
                )

        with result_col2:
            st.metric(
                label="Estimated Attrition Probability",
                value=f"{attrition_probability:.1%}",
            )
            st.progress(attrition_probability)

        st.caption(
            "Important: This is a machine-learning estimate, not a "
            "certainty. The model can make incorrect predictions and "
            "should not be used as the sole basis for employment decisions."
        )

    except Exception as error:
        st.error(f"Prediction could not be completed: {error}")