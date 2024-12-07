import streamlit as st
import pandas as pd
import logging
import plotly.express as px
import plotly.graph_objects as go
import utils  # Importing utils.py module
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class App:
    def __init__(self):
        """Initialize the application."""
        # Fetch file path from .env file
        self.file_path = os.getenv("DATA_PATH")
        if not self.file_path:
            st.error("Error: DATA_PATH not set in .env file.")
            return
        self.df = self.load_data()
        if self.df is None:  # Ensure df is not None
            self.df = pd.DataFrame()

    def load_data(self):
        """Load data from CSV file."""
        try:
            data = pd.read_csv(self.file_path)
            if 'Age' not in data.columns or 'Gender' not in data.columns:
                raise ValueError("Missing required columns: 'Age' and 'Gender'")
            st.success("Data loaded successfully!")
            return data
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None

    def save_data(self):
        """Save the updated data to the CSV file."""
        if not self.df.empty:  # Check if the dataframe is not empty
            self.df.to_csv(self.file_path, index=False)
        else:
            st.warning("No data to save.")
   
    def clean_data(self):
        """Clean the data by making an API request."""
        try:
            response = requests.get("http://127.0.0.1:8000/api/clean_data/")
            if response.status_code == 200:
                self.logger.info("Data successfully cleared.")
                st.success("Data successfully cleared.")
            else:
                self.logger.error("An error occurred while cleaning data.")
                st.error("An error occurred while cleaning data.")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"An error occurred while connecting to the server: {e}")
            st.error(f"An error occurred while connecting to the server: {e}")
    
    
    def display_home(self):
        """Home Page with a greeting and basic data information."""
        st.header("Welcome to the Cancer Data Analysis App!")
        st.subheader("Greetings!")
        st.write(
            """
            Welcome to the Cancer Data Analysis application! Here, we analyze data related to cancer diagnosis,
            including various factors like age, BMI, genetic risk, physical activity, and more.
            
            You can navigate to different sections using the sidebar. Add new records, view graphs, and 
            explore data-driven insights to help in understanding cancer risk and diagnosis.
            """
        )
        st.subheader("Data Overview:")
        if not self.df.empty:
            st.write(self.df.head())  # Display first few records from the data
        else:
            st.warning("No data loaded. Please add some records.")
 
    def validate_data(self, data):
        """Validate data before adding it."""
        if not (0 <= data['Age'] <= 120):
            return False, "Age must be between 0 and 120."
        if data['Gender'] not in [0, 1]:
            return False, "Gender must be 0 (Male) or 1 (Female)."
        if not (0 <= data['BMI'] <= 100):
            return False, "BMI must be between 0 and 100."
        if not (0 <= data['GeneticRisk'] <= 10):
            return False, "Genetic Risk must be between 0 and 10."
        if not (0 <= data['PhysicalActivity'] <= 10):
            return False, "Physical Activity must be between 0 and 10."
        if not (0.0 <= data['AlcoholIntake'] <= 10.0):
            return False, "Alcohol Intake must be between 0.0 and 10.0."
        if data['Diagnosis'] not in [0, 1]:
            return False, "Diagnosis must be 0 (Healthy) or 1 (Cancer)."
        if data['CancerHistory'] not in [0, 1]:
            return False, "Cancer History must be 0 (No) or 1 (Yes)."
        return True, ""

    def display_graphs(self, graph_generator):
        """Display graphs on Streamlit with detailed comments."""
        st.header("Data Analysis Graphs")
        
        # Diagnosis Distribution Chart
        st.subheader("Diagnosis Distribution")
        st.write("This pie chart shows the proportion of patients diagnosed with cancer versus those diagnosed as healthy.")
        st.plotly_chart(graph_generator.generate_diagnosis_chart(), use_container_width=True)

        # Gender Distribution Chart
        st.subheader("Gender Distribution")
        st.write("This donut chart illustrates the gender distribution among patients, showing the proportion of male and female participants.")
        st.plotly_chart(graph_generator.generate_gender_chart(), use_container_width=True)

        # Histogram by Age and Gender
        st.subheader("Age Distribution by Gender")
        st.write("This histogram presents the age distribution of patients, separated by gender. You can observe the age range and differences between male and female participants.")
        histogram_fig = graph_generator.generate_histogram_by_gender(x_var="Age", groupby_var="Gender")
        if histogram_fig:
            st.plotly_chart(histogram_fig, use_container_width=True)

        # Correlation Matrix
        st.subheader("Correlation Matrix")
        st.write("This heatmap represents the Pearson correlation coefficients between different features in the dataset. Positive correlations are shown in red, and negative correlations in blue.")
        st.plotly_chart(graph_generator.generate_correlation_matrix(), use_container_width=True)

        # 3D Scatter Plot
        st.subheader("3D Scatter Plot: Physical Activity, BMI, and Alcohol Intake")
        st.write("This 3D scatter plot visualizes the relationship between physical activity, BMI, and alcohol intake, with colors indicating cancer diagnosis.")
        st.plotly_chart(graph_generator.generate_3d_scatter_plot(), use_container_width=True)

        # Age vs BMI Plot
        
        st.write(
            """
            This scatter plot illustrates the relationship between physical activity and BMI of patients. 
            The color coding distinguishes between healthy individuals (blue) and cancer-diagnosed patients (red). 
            Observing the plot, we can identify whether individuals with lower physical activity levels tend to have higher BMI values.
            """
        )
        st.plotly_chart(graph_generator.generate_physical_activity_vs_bmi_scatter_plot(), use_container_width=True)

        
        
        st.write(
            """
            This scatter plot shows how physical activity levels are related to alcohol intake among patients. 
            The plot is color-coded to represent the diagnosis status: healthy (blue) and cancer-diagnosed (red). 
            The visualization can help assess if individuals with low physical activity levels also exhibit higher alcohol consumption.
            """
        )
        st.plotly_chart(graph_generator.generate_alcohol_intake_histogram(), use_container_width=True)

        
        st.write(
            """
            This scatter plot explores the relationship between BMI and alcohol intake. 
            The data points are color-coded to differentiate between healthy individuals (blue) and those diagnosed with cancer (red). 
            This chart helps identify whether higher alcohol consumption correlates with increased BMI among the studied groups.
            """
        )
        st.plotly_chart(graph_generator.generate_bmi_box_plot(), use_container_width=True)
        

        
        # Genetic Risk Chart
        st.subheader("Genetic Risk Distribution")
        st.write("This line chart displays the distribution of genetic risk among patients, with additional breakdowns based on cancer history and diagnosis.")
        st.plotly_chart(graph_generator.generate_genetic_risk_chart(), use_container_width=True)


    def display_hypothesis_section(self, graph_generator):
        """Displays hypothesis and related graph."""
        st.subheader("Hypothesis")
        
        description = """
        We hypothesize that patients with a high genetic risk (GeneticRisk ≥ 2), low physical activity (PhysicalActivity < 5), 
        and a history of cancer (CancerHistory = 1) have a significantly higher risk of being diagnosed with cancer (Diagnosis = 1).
        In testing this hypothesis, we identified a high-risk group meeting these criteria, and calculated the proportion of cancer diagnoses 
        in this group compared to the rest of the patients. In the high-risk group, cancer was diagnosed in 100% of patients, whereas 
        it was only present in 36.8% of the rest. This significant difference indicates a strong relationship between the identified factors 
        (high genetic risk, low physical activity, and cancer history) and the current cancer diagnosis.
        """
        
        st.write(description)
        
        # Ensure using the graph_generator instance to call plot_hypothesis_bar_chart
        st.plotly_chart(graph_generator.plot_hypothesis_bar_chart(), use_container_width=True)

    def display_form(self):
        """Form to add a new record."""
        st.header("Add New Record")
        st.write("Please fill in the form below to add a new record. Each field includes a description of the expected value.")
        
        with st.form("data_entry_form"):
            # Age
            st.markdown("**Age**: Enter the patient's age in years (0-120).")
            age = st.number_input("Age", min_value=0, max_value=120, step=1)
            
            # Gender
            st.markdown("**Gender**: Specify the patient's gender (0 = Male, 1 = Female).")
            gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 1 else "Male")
            
            # BMI
            st.markdown("**Body Mass Index (BMI)**: Enter the patient's BMI, a measure of body fat based on height and weight (0.0 - 100.0).")
            bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, max_value=100.0, step=0.1)
            
            # Genetic Risk
            st.markdown("**Genetic Risk**: Rate the genetic predisposition to cancer on a scale from 0 (no risk) to 10 (high risk).")
            genetic_risk = st.number_input("Genetic Risk", min_value=0, max_value=10, step=1)
            
            # Physical Activity
            st.markdown("**Physical Activity**: Rate the patient's physical activity on a scale from 0 (no activity) to 10 (very active).")
            physical_activity = st.number_input("Physical Activity", min_value=0, max_value=10, step=1)
            
            # Alcohol Intake
            st.markdown("**Alcohol Intake**: Enter the patient's average alcohol intake per day on a scale from 0.0 (none) to 10.0 (high intake).")
            alcohol_intake = st.number_input("Alcohol Intake", min_value=0.0, max_value=10.0, step=0.1)
            
            # Diagnosis
            st.markdown("**Diagnosis**: Specify the diagnosis (0 = Healthy, 1 = Cancer).")
            diagnosis = st.selectbox("Diagnosis", options=[0, 1], format_func=lambda x: "Cancer" if x == 1 else "Healthy")
            
            # Cancer History
            st.markdown("**Cancer History**: Specify whether the patient has a history of cancer (0 = No, 1 = Yes).")
            cancer_history = st.selectbox("Cancer History", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            
            # Submit Button
            submitted = st.form_submit_button("Add Record")
            if submitted:
                new_record = {
                    "Age": age,
                    "Gender": gender,
                    "BMI": bmi,
                    "GeneticRisk": genetic_risk,
                    "PhysicalActivity": physical_activity,
                    "AlcoholIntake": alcohol_intake,
                    "Diagnosis": diagnosis,
                    "CancerHistory": cancer_history
                }
                
                # Validate data
                is_valid, error_message = self.validate_data(new_record)
                if not is_valid:
                    st.error(f"Invalid data: {error_message}")
                else:
                    # Adding new record using pd.concat() instead of append
                    self.df = pd.concat([self.df, pd.DataFrame([new_record])], ignore_index=True)
                    
                    # Save the updated data
                    self.save_data()
                    
                    st.success("Record successfully added!")
                    
                    # Regenerate graphs with new data
                    graph_generator = utils.CancerDataAnalysis(self.df)
                    self.display_graphs(graph_generator)


    def run(self):
        """Run the Streamlit app."""
        try:
            logging.info("App running...")
            st.sidebar.title("Navigation")
            menu = st.sidebar.radio("Go to", ["Home", "Graphs", "Add Record"])
            
            if menu == "Home":
                self.display_home()
            elif menu == "Graphs":
                if self.df is None or self.df.empty:
                    st.error("No data available to generate graphs.")
                    return  # If no data is available, don't proceed to graph generation
                
                logging.info("Data loaded.")
                graph_generator = utils.CancerDataAnalysis(self.df)
                logging.info("Generating graphs...")
                self.display_graphs(graph_generator)
                logging.info("Graphs displayed.")
                
                self.display_hypothesis_section(graph_generator)  # Display hypotheses
                logging.info("Hypotheses displayed.")
                
            elif menu == "Add Record":
                self.display_form()

        except Exception as e:
            logging.error(f"Error: {e}")
            st.error(f"An error occurred: {e}")


# Run the app
if __name__ == "__main__":
    app = App()
    app.run()
