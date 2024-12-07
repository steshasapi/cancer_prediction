import pandas as pd
import numpy as np
import uuid
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

class CancerDataAnalysis:
    def __init__(self, data): 
        self.df = data

    def generate_diagnosis_chart(self):
        value_counts_diagnosis = self.df['Diagnosis'].replace({1: 'Have Cancer', 0: 'Healthy'}).value_counts()

        fig_diagnosis = go.Figure(data=[
            go.Pie(
                labels=value_counts_diagnosis.index,
                values=value_counts_diagnosis.values,
                marker=dict(colors=['#87CEEB', '#4682B4']),  
                textinfo='label+percent',  
            )
        ])

        fig_diagnosis.update_layout(
            title_text='Diagnosis Distribution',
            template='plotly_dark',
            showlegend=False
    )

        return fig_diagnosis


    def generate_gender_chart(self):
        value_counts_gender = self.df['Gender'].replace({1: 'Women', 0: 'Men'}).value_counts()
        
        fig_gender = go.Figure(data=[
            go.Pie(
                labels=value_counts_gender.index,
                values=value_counts_gender.values,
                hole=0.6,  
                marker=dict(colors=['#87CEEB', '#4682B4']), 
                textinfo='label+percent',  
                textfont=dict(color='black')  
            )
        ])

        fig_gender.update_layout(
            title_text='Gender Distribution',
            template='plotly_dark',
            showlegend=False
    )
        return fig_gender

    
    def generate_histogram_by_gender(self, x_var='Age', groupby_var='Gender'):
        df_agg = self.df.loc[:, [x_var, groupby_var]].groupby(groupby_var)
        vals = [df[x_var].values.tolist() for _, df in df_agg]

        gender_map = {0: 'Male', 1: 'Female'}  

        colors = ['#87CEEB', '#4682B4']  

        fig = go.Figure()

        for i, val in enumerate(vals):
            fig.add_trace(go.Histogram(
                x=val,
                name=gender_map.get(i, str(i)),  
                marker_color=colors[i],
                opacity=1,  
                nbinsx=40  
            ))

        fig.update_layout(
            title_text=f"Histogram of {x_var} by {groupby_var}",
            xaxis_title=x_var,
            yaxis_title="Frequency",
            barmode='group',  # Group the bars
            bargap=0.2,  # Increase the gap between bars
            legend_title="Gender",
            template="plotly_dark",  # Dark theme (change to 'plotly_white' for light theme)
            showlegend=False  # Hide the legend if not needed
        )

        return fig


    def generate_correlation_matrix(self):
        correl_mtx = self.df.corr(method='pearson')

        fig_temp = px.imshow(
            correl_mtx,
            text_auto=True,
            color_continuous_scale='RdBu',
            range_color=[-1, 1],
            labels=dict(color="Pearson's Correlation"),
            x=correl_mtx.columns,
            y=correl_mtx.index
        )

        fig_temp.update_layout(
            title={'text': "Pearson's Correlation Matrix", 'x': 0.01, 'y': 0.97},
            xaxis_title="Features",
            yaxis_title="Features",
            coloraxis_colorbar_title="Pearson's Correlation",
            width=1200,
            height=800,
            coloraxis_colorbar=dict(
            len=0.5
            ),

        )

        return fig_temp




    # Define a consistent blue color palette
    blue_palette = ["#004c6d", "#00789f", "#4ca1c3", "#a0d1e9"]

    def generate_3d_scatter_plot(self):
        """3D Scatter Plot: Physical Activity, BMI, Alcohol Intake."""
        required_columns = ["PhysicalActivity", "BMI", "AlcoholIntake", "Diagnosis"]
        if not all(col in self.df.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {', '.join(required_columns)}")
        
        data = self.df.dropna(subset=required_columns)
        fig = px.scatter_3d(
            data,
            x="PhysicalActivity",
            y="BMI",
            z="AlcoholIntake",
            color="Diagnosis",
            color_discrete_sequence=self.blue_palette,
        )
        fig.update_layout(
            template="plotly_dark",
            title=dict(
                text="3D Scatter Plot: Physical Activity, BMI, Alcohol Intake",
                font=dict(family="Arial", size=18)
            ),
            scene=dict(
                xaxis=dict(title="Physical Activity"),
                yaxis=dict(title="BMI"),
                zaxis=dict(title="Alcohol Intake"),
            ),
            font=dict(family="Arial", size=12)
        )
        return fig

    def generate_physical_activity_vs_bmi_scatter_plot(self):
        """Scatter Plot: Average BMI by Physical Activity."""
        required_columns = ["PhysicalActivity", "BMI", "Diagnosis"]
        if not all(col in self.df.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {', '.join(required_columns)}")
        
        data = self.df.dropna(subset=required_columns)
        grouped_data = data.groupby(["PhysicalActivity", "Diagnosis"], as_index=False).mean()
        
        fig = px.scatter(
            grouped_data,
            x="PhysicalActivity",
            y="BMI",
            color="Diagnosis",
            color_discrete_sequence=self.blue_palette,
            title="Average BMI by Physical Activity",
            labels={"PhysicalActivity": "Physical Activity", "BMI": "Average BMI", "Diagnosis": "Diagnosis"},
        )
        fig.update_traces(marker=dict(size=10, opacity=0.8))
        fig.update_layout(
            template="plotly_white",
            xaxis_title="Physical Activity",
            yaxis_title="Average BMI",
            font=dict(family="Arial", size=12, color="black"),
            title=dict(font=dict(family="Arial", size=18)),
        )
        return fig

    def generate_alcohol_intake_histogram(self):
        """Histogram: Alcohol Intake by Diagnosis."""
        required_columns = ["AlcoholIntake", "Diagnosis"]
        if not all(col in self.df.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {', '.join(required_columns)}")
        
        data = self.df.dropna(subset=required_columns)
        fig = px.histogram(
            data,
            x="AlcoholIntake",
            color="Diagnosis",
            barmode="overlay",
            color_discrete_sequence=self.blue_palette,
            title="Alcohol Intake Distribution by Diagnosis",
            labels={"AlcoholIntake": "Alcohol Intake"},
        )
        fig.update_layout(
            template="plotly_white",
            font=dict(family="Arial", size=12, color="black"),
            title=dict(font=dict(family="Arial", size=18)),
        )
        return fig

    def generate_bmi_box_plot(self):
        """Box Plot: BMI Distribution by Diagnosis."""
        required_columns = ["BMI", "Diagnosis"]
        if not all(col in self.df.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {', '.join(required_columns)}")
        
        data = self.df.dropna(subset=required_columns)
        fig = px.box(
            data,
            x="Diagnosis",
            y="BMI",
            color="Diagnosis",
            color_discrete_sequence=self.blue_palette,
            title="BMI Distribution by Diagnosis",
            labels={"Diagnosis": "Diagnosis (0=Healthy, 1=Cancer)", "BMI": "Body Mass Index (BMI)"},
        )
        fig.update_layout(
            template="plotly_white",
            font=dict(family="Arial", size=12, color="black"),
            title=dict(font=dict(family="Arial", size=18)),
        )
        return fig

    def generate_physical_genetic_plot(self):
        """Histogram: Physical Activity vs Genetic Risk."""
        fig = px.histogram(
            self.df,
            x="PhysicalActivity",
            y="GeneticRisk",
            color="Diagnosis",
            color_discrete_sequence=self.blue_palette[::-1],
            title="Physical Activity vs Genetic Risk",
            labels={"Diagnosis": "Diagnosis (0=Healthy, 1=Cancer)"},
            histfunc="sum",
            barmode="stack",
            marginal="box",
            category_orders={"PhysicalActivity": ["Night", "Low", "Medium", "High"]},
        )
        fig.update_layout(
            template="plotly_dark",
            xaxis=dict(title="Physical Activity", gridcolor="gray"),
            yaxis=dict(title="Sum of Genetic Risk", gridcolor="gray"),
            font=dict(family="Arial", size=12, color="white"),
            title=dict(font=dict(family="Arial", size=18, color="white")),
        )
        return fig

    def generate_genetic_risk_chart(self):
        """Line Chart: Genetic Risk with Diagnosis and Cancer History."""
        if self.df[['GeneticRisk', 'Diagnosis', 'CancerHistory']].isnull().any().any():
            self.df = self.df.dropna(subset=['GeneticRisk', 'Diagnosis', 'CancerHistory'])
        
        plot_data = self.df.groupby(['GeneticRisk', 'Diagnosis', 'CancerHistory']).size().reset_index(name='Count')
        total_counts = plot_data.groupby(['GeneticRisk', 'CancerHistory'])['Count'].transform('sum')
        plot_data['Percentage'] = (plot_data['Count'] / total_counts) * 100

        fig = go.Figure()
        line_styles = ['solid', 'dot', 'dash', 'dashdot']
        for i, cancer_history in enumerate(plot_data['CancerHistory'].unique()):
            for j, diagnosis in enumerate(plot_data['Diagnosis'].unique()):
                subset = plot_data[(plot_data['CancerHistory'] == cancer_history) &
                                   (plot_data['Diagnosis'] == diagnosis)]
                color = self.blue_palette[(i * 2 + j) % len(self.blue_palette)]
                style = line_styles[(i + j) % len(line_styles)]
                fig.add_trace(go.Scatter(
                    x=subset['GeneticRisk'],
                    y=subset['Count'],
                    mode='lines+markers',
                    name=f'Count (CancerHistory={cancer_history}, Diagnosis={diagnosis})',
                    line=dict(width=2, color=color, dash=style),
                ))
                fig.add_trace(go.Scatter(
                    x=subset['GeneticRisk'],
                    y=subset['Percentage'],
                    mode='lines+markers',
                    name=f'Percentage (CancerHistory={cancer_history}, Diagnosis={diagnosis})',
                    line=dict(width=2, color=color, dash=style),
                ))
        fig.update_layout(
            title="Genetic Risk with Diagnosis and Cancer History",
            xaxis=dict(title="Genetic Risk"),
            yaxis=dict(title="Frequency (Count)"),
            yaxis2=dict(title="Percentage (%)", overlaying='y', side='right'),
            font=dict(family="Arial", size=12),
            title_font=dict(family="Arial", size=18),
        )
        return fig

    def plot_hypothesis_bar_chart(self):
        """Bar Chart: Cancer Diagnosis Rate by Patient Group."""
        self.df['HighRiskGroup'] = (
            (self.df['GeneticRisk'] >= 2) &
            (self.df['PhysicalActivity'] < 5) &
            (self.df['CancerHistory'] == 1)
        )
        groups = self.df.groupby('HighRiskGroup')['Diagnosis'].mean().reset_index()
        groups['Group'] = groups['HighRiskGroup'].map({True: 'High Risk', False: 'Other Patients'})

        fig = go.Figure()
        colors = self.blue_palette[:2]
        for i, group in groups.iterrows():
            fig.add_trace(go.Bar(
                x=[group['Group']],
                y=[group['Diagnosis']],
                name=group['Group'],
                marker_color=colors[i],
                text=f"{group['Diagnosis'] * 100:.2f}%",
                textposition='auto'
            ))
        fig.update_layout(
            title="Cancer Diagnosis Rate by Patient Group",
            xaxis_title="Patient Group",
            yaxis_title="Cancer Diagnosis Rate",
            font=dict(family="Arial", size=12, color="black"),
            title_font=dict(family="Arial", size=18),
        )
        return fig


def generate_key(self, prefix="chart"):
    return f"{prefix}_{uuid.uuid4().hex}"
