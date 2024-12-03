import pandas as pd
import logging
import uvicorn
import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import dotenv_values
import numpy as np

class CancerData(BaseModel):
    Gender: int
    Age: int
    BMI: float
    GeneticRisk: int
    PhysicalActivity: int
    AlcoholIntake: float
    Diagnosis: int
    CancerHistory: int

class CancerDataBackend:
    def __init__(self):
        self.setup_logging()
        self.app = FastAPI()
        self.setup_routes()
        self.PATH = dotenv_values('.env')['DATA_PATH']
        
    async def revert_to_initial_data(self):
        try:
            df = await self.load_data()
            initial_max_id = df["Person ID"].min()
            df = df[df["Person ID"] <= initial_max_id]
            await self.save_data(df)
            return {"message": "Data reverted to initial state.", "data": df.head().to_dict()}
        except Exception as e:
            self.logger.error(f"Error while reverting data: {e}")
            return {"error": str(e)}

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def setup_routes(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"]
        )

        @self.app.get("/")
        async def home():
            return {"message": "Welcome to the Cancer Data API"}

        @self.app.post("/api/submit/")
        async def submit_data(data: CancerData):
            return await self.submit_data(data)

        @self.app.get("/api/clean_data/")
        async def clean_data():
            return await self.clean_data()

        @self.app.get("/api/data/")
        async def get_combined_data():
            return await self.get_combined_data()
        
        @self.app.get("/api/revert_to_initial/")
        async def revert_to_initial():
            return await self.revert_to_initial_data()

    async def load_data(self):
        if not os.path.exists(self.PATH):
            raise FileNotFoundError("Data file not found.")
        return pd.read_csv(self.PATH)

    async def save_data(self, df: pd.DataFrame):
        df.to_csv(self.PATH, index=False)

    async def clean_data(self):
        try:
            df = await self.load_data()
            self.logger.info("Original data loaded.")
            df = df.dropna(how="all")
            self.logger.info("Rows with all NaN values removed.")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df = df[df[numeric_cols].notna().all(axis=1)]
            self.logger.info("Rows with NaN values in numeric columns removed.")
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            self.logger.info("Inf values replaced with None.")
            await self.save_data(df)
            self.logger.info("Cleaned data saved.")
            return {"message": "Data cleaned successfully.", "data": df.head().to_dict()}
        except Exception as e:
            self.logger.error(f"Error while cleaning data: {e}")
            return {"error": str(e)}

    async def submit_data(self, data: CancerData):
        try:
            df = await self.load_data()
            last_id = df["Person ID"].max() if "Person ID" in df.columns else 0
            new_id = last_id + 1
            new_data = {
                "Person ID": int(new_id),
                "Gender": data.Gender,
                "Age": data.Age,
                "BMI": data.BMI,
                "GeneticRisk": data.GeneticRisk,
                "PhysicalActivity": data.PhysicalActivity,
                "AlcoholIntake": data.AlcoholIntake,
                "Diagnosis": data.Diagnosis,
                "CancerHistory": data.CancerHistory,
            }
            self.logger.info(f"New data: {new_data}")
            df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
            await self.save_data(df)
            return {"message": "Data submitted successfully!", "data": new_data}
        except Exception as e:
            self.logger.error(f"Error while submitting data: {e}")
            return {"error": str(e)}

    async def get_combined_data(self):
        try:
            df = await self.load_data()
            self.logger.info("Data loaded successfully.")
            df = df.dropna(how="all")
            df = df[df['Age'] > 0]
            return df.to_dict(orient="records")
        except Exception as e:
            self.logger.error(f"Error while loading data: {e}")
            return {"error": str(e)}

    def run(self):
        uvicorn.run(self.app, host="0.0.0.0", port=8000)

# Create the backend instance and expose the app
api = CancerDataBackend()
app = api.app