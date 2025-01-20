import logging
import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from src.api.routes import config, router

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

#
# async def main():
#     try:
#         # Initialize application
#         load_dotenv()
#         app = LoanStrategyApp()
#         data_generator = DataGenerator()
#         # data_generator.generate_csv_data()
#         # Load loan data
#         logger.info("Loading loan data...")
#         loan_data = pd.read_csv("data/loan_applications.csv")
#
#         # Analyze specific pincode
#         pincode = '110080'
#         logger.info(f"Analyzing pincode {pincode}...")
#         results = await app.analyze_pincode(loan_data, pincode)
#
#         # Save results
#         output_path = "output/analysis_results.yaml"
#         logger.info(f"Saving results to {output_path}...")
#
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)
#         with open(output_path, 'w') as f:
#             yaml.dump(results, f, default_flow_style=False)
#
#     except Exception as e:
#         logger.error(f"Error in main execution: {e}")
#         raise

# Initialize the FastAPI app
app = FastAPI(
    title="Loan Strategy Analyzer API",
    description="API for analyzing loan data and generating marketing strategies",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config['api']['cors_origins'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Include API routes
app.include_router(router, prefix="/api")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # models = openai.models.list()
    #
    # # Print the model names
    # for model in models['data']:
    #     print(model['id'])
    # print(f"open api key: {api_key}")
    # asyncio.run(main())
