import logging
from faker import Faker
import random
import pandas as pd
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class DataGeneratorError(Exception):
    """Custom exception for DataGenerator errors"""
    pass


class DataGenerator:
    MIN_LOAN_AMOUNT = 100000
    MAX_LOAN_AMOUNT = 1000000
    BATCH_SIZE = 5000
    MIN_RECORDS_DEFAULT = 50
    MAX_ADDITIONAL_RECORDS = 30
    DEFAULT_OUTPUT_DIR = "data"
    MAX_WORKERS = 4

    def __init__(self, min_records_per_pincode: int = MIN_RECORDS_DEFAULT,
                 output_dir: str = DEFAULT_OUTPUT_DIR,
                 seed: Optional[int] = None):
        try:
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed)

            self.fake = Faker()
            if seed is not None:
                Faker.seed(seed)

            self.min_records_per_pincode = max(self.MIN_RECORDS_DEFAULT, min_records_per_pincode)
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.end_date = datetime.now()
            self.start_date = self.end_date - timedelta(days=5 * 365)
            self.selected_pincodes = self._generate_delhi_ncr_pincodes()
            logger.info(f"Generated {len(self.selected_pincodes)} Delhi NCR pincodes")
            self._load_distributions()
        except Exception as e:
            raise DataGeneratorError(f"Initialization failed: {e}") from e

    def _load_distributions(self):
        self.STATUSES = {
            "Approved": 0.20,
            "Rejected": 0.15,
            "Inquired": 0.10,
            "Closed": 0.40,
            "Ongoing": 0.10,
            "Defaulted (NPA)": 0.05
        }
        if not np.isclose(sum(self.STATUSES.values()), 1.0, rtol=1e-5):
            raise DataGeneratorError("Status probabilities must sum to 1")

        self.LOAN_TYPES = {
            "Home": 0.30,
            "Personal": 0.25,
            "MSME": 0.20,
            "Education": 0.15,
            "Gold": 0.05,
            "Asset": 0.05
        }

        self.OCCUPATIONS = {
            "Salaried": 0.45,
            "Self-Employed": 0.25,
            "Business Owner": 0.15,
            "Student": 0.10,
            "Retired": 0.05
        }

        self.INCOME_RANGES = {
            'low': (300000, 800000),
            'medium': (800001, 2000000),
            'high': (2000001, 15000000)
        }

    def _generate_delhi_ncr_pincodes(self) -> Set[int]:
        """
        Generate set of valid Delhi NCR pincodes

        Returns:
            Set[int]: Set of valid pincodes
        """
        try:
            pincodes = set()

            # Delhi (110001-110096)
            pincodes.update(range(110001, 110097))

            # Noida (201301-201340)
            pincodes.update(range(201301, 201341))

            # Gurgaon (122001-122108)
            pincodes.update(range(122001, 122109))

            # Faridabad
            pincodes.update(range(121001, 121012))
            pincodes.update(range(121101, 121108))
            pincodes.add(124507)

            # Specific Haryana pincodes
            pincodes.update([123003] +
                            list(range(123413, 123419)) +
                            list(range(123502, 123507)))

            # Specific UP pincodes
            pincodes.update(range(201001, 201014))
            pincodes.update(range(201201, 201207))
            pincodes.update({245101, 245201, 245205, 245207, 245208, 245304})

            return pincodes

        except Exception as e:
            raise DataGeneratorError(f"Error generating pincodes: {str(e)}") from e

    def _generate_record_count(self) -> int:
        return self.min_records_per_pincode + random.randint(0, self.MAX_ADDITIONAL_RECORDS)

    def _generate_loan_date(self) -> datetime:
        days_range = (self.end_date - self.start_date).days
        weights = np.linspace(1, 2, days_range)
        random_days = random.choices(range(days_range), weights=weights)[0]
        return self.start_date + timedelta(days=random_days)

    def _generate_amount(self) -> float:
        amount = random.betavariate(2, 3.5) * (self.MAX_LOAN_AMOUNT - self.MIN_LOAN_AMOUNT) + self.MIN_LOAN_AMOUNT
        return round(max(self.MIN_LOAN_AMOUNT, min(amount, self.MAX_LOAN_AMOUNT)), -3)

    def _calculate_loan_status(self, start_date: datetime, tenure_months: int) -> str:
        loan_end_date = start_date + timedelta(days=tenure_months * 30)
        months_elapsed = (datetime.now() - start_date).days / 30

        if months_elapsed >= tenure_months:
            return random.choices(["Closed", "Defaulted (NPA)"], weights=[0.90, 0.10])[0]
        if random.random() < 0.15:
            return random.choices(["Closed", "Defaulted (NPA)"], weights=[0.80, 0.20])[0]
        return "Ongoing"

    def _generate_single_record(self, application_id: int, pincode: int, total_customers: int) -> Dict:
        income_category = random.choices(['low', 'medium', 'high'], weights=[0.33, 0.34, 0.33])[0]
        income = random.randint(*self.INCOME_RANGES[income_category])
        applied_amount = self._generate_amount()
        loan_start_date = self._generate_loan_date()
        tenure_months = random.choice([12, 24, 36, 48, 60])
        status = self._calculate_loan_status(loan_start_date, tenure_months)

        total_payments = tenure_months if status == "Closed" else min((datetime.now() - loan_start_date).days // 30, tenure_months)
        delayed_ratio = random.uniform(0.4, 0.6) if status == "Defaulted (NPA)" else random.uniform(0, 0.2)
        delayed_payments = int(total_payments * delayed_ratio)
        successful_payments = total_payments - delayed_payments

        return {
            "application_id": application_id,
            "customer_id": random.randint(1, total_customers),
            "pincode": pincode,
            "applied_amount": applied_amount,
            "loan_type": random.choices(list(self.LOAN_TYPES.keys()), weights=list(self.LOAN_TYPES.values()))[0],
            "loan_start_date": loan_start_date.isoformat(),
            "income": income,
            "occupation": random.choices(list(self.OCCUPATIONS.keys()), weights=list(self.OCCUPATIONS.values()))[0],
            "status": status,
            "successful_payments": successful_payments,
            "delayed_payments": delayed_payments,
            "total_payments": total_payments,
            "interest_rate": round(random.uniform(8.5, 16.5), 2),
            "tenure_months": tenure_months,
            "npa_flag": 1 if status == "Defaulted (NPA)" else 0
        }

    def _generate_batch(self, start_id: int, batch_size: int, pincode: int, total_customers: int) -> List[Dict]:
        records = []
        for i in range(batch_size):
            records.append(self._generate_single_record(start_id + i, pincode, total_customers))
        return records

    def generate_csv_data(self, output_filename: Optional[str] = None):
        records_per_pincode = {pincode: self._generate_record_count() for pincode in self.selected_pincodes}
        total_records = sum(records_per_pincode.values())
        logger.info(f"Generating {total_records:,} records")

        output_file = Path(output_filename) if output_filename else self.output_dir / "loan_applications.csv"
        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            futures = []
            current_app_id = 1

            for pincode, num_records in records_per_pincode.items():
                while num_records > 0:
                    batch_size = min(self.BATCH_SIZE, num_records)
                    futures.append(executor.submit(self._generate_batch, current_app_id, batch_size, pincode, total_records // 2))
                    current_app_id += batch_size
                    num_records -= batch_size

            all_records = []
            for future in futures:
                all_records.extend(future.result())

            pd.DataFrame(all_records).to_csv(output_file, index=False)
            logger.info(f"Data saved at {output_file}")


# if __name__ == "__main__":
#     generator = DataGenerator(min_records_per_pincode=100, seed=42)
#     generator.generate_csv_data()
