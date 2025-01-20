import logging
from faker import Faker
import random
import pandas as pd
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class DataGenerator:
    MIN_LOAN_AMOUNT = 100000
    MAX_LOAN_AMOUNT = 1000000
    """Generate synthetic loan application data for Delhi NCR region"""

    def __init__(self, min_records_per_pincode: int = 50, output_dir: str = "data"):
        """
        Initialize the Data Generator for Delhi NCR

        Args:
            min_records_per_pincode: Minimum records per pincode (default: 50)
            output_dir: Directory to store output files
        """
        self.fake = Faker()
        self.min_records_per_pincode = min_records_per_pincode
        self.output_dir = Path(output_dir)

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate Delhi NCR pincodes
        self.selected_pincodes = self._generate_delhi_ncr_pincodes()
        logger.info(f"Generated {len(self.selected_pincodes)} Delhi NCR pincodes")

        # Define constants with realistic distributions

        self.STATUSES = {
            "Approved": 0.30,
            "Rejected": 0.12,
            "Inquired": 0.08,
            "Closed": 0.25,
            "Ongoing": 0.20,
            "Defaulted (NPA)": 0.05
        }

        self.LOAN_TYPES = {
            "Home": 0.30,
            "Personal": 0.25,
            "MSME": 0.15,
            "Education": 0.15,
            "Gold": 0.07,
            "Asset": 0.08
        }

        self.OCCUPATIONS = {
            "Salaried": 0.45,
            "Self-Employed": 0.25,
            "Business Owner": 0.15,
            "Student": 0.10,
            "Retired": 0.05
        }

        self.INCOME_RANGES = {
            'low': (200000, 600000),
            'medium': (600001, 1200000),
            'high': (1200001, 5500000)
        }

    def _generate_delhi_ncr_pincodes(self) -> Set[int]:
        """Generate list of valid Delhi NCR pincodes"""
        pincodes = set()

        # Delhi pincodes (110001-110096)
        pincodes.update(range(110001, 110097))

        # Noida pincodes (201301-201340)
        pincodes.update(range(201301, 201341))

        # Gurgaon pincodes (122001-122108)
        pincodes.update(range(122001, 122109))

        # Specific Haryana pincodes
        pincodes.add(123003)
        pincodes.update(range(123413, 123419))
        pincodes.update(range(123502, 123507))

        # Specific UP pincodes
        pincodes.update(range(201001, 201014))
        pincodes.update(range(201201, 201207))
        specific_up_pincodes = {245101, 245201, 245205, 245207, 245208, 245304}
        pincodes.update(specific_up_pincodes)

        # Faridabad pincodes
        pincodes.update(range(121001, 121012))
        pincodes.update(range(121101, 121108))
        pincodes.add(124507)

        return pincodes

    def _generate_record_count(self) -> int:
        """Generate number of records for a pincode"""
        # Ensure minimum records with some variation
        base = self.min_records_per_pincode
        variation = random.randint(12, 50)  # Add 0-20 additional records
        return base + variation

    def _generate_amount_for_location(self, pincode: int, income: float) -> float:
        """Generate loan amount based on location and income"""

        amount = random.betavariate(2, 3.5) * (self.MAX_LOAN_AMOUNT - self.MIN_LOAN_AMOUNT) + self.MIN_LOAN_AMOUNT
        return round(max(self.MIN_LOAN_AMOUNT, min(amount, self.MAX_LOAN_AMOUNT)), -3)

    def _generate_single_record(self,
                                application_id: int,
                                pincode: int,
                                total_customers: int) -> Dict:
        """Generate a single loan application record with NCR-specific distributions"""
        try:
            # Generate income based on area
            if pincode in range(110001, 110096):  # Delhi
                income_category = random.choices(
                    ['low', 'medium', 'high'],
                    weights=[0.3, 0.4, 0.3]
                )[0]
            else:  # Other NCR
                income_category = random.choices(
                    ['low', 'medium', 'high'],
                    weights=[0.4, 0.4, 0.2]
                )[0]

            income_range = self.INCOME_RANGES[income_category]
            income = random.randint(*income_range)

            # Generate loan amount based on location and income
            applied_amount = self._generate_amount_for_location(pincode, income)

            # Generate correlated payments
            total_payments = random.randint(1, 100)
            delayed_ratio = random.uniform(0, 0.4)  # Max 40% delayed
            delayed_payments = int(total_payments * delayed_ratio)
            successful_payments = total_payments - delayed_payments

            # Generate loan date
            end_date = datetime.now()
            start_date = end_date - timedelta(days=6 * 365)  # Last 5 years
            loan_start_date = self.fake.date_between(
                start_date=start_date,
                end_date=end_date
            )

            status = random.choices(
                list(self.STATUSES.keys()),
                weights=list(self.STATUSES.values())
            )[0]

            return {
                "application_id": application_id,
                "customer_id": random.randint(1, total_customers),
                "pincode": pincode,
                "applied_amount": applied_amount,
                "loan_type": random.choices(
                    list(self.LOAN_TYPES.keys()),
                    weights=list(self.LOAN_TYPES.values())
                )[0],
                "loan_start_date": loan_start_date.isoformat(),
                "income": income,
                "occupation": random.choices(
                    list(self.OCCUPATIONS.keys()),
                    weights=list(self.OCCUPATIONS.values())
                )[0],
                "status": status,
                "successful_payments": successful_payments,
                "delayed_payments": delayed_payments,
                "total_payments": total_payments,
                "interest_rate": round(random.uniform(7.5, 18.5), 2),  # NCR rates
                "tenure_months": random.choice([6,12, 18,24, 36, 48, 60, 72,84,96,120]),
                "npa_flag": 1 if status == "Defaulted (NPA)" else 0
            }
        except Exception as e:
            logger.error(f"Error generating record: {str(e)}")
            raise

    def generate_csv_data(self):
        """Generate loan application data for Delhi NCR"""
        try:
            # Calculate records per pincode
            records_per_pincode = {
                pincode: self._generate_record_count()
                for pincode in self.selected_pincodes
            }

            total_records = sum(records_per_pincode.values())
            logger.info(f"Generating {total_records:,} records across {len(self.selected_pincodes):,} NCR pincodes")

            # Create output file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = "data/loan_applications.csv"

            # Generate and save data in batches
            batch_size = 10000
            processed_records = 0
            current_app_id = 1

            for pincode, num_records in records_per_pincode.items():
                logger.info(f"Processing pincode {pincode} with {num_records} records")
                records_generated = 0

                while records_generated < num_records:
                    current_batch = []
                    batch_records = min(batch_size, num_records - records_generated)

                    for _ in range(batch_records):
                        record = self._generate_single_record(
                            application_id=current_app_id,
                            pincode=pincode,
                            total_customers=total_records // 2
                        )
                        current_batch.append(record)
                        current_app_id += 1
                        records_generated += 1

                    # Save batch
                    batch_df = pd.DataFrame(current_batch)
                    mode = "w" if processed_records == 0 else "a"
                    header = processed_records == 0

                    batch_df.to_csv(output_file,
                                    index=False,
                                    mode=mode,
                                    header=header)

                    processed_records += len(current_batch)

                    if processed_records % 10000 == 0:
                        logger.info(f"Generated {processed_records:,}/{total_records:,} records")

            logger.info(f"Data generation complete. File saved at: {output_file}")

            # Verify distribution
            self._verify_distribution(output_file)

        except Exception as e:
            logger.error(f"Error generating data: {str(e)}")
            raise

    def _verify_distribution(self, file_path: Path):
        """Verify pincode distribution and data quality"""
        logger.info("Verifying data distribution and quality...")

        try:
            chunk_size = 10000
            pincode_counts = {}
            total_records = 0

            # Process file in chunks
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                # Update pincode counts
                chunk_counts = chunk['pincode'].value_counts().to_dict()
                for pincode, count in chunk_counts.items():
                    pincode_counts[pincode] = pincode_counts.get(pincode, 0) + count

                total_records += len(chunk)

            # Verify minimum records per pincode
            insufficient_pincodes = {
                pincode: count
                for pincode, count in pincode_counts.items()
                if count < self.min_records_per_pincode
            }

            if insufficient_pincodes:
                logger.warning(
                    f"Found {len(insufficient_pincodes)} pincodes with insufficient records:"
                )
                for pincode, count in insufficient_pincodes.items():
                    logger.warning(f"Pincode {pincode}: {count} records")
            else:
                logger.info("All pincodes have sufficient records")

            # Log distribution summary
            logger.info(f"Total records generated: {total_records:,}")
            logger.info(f"Average records per pincode: {total_records / len(pincode_counts):,.1f}")
            logger.info(f"Number of unique pincodes: {len(pincode_counts):,}")

        except Exception as e:
            logger.error(f"Error verifying distribution: {str(e)}")
            raise