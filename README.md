# Kenya Fraud Check 2025

## Overview
This project implements a machine learning pipeline tailored for detecting identity theft and fraud patterns within the Kenyan digital lending ecosystem. It focuses on specific indicators such as:
- M-Pesa registration status
- Device fingerprint reuse counts
- Kenyan National ID validity checks
- Application time-series patterns (e.g., midnight surges)
- Phone number prefix analysis (Safaricom/Airtel/Telkom ranges)

## Features
- **Synthetic Data Generator**: Simulates realistic Kenyan loan application metadata.
- **Pre-processing**: Handles Kenyan-specific categorical features.
- **Model**: Random Forest Classifier optimized for high-precision fraud detection.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train the model:
   ```bash
   python train_model.py
   ```

## Project Structure
- `train_model.py`: Data generation, feature engineering, and model training.
- `models/`: Directory where serialized models are stored.