#!/usr/bin/env python3
"""
API Endpoint Testing Script for Rakshak Platform
Run this script to test both fraud detection and loan eligibility endpoints.

Usage:
    export DATABRICKS_TOKEN='your-token-here'
    python test_endpoints.py
    
Or:
    python test_endpoints.py --token 'your-token-here'
"""

import requests
import json
import os
import argparse
from datetime import datetime

# Configuration
DATABRICKS_HOST = "https://dbc-a4d8c1e8-b4cf.cloud.databricks.com"
FRAUD_ENDPOINT = f"{DATABRICKS_HOST}/serving-endpoints/rakshak-fraud-api/invocations"
LOAN_ENDPOINT = f"{DATABRICKS_HOST}/serving-endpoints/loan-eligibility-api/invocations"


def test_fraud_detection(token):
    """Test the fraud detection endpoint with sample data."""
    print(f"\n{'='*70}")
    print("🚨 FRAUD DETECTION API TEST")
    print(f"{'='*70}\n")
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # Sample transaction matching the Streamlit form defaults
    transaction = {
        'day_of_week': 0,
        'device_enc': 2,
        'amount_norm': 0.073494,  # 3674.69 / 50000
        'hour_of_day': 3,
        'txn_velocity': 0.0,
        'odd_hour_flag': 1,
        'device_network_interaction': 200,
        'sender_avg_amount_prev': 4303.93,
        'amount_weekend': 0.0,
        'fuel_large_flag': 0,
        'amount_vs_sender_mean': 0.853755,
        'high_amount_odd_hour': 0,
        'cat_device_interaction': 202,
        'same_bank_flag': 0,
        'sender_age_enc': 1,
        'weekend_high_spend': 0,
        'txn_type_enc': 3,
        'high_amount_flag': 0,
        'amount_hour_interaction': 11.02407,
        'sender_txn_count_prev': 77,
        'amount_inr': 3674.69,
        'is_weekend': 0,
        'receiver_age_enc': 0,
        'network_enc': 0,
        'sender_max_amount_prev': 6455.895,
        'amount_ratio_deviation': 0.146245,
        'cat_enc': 2
    }
    
    print(f"📍 Endpoint: {FRAUD_ENDPOINT}")
    print(f"📦 Payload: {len(transaction)} features")
    print(f"💰 Test Amount: ₹{transaction['amount_inr']:,.2f}")
    print(f"⏰ Time: {transaction['hour_of_day']}:00 (Odd Hour: {'Yes' if transaction['odd_hour_flag'] else 'No'})")
    print(f"\n🔄 Sending request...")
    
    try:
        payload = {"dataframe_records": [transaction]}
        response = requests.post(FRAUD_ENDPOINT, headers=headers, json=payload, timeout=30)
        
        print(f"📡 Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ SUCCESS!\n")
            
            print("📄 Full Response Structure:")
            print(json.dumps(result, indent=2))
            
            if "predictions" in result and len(result["predictions"]) > 0:
                prediction = result["predictions"][0]
                print(f"\n{'─'*70}")
                print("🎯 Parsed Prediction:")
                print(f"{'─'*70}")
                
                # Check for expected fields
                fraud_prob = prediction.get("fraud_probability")
                fraud_pred = prediction.get("fraud_prediction")
                
                if fraud_prob is not None:
                    print(f"  🔢 fraud_probability: {fraud_prob:.4f} ({fraud_prob*100:.2f}%)")
                if fraud_pred is not None:
                    verdict = "🔴 FRAUD" if fraud_pred == 1 else "🟢 SAFE"
                    print(f"  ⚖️  fraud_prediction: {fraud_pred} ({verdict})")
                
                # Show any other fields
                other_fields = {k: v for k, v in prediction.items() 
                              if k not in ['fraud_probability', 'fraud_prediction']}
                if other_fields:
                    print(f"\n  📋 Additional Fields:")
                    for key, value in other_fields.items():
                        print(f"     - {key}: {value}")
                        
                return True, prediction
            else:
                print("⚠️  No predictions found in response")
                return False, None
        else:
            print(f"❌ ERROR!")
            print(f"Response: {response.text[:500]}")
            return False, None
            
    except Exception as e:
        print(f"❌ Exception: {type(e).__name__}")
        print(f"Error: {str(e)}")
        return False, None


def test_loan_eligibility(token):
    """Test the loan eligibility endpoint with sample data."""
    print(f"\n{'='*70}")
    print("💰 LOAN ELIGIBILITY API TEST")
    print(f"{'='*70}\n")
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # Sample applicant matching the Streamlit form defaults
    applicant = {
        'age': 30,
        'gender': 1,  # Male
        'married': 0,  # Single
        'dependents': 0,
        'education': 1,  # Bachelor's
        'self_employed': 0,  # Salaried
        'income': 50000,
        'coapplicant_income': 0,
        'loan_amount': 500000,
        'loan_term': 60,
        'credit_score': 750,
        'property_area': 1,
        'existing_emi': 5000,
        'dti_ratio': 10.0,
        'employment_years': 5,
        'bank_years': 3
    }
    
    print(f"📍 Endpoint: {LOAN_ENDPOINT}")
    print(f"📦 Payload: {len(applicant)} features")
    print(f"👤 Age: {applicant['age']}, Income: ₹{applicant['income']:,}/month")
    print(f"💳 Credit Score: {applicant['credit_score']}, DTI: {applicant['dti_ratio']:.1f}%")
    print(f"💰 Loan Request: ₹{applicant['loan_amount']:,} for {applicant['loan_term']} months")
    print(f"\n🔄 Sending request...")
    
    try:
        payload = {"dataframe_records": [applicant]}
        response = requests.post(LOAN_ENDPOINT, headers=headers, json=payload, timeout=30)
        
        print(f"📡 Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ SUCCESS!\n")
            
            print("📄 Full Response Structure:")
            print(json.dumps(result, indent=2))
            
            if "predictions" in result and len(result["predictions"]) > 0:
                prediction = result["predictions"][0]
                print(f"\n{'─'*70}")
                print("🎯 Parsed Prediction:")
                print(f"{'─'*70}")
                
                # Check for multiple possible field names (based on app.py fallback logic)
                eligible = prediction.get("eligible", prediction.get("loan_status"))
                probability = prediction.get("probability", prediction.get("eligibility_score"))
                max_amount = prediction.get("max_loan_amount")
                
                if eligible is not None:
                    verdict = "✅ ELIGIBLE" if eligible else "❌ NOT ELIGIBLE"
                    print(f"  ⚖️  Eligibility: {eligible} ({verdict})")
                if probability is not None:
                    print(f"  🔢 Probability/Score: {probability:.4f} ({probability*100:.2f}%)")
                if max_amount is not None:
                    print(f"  💰 Max Loan Amount: ₹{max_amount:,.2f}")
                
                # Show all fields to understand the schema
                print(f"\n  📋 All Response Fields:")
                for key, value in prediction.items():
                    print(f"     - {key}: {value} (type: {type(value).__name__})")
                        
                return True, prediction
            else:
                print("⚠️  No predictions found in response")
                return False, None
        else:
            print(f"❌ ERROR!")
            print(f"Response: {response.text[:500]}")
            return False, None
            
    except Exception as e:
        print(f"❌ Exception: {type(e).__name__}")
        print(f"Error: {str(e)}")
        return False, None


def check_endpoint_health(token, endpoint_name):
    """Check if an endpoint is healthy."""
    try:
        url = f"{DATABRICKS_HOST}/api/2.0/serving-endpoints/{endpoint_name}"
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(url, headers=headers, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            state = data.get("state", {}).get("ready", "UNKNOWN")
            return state == "READY", state
        return False, f"HTTP {response.status_code}"
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description='Test Rakshak API endpoints')
    parser.add_argument('--token', help='Databricks access token', default=None)
    args = parser.parse_args()
    
    # Get token from args or environment
    token = args.token or os.getenv("DATABRICKS_TOKEN", "")
    
    if not token:
        print("❌ ERROR: No Databricks token provided!")
        print("\nPlease provide a token using one of these methods:")
        print("  1. Set environment variable: export DATABRICKS_TOKEN='your-token-here'")
        print("  2. Use command line: python test_endpoints.py --token 'your-token-here'")
        print("\nGet your token from: Databricks UI → User Settings → Access Tokens")
        return
    
    print("="*70)
    print("🛡️  RAKSHAK API ENDPOINT TESTING")
    print("="*70)
    print(f"🕐 Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌐 Host: {DATABRICKS_HOST}")
    print(f"🔑 Token: {'*' * 20}{token[-4:] if len(token) > 4 else '****'}")
    
    # Check endpoint health first
    print(f"\n{'─'*70}")
    print("📡 Checking Endpoint Health...")
    print(f"{'─'*70}")
    
    fraud_healthy, fraud_state = check_endpoint_health(token, "rakshak-fraud-api")
    loan_healthy, loan_state = check_endpoint_health(token, "loan-eligibility-api")
    
    print(f"  • rakshak-fraud-api: {'✅ READY' if fraud_healthy else f'❌ {fraud_state}'}")
    print(f"  • loan-eligibility-api: {'✅ READY' if loan_healthy else f'❌ {loan_state}'}")
    
    # Test endpoints
    fraud_success, fraud_result = test_fraud_detection(token)
    loan_success, loan_result = test_loan_eligibility(token)
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 TEST SUMMARY")
    print(f"{'='*70}")
    print(f"  • Fraud Detection API: {'✅ PASSED' if fraud_success else '❌ FAILED'}")
    print(f"  • Loan Eligibility API: {'✅ PASSED' if loan_success else '❌ FAILED'}")
    
    if fraud_success and loan_success:
        print("\n🎉 All tests passed! Both endpoints are working correctly.")
        print("\n💡 Next Steps:")
        print("  1. Review the response structures above")
        print("  2. Update app.py if field names don't match")
        print("  3. Run your Streamlit app with confidence!")
    elif not fraud_success and not loan_success:
        print("\n⚠️  Both endpoints failed. Possible issues:")
        print("  • Invalid or expired token")
        print("  • Endpoints not deployed or in stopped state")
        print("  • Network connectivity issues")
    else:
        print("\n⚠️  Some endpoints failed. Check the detailed output above.")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
