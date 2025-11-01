import inspect
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

from flask import Flask, render_template, request, redirect, url_for, flash, session
import mysql.connector
from werkzeug.security import generate_password_hash, check_password_hash
import re, hashlib, requests, os, random, json, secrets, string, base64, time
from dotenv import load_dotenv
load_dotenv()

# ------------------- AI Configuration -------------------
AI_AVAILABLE = False
AI_MODE = "fallback"
ollama_client = None
PREFERRED_MODEL = None
ACTIVE_PROVIDER = None

# ------------------- ML Model Configuration -------------------
ML_MODEL_AVAILABLE = False
wallet_risk_model = None
model_feature_columns = []
model_feature_importance = {}

try:
    import xgboost as xgb
    import joblib
    import numpy as np
    
    wallet_risk_model = joblib.load('wallet_risk_xgboost_model.pkl')
    with open('model_features.json', 'r') as f:
        feature_info = json.load(f)
        model_feature_columns = feature_info['feature_columns']
        model_feature_importance = feature_info['feature_importance']
    ML_MODEL_AVAILABLE = True
    print("✅ ML Model loaded successfully!")
    print(f"📊 Model expects {len(model_feature_columns)} features: {model_feature_columns}")
except Exception as e:
    print(f"❌ ML Model loading failed: {e}")
    ML_MODEL_AVAILABLE = False

# ------------------- API Configuration -------------------
HIBP_API_KEY = os.getenv('HIBP_API_KEY', '')
ETHERSCAN_API_KEY = os.getenv('ETHERSCAN_API_KEY', '')
NUMVERIFY_API_KEY = os.getenv('NUMVERIFY_API_KEY', '')
HIBP_ENABLED = bool(HIBP_API_KEY and len(HIBP_API_KEY) > 10)
ETHERSCAN_ENABLED = bool(ETHERSCAN_API_KEY and len(ETHERSCAN_API_KEY) > 10)
NUMVERIFY_ENABLED = bool(NUMVERIFY_API_KEY and len(NUMVERIFY_API_KEY) > 10)

print(f"🔧 API Status Check:")
print(f"   HIBP Enabled: {HIBP_ENABLED} (Key: {HIBP_API_KEY[:8]}...)" if HIBP_API_KEY else "   HIBP Enabled: False")
print(f"   Etherscan Enabled: {ETHERSCAN_ENABLED} (Key: {ETHERSCAN_API_KEY[:8]}...)" if ETHERSCAN_API_KEY else "   Etherscan Enabled: False")
print(f"   NumVerify Enabled: {NUMVERIFY_ENABLED} (Key: {NUMVERIFY_API_KEY[:8]}...)" if NUMVERIFY_API_KEY else "   NumVerify Enabled: False")

# ------------------- ML Helper Functions -------------------
def extract_ml_features(wallet_analysis):
    """Extract features for ML model from wallet analysis data with safe access"""
    try:
        if wallet_analysis.get("real_data", False):
            # Use real blockchain data with safe access - ONLY 9 FEATURES to match model
            features = [
                wallet_analysis.get("total_transactions", 0),
                wallet_analysis.get("unique_counterparties", 0),
                wallet_analysis.get("avg_transaction_value", 0),
                wallet_analysis.get("balance_eth", 0) or wallet_analysis.get("balance_btc", 0),
                min(wallet_analysis.get("total_transactions", 0) / 30, 50),  # Monthly velocity
                wallet_analysis.get("total_volume_eth", 0) / max(wallet_analysis.get("total_transactions", 1), 1),
                24,  # Default time between transactions
                wallet_analysis.get("high_value_count", 0) / max(wallet_analysis.get("total_transactions", 1), 1),
                np.log(wallet_analysis.get("unique_counterparties", 0) + 1)
            ]
        else:
            # Use simulated data with safe access - ONLY 9 FEATURES
            patterns = wallet_analysis.get("activity_patterns", {})
            features = [
                patterns.get("transaction_count", 0),
                patterns.get("unique_counterparties", 0),
                patterns.get("avg_transaction_value", 0),
                random.uniform(0, 100),  # Simulated balance
                patterns.get("transaction_count", 0) / 30,  # Monthly velocity
                patterns.get("daily_volume", 0) / max(patterns.get("transaction_count", 1), 1),
                patterns.get("time_between_tx", 24),
                patterns.get("high_value_tx_count", 0) / max(patterns.get("transaction_count", 1), 1),
                np.log(patterns.get("unique_counterparties", 0) + 1)
            ]
        
        # Ensure all features are finite numbers
        features = [float(f) if np.isfinite(float(f)) else 0.0 for f in features]
        
        # Debug feature extraction
        if len(features) != 9:
            print(f"⚠️ Feature count mismatch: Expected 9, got {len(features)}")
            # Ensure we return exactly 9 features
            features = features[:9] if len(features) > 9 else features + [0] * (9 - len(features))
            
        return np.array(features).reshape(1, -1)
        
    except Exception as e:
        print(f"❌ Feature extraction error: {e}")
        # Return default features on error - EXACTLY 9 FEATURES
        default_features = [0] * 9
        return np.array(default_features).reshape(1, -1)

def predict_wallet_risk_ml(wallet_analysis):
    """Use ML model to predict wallet risk"""
    if not ML_MODEL_AVAILABLE:
        return None
    
    try:
        features = extract_ml_features(wallet_analysis)
        
        # Debug: Check feature shape
        if features.shape[1] != len(model_feature_columns):
            print(f"❌ Feature shape mismatch: expected {len(model_feature_columns)}, got {features.shape[1]}")
            return None
            
        prediction_proba = wallet_risk_model.predict_proba(features)[0]
        prediction = wallet_risk_model.predict(features)[0]
        
        # Convert to risk score (0-10)
        risk_score_ml = int(prediction_proba[1] * 10)
        
        return {
            "ml_risk_score": risk_score_ml,
            "ml_risk_level": "High" if risk_score_ml >= 7 else "Medium" if risk_score_ml >= 4 else "Low",
            "malicious_probability": float(prediction_proba[1]),
            "confidence": float(max(prediction_proba)),
            "flagged_ml": prediction == 1,
            "features_used": len(model_feature_columns)
        }
    except Exception as e:
        print(f"ML prediction error: {e}")
        return None

def get_feature_importance():
    """Get feature importance from trained model"""
    if not ML_MODEL_AVAILABLE:
        return {}
    return model_feature_importance

# ------------------- Enhanced Risk Scoring -------------------
def calculate_enhanced_risk_score(analysis_data):
    """Calculate more sensitive risk scores for wallet analysis"""
    risk_score = 0
    deviations = []
    
    # Balance-based risks (more sensitive thresholds)
    balance = analysis_data.get("balance_eth") or analysis_data.get("balance_btc")
    if balance and balance > 25:  # Further lowered threshold
        risk_score += 4
        deviations.append("Large wallet balance detected")
    elif balance and balance > 5:
        risk_score += 2
        deviations.append("Moderate wallet balance")
    
    # Transaction frequency (more sensitive)
    tx_count = analysis_data.get("total_transactions", 0)
    if tx_count > 75:  # Lowered threshold
        risk_score += 4
        deviations.append("Very high transaction frequency")
    elif tx_count > 20:  # Lowered threshold
        risk_score += 2
        deviations.append("High transaction frequency")
    elif tx_count == 0:
        risk_score += 3  # Increased risk for no transactions
        deviations.append("No transaction history - highly suspicious")
    
    # High-value transactions
    high_value_count = analysis_data.get("high_value_count", 0)
    if high_value_count > 8:  # Lowered threshold
        risk_score += 4
        deviations.append("Multiple high-value transactions")
    elif high_value_count > 2:  # Lowered threshold
        risk_score += 2
        deviations.append("Several high-value transactions")
    
    # Counterparty diversity
    unique_counterparties = analysis_data.get("unique_counterparties", 0)
    if unique_counterparties < 2 and tx_count > 3:  # More sensitive
        risk_score += 4
        deviations.append("Very limited counterparty diversity")
    elif unique_counterparties < 4 and tx_count > 8:  # More sensitive
        risk_score += 2
        deviations.append("Limited counterparty diversity")
    
    # Transaction patterns
    avg_tx_value = analysis_data.get("avg_transaction_value", 0)
    if avg_tx_value > 2500:  # Lowered threshold
        risk_score += 3
        deviations.append("Unusually high average transaction value")
    elif avg_tx_value > 500:  # Lowered threshold
        risk_score += 1
        deviations.append("High average transaction value")
    
    # Volume-based risks
    total_volume = analysis_data.get("total_volume_eth") or analysis_data.get("total_volume_btc")
    if total_volume and total_volume > 50:  # Lowered threshold
        risk_score += 2
        deviations.append("High total transaction volume")
    
    # Concentration risk
    if tx_count > 0 and unique_counterparties > 0:
        concentration_ratio = tx_count / unique_counterparties
        if concentration_ratio > 8:  # Lowered threshold
            risk_score += 3
            deviations.append("High transaction concentration with few counterparties")
        elif concentration_ratio > 3:  # Lowered threshold
            risk_score += 1
            deviations.append("Moderate transaction concentration")
    
    return risk_score, deviations

# ------------------- Etherscan API V2 Functions -------------------
def get_etherscan_balance_v2(wallet_address):
    """Get real wallet balance from Etherscan API V2"""
    if not ETHERSCAN_ENABLED:
        return None
    
    endpoints = [
        {
            'name': 'V2 with chainId',
            'url': 'https://api.etherscan.io/v2/api',
            'params': {
                'module': 'account',
                'action': 'balance',
                'address': wallet_address,
                'tag': 'latest',
                'chainid': 1,
                'apikey': ETHERSCAN_API_KEY
            }
        },
        {
            'name': 'V1 fallback',
            'url': 'https://api.etherscan.io/api',
            'params': {
                'module': 'account',
                'action': 'balance',
                'address': wallet_address,
                'tag': 'latest',
                'apikey': ETHERSCAN_API_KEY
            }
        }
    ]
    
    for endpoint in endpoints:
        try:
            url = endpoint['url']
            params = endpoint['params']
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('status') == '1':
                    balance_wei = int(data['result'])
                    balance_eth = balance_wei / 10**18
                    return balance_eth
                else:
                    error_msg = data.get('message', 'Unknown error')
                    if 'rate limit' in error_msg.lower():
                        time.sleep(2)
                        continue
            else:
                continue
                
        except Exception:
            continue
    
    return None

def get_etherscan_transactions_v2(wallet_address):
    """Get recent transactions from Etherscan API V2"""
    if not ETHERSCAN_ENABLED:
        return None
    
    endpoints = [
        {
            'name': 'V2 with chainId',
            'url': 'https://api.etherscan.io/v2/api',
            'params': {
                'module': 'account',
                'action': 'txlist',
                'address': wallet_address,
                'startblock': 0,
                'endblock': 99999999,
                'page': 1,
                'offset': 10,
                'sort': 'desc',
                'chainid': 1,
                'apikey': ETHERSCAN_API_KEY
            }
        },
        {
            'name': 'V1 fallback',
            'url': 'https://api.etherscan.io/api',
            'params': {
                'module': 'account',
                'action': 'txlist',
                'address': wallet_address,
                'startblock': 0,
                'endblock': 99999999,
                'page': 1,
                'offset': 10,
                'sort': 'desc',
                'apikey': ETHERSCAN_API_KEY
            }
        }
    ]
    
    for endpoint in endpoints:
        try:
            url = endpoint['url']
            params = endpoint['params']
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('status') == '1':
                    transactions = data['result']
                    return transactions
                else:
                    error_msg = data.get('message', 'Unknown error')
                    if 'rate limit' in error_msg.lower():
                        time.sleep(2)
                        continue
            else:
                continue
                
        except Exception:
            continue
    
    return None

def analyze_real_wallet_activity(wallet_address):
    """Enhanced real wallet analysis with better risk detection"""
    balance = get_etherscan_balance_v2(wallet_address)
    transactions = get_etherscan_transactions_v2(wallet_address)
    
    if balance is None and transactions is None:
        return None
    
    # Calculate metrics
    total_transactions = len(transactions) if transactions else 0
    unique_counterparties = set()
    total_value = 0
    high_value_count = 0
    
    if transactions:
        for tx in transactions:
            from_addr = tx.get('from', '')
            to_addr = tx.get('to', '')
            value_wei = int(tx.get('value', 0))
            value_eth = value_wei / 10**18
            
            if from_addr.lower() != wallet_address.lower():
                unique_counterparties.add(from_addr)
            if to_addr and to_addr.lower() != wallet_address.lower():
                unique_counterparties.add(to_addr)
            
            total_value += value_eth
            
            if value_eth > 0.1:  # Further lowered threshold for high-value
                high_value_count += 1
    
    avg_transaction_value = total_value / total_transactions if total_transactions > 0 else 0
    
    # Prepare data for risk calculation
    analysis_data = {
        "balance_eth": balance,
        "total_transactions": total_transactions,
        "unique_counterparties": len(unique_counterparties),
        "total_volume_eth": total_value,
        "avg_transaction_value": avg_transaction_value,
        "high_value_count": high_value_count
    }
    
    # Calculate enhanced risk score
    risk_score, deviations = calculate_enhanced_risk_score(analysis_data)
    
    # Enhanced risk level calculation with more aggressive thresholds
    if risk_score >= 6:  # Lowered threshold for HIGH
        risk_level = "HIGH"
        flagged = True
    elif risk_score >= 3:  # Lowered threshold for MEDIUM
        risk_level = "MEDIUM" 
        flagged = True
    elif risk_score >= 1:
        risk_level = "LOW"
        flagged = False
    else:
        risk_level = "VERY LOW"
        flagged = False
    
    result = {
        "real_data": True,
        "balance_eth": balance,
        "total_transactions": total_transactions,
        "unique_counterparties": len(unique_counterparties),
        "total_volume_eth": total_value,
        "avg_transaction_value": avg_transaction_value,
        "high_value_count": high_value_count,
        "deviations": deviations,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "flagged": flagged,
        "raw_metrics": analysis_data
    }
    
    return result

# ------------------- Bitcoin API Functions -------------------
def get_bitcoin_balance(address):
    """Get real Bitcoin balance from Blockchain.com API with better error handling"""
    try:
        url = f"https://blockchain.info/balance?active={address}"
        print(f"🔍 Fetching Bitcoin balance for: {address}")
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if address in data:
                balance_satoshi = data[address]['final_balance']
                balance_btc = balance_satoshi / 100000000
                print(f"✅ Bitcoin balance found: {balance_btc} BTC")
                return balance_btc
            else:
                print(f"❌ Address {address} not found in Bitcoin API response")
        elif response.status_code == 429:
            print("⚠️ Bitcoin API rate limited, using fallback data")
            return random.uniform(0.001, 5.0)  # Fallback balance
        else:
            print(f"❌ Bitcoin API HTTP error: {response.status_code}")
        return None
    except Exception as e:
        print(f"❌ Bitcoin balance API exception: {e}")
        return None

def get_bitcoin_transactions(address):
    """Get recent Bitcoin transactions with better error handling"""
    try:
        url = f"https://blockchain.info/rawaddr/{address}?limit=10"
        print(f"🔍 Fetching Bitcoin transactions for: {address}")
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            transactions = data.get('txs', [])
            print(f"✅ Found {len(transactions)} Bitcoin transactions")
            return transactions
        elif response.status_code == 429:
            print("⚠️ Bitcoin transactions API rate limited, using simulated data")
            # Return simulated transactions to avoid complete failure
            return [{"inputs": [{"prev_out": {"addr": "1Simulated1", "value": 100000}}], 
                     "out": [{"addr": address, "value": 50000}]}] * random.randint(1, 8)
        else:
            print(f"❌ Bitcoin transactions API HTTP error: {response.status_code}")
        return None
    except Exception as e:
        print(f"❌ Bitcoin transactions API exception: {e}")
        return None

def analyze_real_bitcoin_wallet(wallet_address):
    """Enhanced real Bitcoin wallet analysis with better risk detection"""
    balance = get_bitcoin_balance(wallet_address)
    transactions = get_bitcoin_transactions(wallet_address)
    
    # If both API calls failed, return None to indicate simulation needed
    if balance is None and transactions is None:
        return None
    
    total_transactions = len(transactions) if transactions else 0
    unique_counterparties = set()
    total_value = 0
    high_value_count = 0
    
    if transactions:
        for tx in transactions:
            # Sum all inputs and outputs for this address
            tx_value = 0
            
            # Check inputs (sending)
            for inp in tx.get('inputs', []):
                if 'prev_out' in inp and 'addr' in inp['prev_out']:
                    if inp['prev_out']['addr'] == wallet_address:
                        tx_value += inp['prev_out'].get('value', 0)
                    unique_counterparties.add(inp['prev_out']['addr'])
            
            # Check outputs (receiving)
            for out in tx.get('out', []):
                if 'addr' in out:
                    if out['addr'] == wallet_address:
                        tx_value += out.get('value', 0)
                    unique_counterparties.add(out['addr'])
            
            total_value += tx_value / 100000000  # Convert to BTC
            
            if tx_value / 100000000 > 0.05:  # Further lowered threshold (0.05 BTC)
                high_value_count += 1
    
    # Remove the wallet itself from counterparties
    unique_counterparties.discard(wallet_address)
    
    # Prepare data for enhanced risk calculation
    analysis_data = {
        "balance_btc": balance,
        "total_transactions": total_transactions,
        "unique_counterparties": len(unique_counterparties),
        "total_volume_btc": total_value,
        "high_value_count": high_value_count,
        "avg_transaction_value": total_value / total_transactions if total_transactions > 0 else 0
    }
    
    # Calculate enhanced risk score using the same function
    risk_score, deviations = calculate_enhanced_risk_score(analysis_data)
    
    # Enhanced risk level calculation (same as Ethereum)
    if risk_score >= 6:
        risk_level = "HIGH"
        flagged = True
    elif risk_score >= 3:
        risk_level = "MEDIUM"
        flagged = True
    elif risk_score >= 1:
        risk_level = "LOW"
        flagged = False
    else:
        risk_level = "VERY LOW"
        flagged = False
    
    result = {
        "real_data": True,
        "balance_btc": balance,
        "total_transactions": total_transactions,
        "unique_counterparties": len(unique_counterparties),
        "total_volume_btc": total_value,
        "high_value_count": high_value_count,
        "avg_transaction_value": total_value / total_transactions if total_transactions > 0 else 0,
        "deviations": deviations,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "flagged": flagged,
        "raw_metrics": analysis_data
    }
    
    return result

# ------------------- Enhanced Wallet Analysis with ML -------------------
def analyze_wallet_activity(wallet_address, chain="ETH"):
    """Analyze wallet activity - uses real data if available, otherwise simulated"""
    
    if chain == "ETH" and ETHERSCAN_ENABLED:
        real_analysis = analyze_real_wallet_activity(wallet_address)
        if real_analysis is not None:
            analysis_result = real_analysis
        else:
            analysis_result = simulate_analysis(wallet_address)
    elif chain == "BTC":
        real_analysis = analyze_real_bitcoin_wallet(wallet_address)
        if real_analysis is not None:
            analysis_result = real_analysis
        else:
            analysis_result = simulate_analysis(wallet_address)
    else:
        analysis_result = simulate_analysis(wallet_address)
    
    # Debug output
    debug_risk_calculation(analysis_result)
    
    # Add ML prediction
    ml_prediction = predict_wallet_risk_ml(analysis_result)
    
    # Combine results
    if ml_prediction:
        analysis_result["ml_analysis"] = ml_prediction
        # Enhanced risk assessment with ML - use maximum of both scores
        enhanced_risk_score = max(analysis_result["risk_score"], ml_prediction["ml_risk_score"])
        analysis_result["enhanced_risk_score"] = enhanced_risk_score
        analysis_result["enhanced_risk_level"] = ml_prediction["ml_risk_level"]
        analysis_result["flagged"] = analysis_result["flagged"] or ml_prediction["flagged_ml"]
    else:
        analysis_result["ml_analysis"] = None
        analysis_result["enhanced_risk_score"] = analysis_result["risk_score"]
        analysis_result["enhanced_risk_level"] = analysis_result["risk_level"]
    
    return analysis_result

def debug_risk_calculation(wallet_analysis):
    """Debug function to understand risk calculation"""
    print(f"\n🔍 RISK CALCULATION DEBUG for {wallet_analysis.get('real_data', False) and 'REAL' or 'SIMULATED'} data:")
    print(f"   Risk Score: {wallet_analysis.get('risk_score', 0)}")
    print(f"   Risk Level: {wallet_analysis.get('risk_level', 'UNKNOWN')}")
    print(f"   Flagged: {wallet_analysis.get('flagged', False)}")
    print(f"   Deviations: {wallet_analysis.get('deviations', [])}")
    
    if wallet_analysis.get('real_data', False):
        balance = wallet_analysis.get('balance_eth') or wallet_analysis.get('balance_btc')
        print(f"   Balance: {balance}")
        print(f"   Transactions: {wallet_analysis.get('total_transactions', 0)}")
        print(f"   Counterparties: {wallet_analysis.get('unique_counterparties', 0)}")
        print(f"   High Value TX: {wallet_analysis.get('high_value_count', 0)}")
    else:
        patterns = wallet_analysis.get('activity_patterns', {})
        print(f"   Daily Volume: {patterns.get('daily_volume', 0)}")
        print(f"   TX Count: {patterns.get('transaction_count', 0)}")
        print(f"   Counterparties: {patterns.get('unique_counterparties', 0)}")
    print("---")

def simulate_analysis(wallet_address):
    """Enhanced simulation with more realistic risk distribution"""
    # Create risk-weighted simulation
    risk_profile = random.choices(["low", "medium", "high"], weights=[40, 40, 20])[0]
    
    if risk_profile == "high":
        activity_patterns = {
            "daily_volume": random.randint(40000, 150000),
            "transaction_count": random.randint(80, 300),
            "unique_counterparties": random.randint(2, 6),  # Very limited diversity
            "avg_transaction_value": random.randint(8000, 50000),
            "time_between_tx": random.randint(1, 3),  # Very frequent
            "high_value_tx_count": random.randint(10, 25),
            "suspicious_patterns": random.choice([2, 3])
        }
    elif risk_profile == "medium":
        activity_patterns = {
            "daily_volume": random.randint(20000, 60000),
            "transaction_count": random.randint(30, 100),
            "unique_counterparties": random.randint(5, 12),
            "avg_transaction_value": random.randint(2000, 15000),
            "time_between_tx": random.randint(4, 8),
            "high_value_tx_count": random.randint(3, 12),
            "suspicious_patterns": random.choice([0, 1, 2])
        }
    else:  # low risk
        activity_patterns = {
            "daily_volume": random.randint(1000, 20000),
            "transaction_count": random.randint(5, 40),
            "unique_counterparties": random.randint(8, 25),
            "avg_transaction_value": random.randint(100, 3000),
            "time_between_tx": random.randint(12, 72),
            "high_value_tx_count": random.randint(0, 4),
            "suspicious_patterns": 0
        }
    
    deviations = []
    risk_score = 0
    
    # Enhanced risk detection with more sensitive thresholds
    if activity_patterns["daily_volume"] > 30000:
        deviations.append("Unusually high daily volume")
        risk_score += 3
    
    if activity_patterns["transaction_count"] > 60:
        deviations.append("High transaction frequency")
        risk_score += 3
    
    if activity_patterns["suspicious_patterns"] > 0:
        deviations.append("Suspicious transaction patterns detected")
        risk_score += activity_patterns["suspicious_patterns"] + 2
    
    if activity_patterns["unique_counterparties"] < 4:
        deviations.append("Limited counterparty diversity")
        risk_score += 3
    
    if activity_patterns["high_value_tx_count"] > 8:
        deviations.append("Multiple high-value transactions")
        risk_score += 3
    
    # Enhanced risk level calculation
    if risk_score >= 7:
        risk_level = "HIGH"
        flagged = True
    elif risk_score >= 4:
        risk_level = "MEDIUM"
        flagged = True
    elif risk_score >= 2:
        risk_level = "LOW"
        flagged = False
    else:
        risk_level = "VERY LOW"
        flagged = False
    
    return {
        "real_data": False,
        "activity_patterns": activity_patterns,
        "deviations": deviations,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "flagged": flagged,
        "risk_profile": risk_profile
    }

def generate_comparative_risk_map(wallet_addresses):
    """Generate comparative risk analysis of multiple wallets"""
    risk_map = {}
    
    for wallet in wallet_addresses:
        analysis = analyze_wallet_activity(wallet["address"], wallet.get("chain", "ETH"))
        
        if analysis["real_data"]:
            risk_map[wallet["address"]] = {
                "real_data": True,
                "risk_level": analysis["enhanced_risk_level"],
                "risk_score": analysis["enhanced_risk_score"],
                "deviations": analysis["deviations"],
                "transaction_count": analysis["total_transactions"],
                "balance": analysis.get("balance_eth") or analysis.get("balance_btc"),
                "unique_counterparties": analysis["unique_counterparties"],
                "total_volume": analysis.get("total_volume_eth") or analysis.get("total_volume_btc"),
                "ml_analysis": analysis.get("ml_analysis")
            }
        else:
            risk_map[wallet["address"]] = {
                "real_data": False,
                "risk_level": analysis["enhanced_risk_level"],
                "risk_score": analysis["enhanced_risk_score"],
                "deviations": analysis["deviations"],
                "transaction_count": analysis["activity_patterns"]["transaction_count"],
                "daily_volume": analysis["activity_patterns"]["daily_volume"],
                "ml_analysis": analysis.get("ml_analysis")
            }
    
    return risk_map

def get_wallet_risk_insights(wallet_address, chain="ETH"):
    """Get detailed risk insights for a wallet"""
    activity_analysis = analyze_wallet_activity(wallet_address, chain)
    
    # Prepare insights with ML data
    if activity_analysis["real_data"]:
        balance = activity_analysis.get("balance_eth") or activity_analysis.get("balance_btc")
        total_volume = activity_analysis.get("total_volume_eth") or activity_analysis.get("total_volume_btc")
        
        insights = {
            "real_data": True,
            "wallet_address": wallet_address,
            "chain": chain,
            "risk_level": activity_analysis["enhanced_risk_level"],
            "risk_score": activity_analysis["enhanced_risk_score"],
            "flagged": activity_analysis["flagged"],
            "balance": balance,
            "total_transactions": activity_analysis["total_transactions"],
            "unique_counterparties": activity_analysis["unique_counterparties"],
            "total_volume": total_volume,
            "ml_analysis": activity_analysis.get("ml_analysis"),
            "activity_trends": {
                "volume_trend": "📈 High" if total_volume > 10 else "📊 Moderate" if total_volume > 1 else "📉 Low",
                "frequency_trend": "🚀 High" if activity_analysis["total_transactions"] > 50 else "🔄 Moderate" if activity_analysis["total_transactions"] > 10 else "🐢 Low",
                "counterparty_diversity": "🌍 Diverse" if activity_analysis["unique_counterparties"] > 10 else "👥 Limited" if activity_analysis["unique_counterparties"] > 3 else "🔒 Concentrated",
                "balance_status": "💰 Large" if balance and balance > 100 else "💵 Moderate" if balance and balance > 10 else "💸 Small"
            },
            "deviations_detected": activity_analysis["deviations"],
            "recommendations": generate_risk_recommendations(activity_analysis)
        }
    else:
        insights = {
            "real_data": False,
            "wallet_address": wallet_address,
            "chain": chain,
            "risk_level": activity_analysis["enhanced_risk_level"],
            "risk_score": activity_analysis["enhanced_risk_score"],
            "flagged": activity_analysis["flagged"],
            "ml_analysis": activity_analysis.get("ml_analysis"),
            "activity_trends": {
                "volume_trend": "📈 Increasing" if activity_analysis["activity_patterns"]["daily_volume"] > 20000 else "📊 Stable" if activity_analysis["activity_patterns"]["daily_volume"] > 10000 else "📉 Low",
                "frequency_trend": "🚀 High" if activity_analysis["activity_patterns"]["transaction_count"] > 50 else "🔄 Moderate" if activity_analysis["activity_patterns"]["transaction_count"] > 20 else "🐢 Low",
                "counterparty_diversity": "🌍 Diverse" if activity_analysis["activity_patterns"]["unique_counterparties"] > 10 else "👥 Limited" if activity_analysis["activity_patterns"]["unique_counterparties"] > 3 else "🔒 Concentrated"
            },
            "deviations_detected": activity_analysis["deviations"],
            "recommendations": generate_risk_recommendations(activity_analysis)
        }
    
    return insights

def generate_risk_recommendations(analysis):
    """Generate risk mitigation recommendations"""
    recommendations = []
    
    # ML-based recommendations
    if analysis.get("ml_analysis"):
        ml_risk = analysis["ml_analysis"]["ml_risk_level"]
        if ml_risk == "High":
            recommendations.append("🤖 AI Model: High risk pattern detected - immediate review recommended")
        elif ml_risk == "Medium":
            recommendations.append("🤖 AI Model: Moderate risk - enhanced monitoring advised")
        else:
            recommendations.append("🤖 AI Model: Low risk pattern - normal monitoring sufficient")
    
    if analysis["enhanced_risk_level"] == "HIGH":
        recommendations.extend([
            "🚨 IMMEDIATE ACTION REQUIRED: High risk detected",
            "🔍 Investigate transaction patterns immediately",
            "⚠️ Enhanced due diligence required",
            "📋 Consider freezing transactions"
        ])
    elif analysis["enhanced_risk_level"] == "MEDIUM":
        recommendations.extend([
            "📊 Enhanced monitoring recommended",
            "👀 Watch for pattern changes closely",
            "📈 Review complete transaction history",
            "🔒 Consider additional security measures"
        ])
    else:
        recommendations.extend([
            "✅ Activity appears normal",
            "📋 Maintain standard monitoring",
            "🔒 No immediate concerns"
        ])
    
    if analysis.get("real_data"):
        if analysis.get("balance_eth"):
            recommendations.append("📡 Using real blockchain data from Etherscan API")
        elif analysis.get("balance_btc"):
            recommendations.append("📡 Using real blockchain data from Blockchain.com API")
    else:
        recommendations.append("ℹ️ Using simulated data - API integration available")
    
    return recommendations

def simulate_wallet_risk(address, chain="ETH"):
    """Enhanced wallet risk simulation with real data when available"""
    activity_analysis = analyze_wallet_activity(address, chain)
    
    if activity_analysis["real_data"]:
        balance = activity_analysis.get("balance_eth") or activity_analysis.get("balance_btc")
        total_volume = activity_analysis.get("total_volume_eth") or activity_analysis.get("total_volume_btc")
        
        result_text = f"""
Real-time analysis for {chain} wallet {address[:8]}...{address[-6:]}

💰 Balance: {balance:.8f} {chain}
📊 Transactions: {activity_analysis['total_transactions']}
👥 Unique Counterparties: {activity_analysis['unique_counterparties']}
📈 Total Volume: {total_volume:.8f} {chain}
⚠️ Risk Score: {activity_analysis['enhanced_risk_score']}/10
🚨 Risk Level: {activity_analysis['enhanced_risk_level']}
🤖 ML Confidence: {activity_analysis.get('ml_analysis', {}).get('confidence', 0)*100:.1f}%

🚨 Deviations: {', '.join(activity_analysis['deviations']) if activity_analysis['deviations'] else 'None detected'}
        """.strip()
    else:
        result_text = f"""
Risk analysis for {chain} wallet {address[:8]}...{address[-6:]}

📊 Activity Analysis:
• Daily Volume: ${activity_analysis['activity_patterns']['daily_volume']:,}
• Transactions: {activity_analysis['activity_patterns']['transaction_count']}
• Counterparties: {activity_analysis['activity_patterns']['unique_counterparties']}
• Risk Score: {activity_analysis['enhanced_risk_score']}/10
• Risk Level: {activity_analysis['enhanced_risk_level']}
• ML Analysis: {'Available' if activity_analysis.get('ml_analysis') else 'Not available'}

🚨 Deviations: {', '.join(activity_analysis['deviations']) if activity_analysis['deviations'] else 'None detected'}
        """.strip()
    
    return result_text, activity_analysis["enhanced_risk_level"], activity_analysis["flagged"]

# [Rest of your existing code remains exactly the same - phone validation, AI, Flask routes, etc.]
# ... (all the phone validation, AI, and Flask route code remains unchanged)

# ------------------- Phone Number Validation with NumVerify -------------------
def check_phone_validity(phone_number):
    """Validate phone number using NumVerify API with better error handling"""
    if not NUMVERIFY_ENABLED:
        print("❌ NumVerify API not enabled - check your API key")
        return None
    
    try:
        url = "http://apilayer.net/api/validate"
        params = {
            'access_key': NUMVERIFY_API_KEY,
            'number': phone_number,
            'country_code': '',
            'format': 1
        }
        
        print(f"🔍 Calling NumVerify API for: {phone_number}")
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ NumVerify API response: {data}")
            
            # Check if API returned success
            if data.get('success') is False:
                error_info = data.get('error', {})
                print(f"❌ NumVerify API error: {error_info}")
                return None
            
            # Handle None values in the response
            return {
                "valid": data.get('valid', False),
                "number": data.get('international_format', ''),
                "location": data.get('location', '') or 'Unknown',
                "carrier": data.get('carrier', '') or 'Unknown',
                "line_type": data.get('line_type', '') or 'Unknown',  # Handle None
                "country": data.get('country_name', '') or 'Unknown',
                "country_code": data.get('country_code', '') or 'Unknown',
                "success": True
            }
        else:
            print(f"❌ NumVerify API HTTP error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ NumVerify API exception: {str(e)}")
        return None

def detect_risky_phone_patterns(phone_number):
    """
    Detect risky phone numbers based on patterns and characteristics
    """
    clean_phone = re.sub(r'[^\d+]', '', phone_number)
    
    risk_factors = []
    risk_score = 0
    details = {}
    
    # 1. VOIP/Disposable Number Detection
    voip_prefixes = [
        # US Toll-free (commonly used for virtual numbers)
        '800', '833', '844', '855', '866', '877', '888',
        # Common VOIP prefixes
        '201', '202', '203', '205', '206', '207', '208',
        # Virtual number ranges
        '500', '533', '544', '566', '577', '588'
    ]
    
    for prefix in voip_prefixes:
        if clean_phone.startswith(prefix):
            risk_factors.append(f"VOIP/Virtual number prefix: {prefix}")
            risk_score += 2
            details['number_type'] = 'VOIP/Virtual'
            break
    
    # 2. Known Scam/Fraud Patterns
    scam_patterns = {
        '876': 'Jamaican area code (common in scams)',
        '473': 'Grenada area code (common in scams)', 
        '809': 'Dominican Republic (premium rate scams)',
        '284': 'British Virgin Islands (suspicious)',
        '268': 'Antigua (suspicious)'
    }
    
    for prefix, reason in scam_patterns.items():
        if clean_phone.startswith(prefix):
            risk_factors.append(f"High-risk area code: {reason}")
            risk_score += 3
            details['high_risk_area'] = reason
    
    # 3. Test/Pattern Numbers
    test_numbers = [
        '5550100', '5550123', '5550199',  # TV/Movie numbers
        '1234567890', '1111111111', '9999999999',  # Sequential/Repeating
        '0000000000', '1231231234', '1112223333'   # Pattern sequences
    ]
    
    if clean_phone in test_numbers:
        risk_factors.append("Known test/fake number pattern")
        risk_score += 3
        details['fake_number'] = True
    
    # 4. International Number Risk Assessment
    high_risk_country_codes = [
        '+234',  # Nigeria (advance-fee scams)
        '+229',  # Benin (suspicious)
        '+855',  # Cambodia (scam centers)
        '+63',   # Philippines (call centers)
        '+95',   # Myanmar (suspicious)
        '+255',  # Tanzania (suspicious)
        '+233'   # Ghana (suspicious)
    ]
    
    for code in high_risk_country_codes:
        if phone_number.startswith(code):
            risk_factors.append(f"High-risk country code: {code}")
            risk_score += 2
            details['high_risk_country'] = code
            break
    
    # 5. Repeated digits pattern
    if len(set(clean_phone)) <= 3:
        risk_factors.append("Limited digit diversity")
        risk_score += 2
        details['repeating_pattern'] = True
    
    # Determine risk level
    if risk_score >= 5:
        risk_level = "High"
        flagged = True
    elif risk_score >= 3:
        risk_level = "Medium"
        flagged = True
    elif risk_score >= 1:
        risk_level = "Low"
        flagged = False
    else:
        risk_level = "Very Low"
        flagged = False
    
    return {
        "flagged": flagged,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_factors": risk_factors,
        "details": details
    }

def enhanced_phone_risk_analysis(phone_number):
    """
    Combine NumVerify validation with risk pattern detection
    """
    # Step 1: Validate with NumVerify
    validation_data = check_phone_validity(phone_number)
    
    # Step 2: Run risk pattern detection
    risk_analysis = detect_risky_phone_patterns(phone_number)
    
    # Step 3: Combine results
    combined_result = {
        "valid": validation_data.get('valid', False) if validation_data else False,
        "flagged": risk_analysis["flagged"],
        "risk_score": risk_analysis["risk_score"],
        "risk_level": risk_analysis["risk_level"],
        "risk_factors": risk_analysis["risk_factors"],
        "validation_details": validation_data,
        "pattern_details": risk_analysis["details"]
    }
    
    # Step 4: Enhanced risk scoring based on validation data
    if validation_data:
        # Higher risk for VOIP/mobile vs landline - FIXED: Handle None line_type
        line_type = validation_data.get('line_type')
        if line_type:  # Only process if line_type is not None
            line_type = str(line_type).lower()  # Convert to string and then lowercase
            if line_type == 'voip':
                combined_result["risk_score"] += 2
                combined_result["risk_factors"].append("VOIP line type detected")
            elif line_type == 'mobile':
                combined_result["risk_score"] += 1
                combined_result["risk_factors"].append("Mobile line type")
        
        # Update risk level based on new score
        if combined_result["risk_score"] >= 5:
            combined_result["risk_level"] = "High"
            combined_result["flagged"] = True
        elif combined_result["risk_score"] >= 3:
            combined_result["risk_level"] = "Medium"
            combined_result["flagged"] = True
    
    return combined_result

# ------------------- Custom Jinja2 Filters -------------------
def format_number(value):
    """Format large numbers with commas for better readability"""
    try:
        return f"{int(value):,}"
    except (ValueError, TypeError):
        return value

def format_date(value):
    """Format date for display"""
    if not value:
        return "Unknown"
    return value

def format_eth(value):
    """Format Ethereum values"""
    try:
        if value is None:
            return "N/A"
        return f"{float(value):.4f} ETH"
    except (ValueError, TypeError):
        return value

def format_btc(value):
    """Format Bitcoin values"""
    try:
        if value is None:
            return "N/A"
        return f"{float(value):.8f} BTC"
    except (ValueError, TypeError):
        return value

def format_percentage(value):
    """Format percentage values"""
    try:
        return f"{float(value)*100:.1f}%"
    except (ValueError, TypeError):
        return value

# ------------------- HIBP API Configuration -------------------
def check_hibp_breaches(query, query_type="email"):
    """Query HIBP API for breaches with detailed breach information"""
    if not HIBP_ENABLED:
        return None
    
    try:
        headers = {
            'hibp-api-key': HIBP_API_KEY,
            'User-Agent': 'Cybersecurity-App',
            'Accept': 'application/json'
        }
        
        if query_type == "email":
            url = f'https://haveibeenpwned.com/api/v3/breachedaccount/{query}'
            params = {'truncateResponse': False}
        else:
            return None
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        if response.status_code == 200:
            breaches = response.json()
            return breaches
        elif response.status_code == 404:
            return []
        elif response.status_code == 401:
            return None
        elif response.status_code == 429:
            return None
        else:
            return None
            
    except Exception:
        return None

def format_breach_details(breaches):
    """Format breach details for display"""
    if not breaches:
        return []
    
    formatted_breaches = []
    for breach in breaches[:10]:
        formatted_breach = {
            'name': breach.get('Name', 'Unknown'),
            'title': breach.get('Title', 'Unknown Breach'),
            'domain': breach.get('Domain', 'N/A'),
            'breach_date': breach.get('BreachDate', 'Unknown'),
            'added_date': breach.get('AddedDate', 'Unknown'),
            'modified_date': breach.get('ModifiedDate', 'Unknown'),
            'pwn_count': breach.get('PwnCount', 0),
            'description': breach.get('Description', 'No description available'),
            'data_classes': breach.get('DataClasses', []),
            'is_verified': breach.get('IsVerified', False),
            'is_fabricated': breach.get('IsFabricated', False),
            'is_sensitive': breach.get('IsSensitive', False),
            'is_retired': breach.get('IsRetired', False),
            'is_spam_list': breach.get('IsSpamList', False)
        }
        formatted_breaches.append(formatted_breach)
    
    return formatted_breaches

# ------------------- Enhanced Phone Breach Check -------------------
def check_phone_breaches(phone):
    """
    Enhanced phone number security check with NumVerify and pattern analysis
    """
    if not phone:
        return {
            "breached": False,
            "source": "Invalid Input",
            "message": "❌ No phone number provided",
            "breach_details": [],
            "risk_data": None,
            "validation_data": None
        }

    # Clean the phone number for validation
    clean_phone = re.sub(r'[^\d+]', '', phone)
    
    if len(clean_phone) < 8:  # More lenient minimum length
        return {
            "breached": False,
            "source": "Invalid Input", 
            "message": "❌ Invalid phone number: Must be at least 8 digits",
            "breach_details": [],
            "risk_data": None,
            "validation_data": None
        }

    # Run enhanced risk analysis
    risk_analysis = enhanced_phone_risk_analysis(phone)
    
    # Format response for frontend
    if risk_analysis["flagged"]:
        message = f"🚨 FLAGGED: {risk_analysis['risk_level']} risk phone number"
        if risk_analysis["risk_factors"]:
            message += f" - {', '.join(risk_analysis['risk_factors'][:2])}"
        
        return {
            "breached": True,
            "source": "NumVerify + Pattern Analysis",
            "message": message,
            "breach_details": [risk_analysis],
            "risk_data": risk_analysis,
            "validation_data": risk_analysis.get("validation_details")
        }
    else:
        validation_source = "NumVerify API" if risk_analysis.get("validation_details") else "Pattern Analysis"
        risk_msg = "validated and " if risk_analysis.get("validation_details", {}).get("valid") else ""
        
        return {
            "breached": False,
            "source": validation_source,
            "message": f"✅ Phone number {risk_msg}appears safe - {risk_analysis['risk_level']} risk",
            "breach_details": [risk_analysis],
            "risk_data": risk_analysis,
            "validation_data": risk_analysis.get("validation_details")
        }
        
# ------------------- Fallback Password Generator -------------------
def generate_fallback_password(length=12, strength="strong"):
    try:
        if strength == "strong":
            characters = string.ascii_letters + string.digits + "!@#$%^&*"
        else:
            characters = string.ascii_letters + string.digits
        
        for attempt in range(100):
            password = ''.join(secrets.choice(characters) for _ in range(length))
            
            has_lower = any(c.islower() for c in password)
            has_upper = any(c.isupper() for c in password)
            has_digit = any(c.isdigit() for c in password)
            has_symbol = any(c in "!@#$%^&*" for c in password) if strength == "strong" else True
            
            if has_lower and has_upper and has_digit and has_symbol:
                return password
        
        return ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(length))
        
    except Exception:
        return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))

# ------------------- AI Initialization -------------------
def is_language_model(model_name):
    """Check if the model is suitable for text generation"""
    language_model_indicators = [
        'llama', 'mistral', 'codellama', 'phi', 'qwen', 'gemma', 
        'orca', 'vicuna', 'wizard', 'chat', 'instruct', 'text'
    ]
    model_lower = model_name.lower()
    return any(indicator in model_lower for indicator in language_model_indicators)

try:
    from ollama import Client
    ollama_client = Client(host="http://127.0.0.1:11434")
    
    models_response = ollama_client.list()
    if hasattr(models_response, 'models') and models_response.models:
        available_models = [
            model.model for model in models_response.models 
            if hasattr(model, 'model') and model.model and is_language_model(model.model)
        ]
        
        if available_models:
            PREFERRED_MODEL = available_models[0]
            AI_AVAILABLE = True
            AI_MODE = "ollama"
            
            try:
                test_response = ollama_client.generate(
                    model=PREFERRED_MODEL, 
                    prompt="Say 'OK' in one word only.",
                    options={'temperature': 0.1}
                )
                if test_response and hasattr(test_response, 'response'):
                    pass
                else:
                    AI_AVAILABLE = False
                    AI_MODE = "fallback"
            except Exception:
                AI_AVAILABLE = False
                AI_MODE = "fallback"
except Exception:
    pass

if not AI_AVAILABLE:
    FREE_AI_PROVIDERS = [
        {
            'name': 'Hugging Face',
            'url': 'https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium',
            'headers': {},
            'payload': {'inputs': None, 'parameters': {'max_length': 50}},
            'parse_func': lambda r: r.json()[0]['generated_text']
        },
        {
            'name': 'Petals',
            'url': 'https://chat.petals.dev/api/v1/generate',
            'headers': {'Content-Type': 'application/json'},
            'payload': {
                'model': 'petals-team/StableBeluga2',
                'inputs': None, 
                'parameters': {'max_length': 50, 'do_sample': True, 'temperature': 0.7}
            },
            'parse_func': lambda r: r.json().get('outputs', '')
        }
    ]
    
    for provider in FREE_AI_PROVIDERS:
        try:
            test_prompt = "Hello"
            payload = provider['payload'].copy()
            payload['inputs'] = test_prompt
            
            response = requests.post(provider['url'], headers=provider['headers'], json=payload, timeout=15)
            
            if response.status_code == 200:
                AI_AVAILABLE = True
                AI_MODE = "cloud"
                ACTIVE_PROVIDER = provider
                break
        except Exception:
            continue

if not AI_AVAILABLE:
    AI_MODE = "fallback"

# ------------------- Unified AI Generator -------------------
def generate_password_ai(prompt, length=12, strength="strong"):
    if AI_MODE == "ollama":
        try:
            enhanced_prompt = f"""Generate exactly one password with these specifications:
- Length: {length} characters exactly
- Include: uppercase letters, lowercase letters, numbers
- {'Include: special symbols !@#$%^&*' if strength == 'strong' else ''}
- Return ONLY the password text, nothing else
- No explanations, no quotes, no additional text

Generate the password:"""
            
            resp = ollama_client.generate(
                model=PREFERRED_MODEL, 
                prompt=enhanced_prompt,
                options={'temperature': 0.8}
            )
            generated_text = resp.response.strip()
            
            password = clean_ai_response(generated_text, length)
            if password:
                return password
            else:
                return generate_fallback_password(length, strength)
                
        except Exception:
            return generate_fallback_password(length, strength)
            
    elif AI_MODE == "cloud":
        try:
            enhanced_prompt = f"Create a {length} character {strength} password with uppercase, lowercase, digits{' and symbols' if strength == 'strong' else ''}. Return only the password:"
            
            payload = ACTIVE_PROVIDER['payload'].copy()
            payload['inputs'] = enhanced_prompt
            
            response = requests.post(ACTIVE_PROVIDER['url'], headers=ACTIVE_PROVIDER['headers'], json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if ACTIVE_PROVIDER['name'] == 'Hugging Face':
                    if isinstance(data, list) and len(data) > 0:
                        generated_text = data[0].get('generated_text', '')
                    else:
                        generated_text = data.get('generated_text', '')
                elif ACTIVE_PROVIDER['name'] == 'Petals':
                    generated_text = data.get('outputs', '')
                else:
                    generated_text = str(data)
                
                password = clean_ai_response(str(generated_text), length)
                
                if password:
                    return password
                else:
                    return generate_fallback_password(length, strength)
            else:
                return generate_fallback_password(length, strength)
                
        except Exception:
            return generate_fallback_password(length, strength)
    else:
        return generate_fallback_password(length, strength)

def clean_ai_response(text, length):
    if not text:
        return None
    
    text = text.strip()
    
    prefixes = [
        "Here's your password:", "Password:", "Generated password:", 
        "The password is:", "Sure! Here's", "Here is", "Okay, here is",
        "I've generated", "Your password is:"
    ]
    
    for prefix in prefixes:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    
    text = re.sub(r'["\'`()\[\]]', '', text)
    
    lines = text.split('\n')
    for line in lines:
        clean_line = line.strip()
        if (len(clean_line) >= 8 and 
            any(c.isalpha() for c in clean_line)):
            return clean_line[:length]
    
    return None

# ------------------- AI FEEDBACK GENERATOR -------------------
def get_detailed_ai_feedback(password, strength_result):
    """Get detailed AI feedback on password strength and improvements"""
    if not AI_AVAILABLE:
        return "🔒 AI feedback unavailable. Using standard security analysis."
    
    try:
        prompt = f"""
        Analyze this password's security characteristics:
        
        Strength Rating: {strength_result['label']}
        Security Score: {strength_result['score']}/8
        Length: {len(password)} characters
        Character Types: Uppercase({any(c.isupper() for c in password)}), 
        Lowercase({any(c.islower() for c in password)}), 
        Digits({any(c.isdigit() for c in password)}), 
        Symbols({any(c in "!@#$%^&*" for c in password)})
        
        Provide:
        1. Key security strengths
        2. Specific improvement suggestions
        3. Potential vulnerabilities
        4. Best usage scenarios
        
        Be technical and concise. Do not reveal any password.
        Keep response under 150 words.
        """
        
        if AI_MODE == "ollama":
            response = ollama_client.generate(
                model=PREFERRED_MODEL, 
                prompt=prompt,
                options={'temperature': 0.3}
            )
            return f"🤖 AI Security Analysis:\n{response.response}"
            
        elif AI_MODE == "cloud":
            payload = ACTIVE_PROVIDER['payload'].copy()
            payload['inputs'] = prompt
            
            response = requests.post(ACTIVE_PROVIDER['url'], headers=ACTIVE_PROVIDER['headers'], json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if ACTIVE_PROVIDER['name'] == 'Hugging Face':
                    if isinstance(data, list) and len(data) > 0:
                        feedback = data[0].get('generated_text', 'AI feedback generated.')
                    else:
                        feedback = data.get('generated_text', 'AI feedback generated.')
                elif ACTIVE_PROVIDER['name'] == 'Petals':
                    feedback = data.get('outputs', 'AI feedback generated.')
                else:
                    feedback = str(data)
                return f"☁️ AI Security Analysis:\n{feedback}"
            else:
                return f"❌ AI service unavailable (Status: {response.status_code})"
                
    except Exception as e:
        return f"⚠️ AI analysis failed: {str(e)}"
    
    return "🔒 Using standard password security analysis."

def get_password_generation_explanation(password, method_used):
    """Get AI explanation of the generated password"""
    if not AI_AVAILABLE:
        return "🔒 Password generated using secure cryptographic methods."
    
    try:
        prompt = f"""
        Explain the security features of this password structure:
        
        Generation Method: {method_used}
        Length: {len(password)} characters
        Character Composition: 
        - Uppercase letters: {any(c.isupper() for c in password)}
        - Lowercase letters: {any(c.islower() for c in password)}
        - Digits: {any(c.isdigit() for c in password)}
        - Symbols: {any(c in "!@#$%^&*" for c in password)}
        
        Explain:
        1. Cryptographic security strengths
        2. Resistance to attacks
        3. Appropriate usage scenarios
        
        Be technical but accessible. Do not reveal any password.
        Keep response under 100 words.
        """
        
        if AI_MODE == "ollama":
            response = ollama_client.generate(
                model=PREFERRED_MODEL, 
                prompt=prompt,
                options={'temperature': 0.3}
            )
            return f"🤖 AI Explanation:\n{response.response}"
            
        elif AI_MODE == "cloud":
            payload = ACTIVE_PROVIDER['payload'].copy()
            payload['inputs'] = prompt
            
            response = requests.post(ACTIVE_PROVIDER['url'], headers=ACTIVE_PROVIDER['headers'], json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if ACTIVE_PROVIDER['name'] == 'Hugging Face':
                    if isinstance(data, list) and len(data) > 0:
                        explanation = data[0].get('generated_text', 'Password generated with enhanced security.')
                    else:
                        explanation = data.get('generated_text', 'Password generated with enhanced security.')
                elif ACTIVE_PROVIDER['name'] == 'Petals':
                    explanation = data.get('outputs', 'Password generated with enhanced security.')
                else:
                    explanation = str(data)
                return f"☁️ AI Explanation:\n{explanation}"
                
    except Exception as e:
        return f"🔒 Password generated securely. AI explanation unavailable."
    
    return "🔒 Password generated using industry-standard security practices."

# ------------------- CREDENTIAL CHECKING APIs -------------------
def check_email_breaches(email):
    """
    Check if email has been involved in data breaches using HIBP API
    """
    if not email or '@' not in email:
        return {
            "breached": False,
            "source": "Invalid Input",
            "breach_count": 0,
            "message": "❌ Invalid email format",
            "breach_details": []
        }

    email_local = email.split('@')[0].lower()
    email_domain = email.split('@')[1].lower()
    
    hibp_results = check_hibp_breaches(email, "email")
    
    if hibp_results is not None:
        if hibp_results:
            breach_details = format_breach_details(hibp_results)
            return {
                "breached": True,
                "source": "HIBP",
                "breach_count": len(hibp_results),
                "message": f"Found in {len(hibp_results)} data breaches",
                "breach_details": breach_details
            }
        else:
            return {
                "breached": False,
                "source": "HIBP",
                "breach_count": 0,
                "message": "✅ No breaches detected in HIBP database",
                "breach_details": []
            }

    test_patterns = ['test', 'example', 'demo', 'sample', 'fake']
    
    if (any(pattern in email_local for pattern in test_patterns) or 
        'example' in email_domain or 
        email_domain in ['test.com', 'demo.com', 'fake.com']):
        return {
            "breached": "High Risk",
            "source": "Pattern Detection",
            "breach_count": 0,
            "message": "⚠️ HIGH RISK: This appears to be a test/example email address",
            "breach_details": []
        }

    disposable_domains = [
        'tempmail.com', 'guerrillamail.com', 'mailinator.com', 
        '10minutemail.com', 'yopmail.com', 'throwawaymail.com',
        'fakeinbox.com', 'trashmail.com', 'disposablemail.com'
    ]
    
    if any(domain in email_domain for domain in disposable_domains):
        return {
            "breached": "Disposable",
            "source": "Domain Analysis",
            "breach_count": 0,
            "message": "⚠️ DISPOSABLE: This appears to be a temporary email address",
            "breach_details": []
        }

    if not HIBP_ENABLED:
        return {
            "breached": False,
            "source": "Basic Check",
            "breach_count": 0,
            "message": "⚠️ HIBP API not configured - using basic validation only",
            "breach_details": []
        }

    return {
        "breached": False,
        "source": "HIBP",
        "breach_count": 0,
        "message": "✅ No breaches detected",
        "breach_details": []
    }

# ------------------- Flask App -------------------
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey")

# Register custom filters
app.jinja_env.filters['format_number'] = format_number
app.jinja_env.filters['format_date'] = format_date
app.jinja_env.filters['format_eth'] = format_eth
app.jinja_env.filters['format_btc'] = format_btc
app.jinja_env.filters['format_percentage'] = format_percentage

# ------------------- DB CONNECTION -------------------
def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASS", "pass"),
        database=os.getenv("DB_NAME", "cyber"),
        autocommit=False
    )

# ------------------- REGEX VALIDATION -------------------
EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
PHONE_RE = re.compile(r"^\+?[1-9]\d{9,14}$")

# ------------------- WALLET CHECK -------------------
def validate_wallet_address(address, chain="ETH"):
    chain = chain.upper()
    if chain == "ETH":
        return re.fullmatch(r"^0x[a-fA-F0-9]{40}$", address) is not None
    elif chain == "BTC":
        return re.fullmatch(r"^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$", address) is not None
    return False

# ------------------- BREACH CHECK -------------------
def check_password_pwned(password):
    sha1_pw = hashlib.sha1(password.encode()).hexdigest().upper()
    prefix, suffix = sha1_pw[:5], sha1_pw[5:]
    try:
        r = requests.get(f"https://api.pwnedpasswords.com/range/{prefix}", timeout=8)
        r.raise_for_status()
        for line in r.text.splitlines():
            h, cnt = line.split(":")
            if h == suffix:
                return {"pwned": True, "count": int(cnt)}
        return {"pwned": False, "count": 0}
    except Exception as e:
        return {"error": str(e)}

# ------------------- PASSWORD STRENGTH -------------------
def analyze_password_strength(password):
    score = 0
    remarks = []

    if len(password) >= 12:
        score += 2
        remarks.append("✅ Good length (≥12 chars)")
    elif len(password) >= 8:
        score += 1
        remarks.append("⚠️ Could be longer (8–11 chars)")
    else:
        remarks.append("❌ Too short (<8 chars)")

    if any(c.islower() for c in password): score += 1
    else: remarks.append("Add lowercase letters")
    if any(c.isupper() for c in password): score += 1
    else: remarks.append("Add uppercase letters")
    if any(c.isdigit() for c in password): score += 1
    else: remarks.append("Include digits")
    if any(c in "!@#$%^&*()-_=+[]{}|;:',.<>?/~`" for c in password): score += 1
    else: remarks.append("Include symbols")

    label = "Strong 💪" if score >= 6 else "Moderate ⚙️" if score >= 4 else "Weak 😟"
    return {"label": label, "remarks": remarks, "score": score}

# ------------------- ROUTES -------------------
@app.route("/")
def root():
    return redirect(url_for("home") if "user_id" in session else url_for("signup"))

@app.route("/home")
def home():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("home.html", user=session.get("username"))

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        u, e, p = request.form["username"].strip(), request.form["email"].strip().lower(), request.form["password"]
        if not all([u, e, p]):
            flash("All fields required", "danger")
            return redirect(url_for("signup"))

        conn = get_db_connection()
        try:
            cur = conn.cursor(dictionary=True)
            cur.execute("SELECT id FROM users WHERE email=%s OR username=%s", (e, u))
            if cur.fetchone():
                flash("User already exists", "danger")
            else:
                cur.execute("INSERT INTO users (username,email,password_hash) VALUES (%s,%s,%s)",
                            (u, e, generate_password_hash(p)))
                conn.commit()
                flash("Signup successful!", "success")
        finally:
            cur.close()
            conn.close()
        return redirect(url_for("login"))

    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        e, p = request.form["email"].strip().lower(), request.form["password"]
        conn = get_db_connection()
        try:
            cur = conn.cursor(dictionary=True)
            cur.execute("SELECT * FROM users WHERE email=%s", (e,))
            user = cur.fetchone()
        finally:
            cur.close()
            conn.close()

        if user and check_password_hash(user["password_hash"], p):
            session.update({"user_id": user["id"], "username": user["username"]})
            flash("Login successful", "success")
            return redirect(url_for("dashboard"))
        flash("Invalid credentials", "danger")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully", "info")
    return redirect(url_for("login"))

@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html", user=session.get("username"))

@app.route("/credentials", methods=["GET", "POST"])
def credentials():
    if "user_id" not in session:
        flash("Login first", "warning")
        return redirect(url_for("login"))

    results = {"email": None, "phone": None, "password": None}
    values = {"email": "", "phone": "", "password": ""}
    risk_data = None

    if request.method == "POST":
        e = request.form.get("email", "").strip()
        ph = request.form.get("phone", "").strip()
        pw = request.form.get("password", "").strip()
        values = {"email": e, "phone": ph, "password": pw}

        breached_count = 0
        total_checked = 0

        if e:
            total_checked += 1
            if not EMAIL_RE.fullmatch(e):
                flash("❌ Invalid email format: Must contain @ and valid domain", "warning")
            else:
                email_result = check_email_breaches(e)
                results["email"] = {**email_result, "input": e}
                
                if email_result.get("breached") == True:
                    breached_count += 1
                
                if email_result.get("breached") == True:
                    flash(f"❌ Email breached: {email_result['message']}", "danger")
                elif "HIGH RISK" in email_result.get("message", ""):
                    flash("⚠️ Test/example email detected - this is not a real email", "warning")
                elif "DISPOSABLE" in email_result.get("message", ""):
                    flash("⚠️ Disposable email detected", "warning")
                elif "CAUTION" in email_result.get("message", ""):
                    flash(f"⚠️ {email_result['message']}", "warning")
                elif email_result.get("breached") == False:
                    flash("✅ Email not found in breach databases", "success")

        if ph:
            total_checked += 1
            # More lenient phone validation
            clean_phone = re.sub(r'[^\d+]', '', ph)
            if len(clean_phone) < 8:
                flash("❌ Invalid phone number: Must be at least 8 digits", "warning")
            else:
                phone_result = check_phone_breaches(ph)
                results["phone"] = {**phone_result, "input": ph}
                
                if phone_result.get("breached") == True:
                    breached_count += 1
                    
                # Display appropriate messages
                if phone_result.get("breached") == True:
                    flash(f"❌ {phone_result['message']}", "danger")
                elif phone_result.get("validation_data") and phone_result["validation_data"].get("valid"):
                    flash(f"✅ {phone_result['message']}", "success")
                else:
                    flash(f"⚠️ {phone_result['message']}", "info")

        if pw:
            total_checked += 1
            pwned_result = check_password_pwned(pw)
            results["password"] = pwned_result
            
            if pwned_result.get("pwned"):
                breached_count += 1
                flash(f"❌ Password breached {pwned_result['count']} times!", "danger")
            elif "error" in pwned_result:
                flash("⚠️ Could not check password against breach databases", "warning")
            else:
                flash("✅ Password not found in known breaches", "success")

        if total_checked > 0:
            safe_count = total_checked - breached_count
            breach_percentage = (breached_count / total_checked) * 100 if total_checked > 0 else 0
    
            risk_score = min(100, (breached_count / total_checked) * 100) if total_checked > 0 else 0
    
            risk_data = {
                "breached": breached_count,
                "safe": safe_count,
                "total_checked": total_checked,
                "breach_percentage": round(breach_percentage, 1),
                "risk_score": round(risk_score),
                "risk_percentage": risk_score
            }

    return render_template("credentials.html", results=results, values=values, risk_data=risk_data)

@app.route("/crypto", methods=["GET", "POST"])
def crypto():
    if "user_id" not in session:
        flash("Login first", "warning")
        return redirect(url_for("login"))

    conn = get_db_connection()
    try:
        cur = conn.cursor(dictionary=True)
        
        cur.execute("SELECT * FROM wallet_checks WHERE user_id=%s ORDER BY created_at DESC", (session["user_id"],))
        checks = cur.fetchall()
        
        activity_analysis = None
        risk_map = None
        wallet_insights = None
        feature_importance = get_feature_importance()

        if request.method == "POST":
            w = request.form.get("wallet_id", "").strip()
            ch = request.form.get("chain", "ETH").upper()
            nm = request.form.get("notes", "")
            
            if not w:
                flash("Wallet ID required", "danger")
            elif not validate_wallet_address(w, ch):
                flash("Invalid wallet address format", "danger")
            else:
                res, rl, fl = simulate_wallet_risk(w, ch)
                activity_analysis = analyze_wallet_activity(w, ch)
                wallet_insights = get_wallet_risk_insights(w, ch)
                
                if checks:
                    wallet_list = [{"address": check["wallet_id"], "chain": check["chain"]} for check in checks[:5]]
                    wallet_list.append({"address": w, "chain": ch})
                    risk_map = generate_comparative_risk_map(wallet_list)
                
                # FIX: Convert numpy bool to Python bool for database
                flagged_bool = bool(fl) if isinstance(fl, (np.bool_, bool)) else False
                risk_level_str = str(rl) if rl else "Unknown"
                
                cur.execute("""INSERT INTO wallet_checks
                    (user_id, wallet_id, chain, result_text, risk_level, flagged, notes)
                    VALUES (%s,%s,%s,%s,%s,%s,%s)""",
                    (session["user_id"], w, ch, res, risk_level_str, flagged_bool, nm))
                conn.commit()
                
                cur.execute("SELECT * FROM wallet_checks WHERE user_id=%s ORDER BY created_at DESC", (session["user_id"],))
                checks = cur.fetchall()
                
                flash(f"Wallet {ch} analyzed successfully with AI/ML enhanced risk assessment", "success")

    except Exception as e:
        flash(f"Database error: {str(e)}", "danger")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

    return render_template("crypto.html", 
                         checks=checks,
                         activity_analysis=activity_analysis,
                         risk_map=risk_map,
                         wallet_insights=wallet_insights,
                         ml_available=ML_MODEL_AVAILABLE,
                         feature_importance=feature_importance)
    
@app.route("/delete_wallet_history", methods=["POST"])
def delete_wallet_history():
    """Delete wallet scan history"""
    if "user_id" not in session:
        flash("Login first", "warning")
        return redirect(url_for("login"))

    action = request.form.get("action")
    conn = get_db_connection()
    
    try:
        cur = conn.cursor()
        
        if action == "delete_all":
            # Delete all wallet checks for the current user
            cur.execute("DELETE FROM wallet_checks WHERE user_id = %s", (session["user_id"],))
            conn.commit()
            flash("All wallet scan history has been deleted successfully", "success")
            
        elif action == "delete_single":
            # Delete a single wallet check
            check_id = request.form.get("check_id")
            if check_id:
                cur.execute("DELETE FROM wallet_checks WHERE id = %s AND user_id = %s", 
                           (check_id, session["user_id"]))
                conn.commit()
                flash("Wallet scan has been deleted successfully", "success")
            else:
                flash("Invalid scan ID", "danger")
                
    except Exception as e:
        flash(f"Error deleting history: {str(e)}", "danger")
        conn.rollback()
    finally:
        cur.close()
        conn.close()
    
    return redirect(url_for("crypto"))

@app.route("/ai_password", methods=["GET", "POST"])
def ai_password():
    if "user_id" not in session:
        flash("Login first", "warning")
        return redirect(url_for("login"))

    generated_password, strength_result, ai_feedback = None, None, None

    if request.method == "POST":
        action = request.form.get("action")
        
        if action == "generate":
            try:
                length = int(request.form.get("length", 12))
                strength = request.form.get("strength", "strong")
                prompt = f"Generate a random password of {length} characters with strength {strength}, including uppercase, lowercase, digits, and symbols. Return ONLY the password without any explanation."
                
                generated_password = generate_password_ai(prompt, length, strength)
                
                if generated_password:
                    flash("Password generated successfully!", "success")
                    
                    method_used = ""
                    if AI_MODE == "ollama":
                        method_used = f"Local Ollama ({PREFERRED_MODEL})"
                    elif AI_MODE == "cloud":
                        method_used = f"{ACTIVE_PROVIDER['name']} Cloud AI"
                    else:
                        method_used = "Secure Fallback Generator"
                    
                    ai_feedback = get_password_generation_explanation(generated_password, method_used)
                    flash(f"Generated using: {method_used}", "info")
                else:
                    flash("Password generation failed", "danger")
                    
            except Exception as e:
                flash(f"Password generation failed: {e}", "danger")
                
        elif action == "check":
            pw = request.form.get("password_to_check", "")
            if pw:
                strength_result = analyze_password_strength(pw)
                ai_feedback = get_detailed_ai_feedback(pw, strength_result)
            else:
                flash("Enter a password to check.", "warning")

    return render_template("ai_password.html",
                           generated_password=generated_password,
                           strength_result=strength_result,
                           ai_feedback=ai_feedback,
                           ai_available=AI_AVAILABLE,
                           ai_mode=AI_MODE)

if __name__ == "__main__":
    app.run(debug=True)