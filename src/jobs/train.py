import subprocess
import sys

# Ensure required libraries are installed
try:
    import stable_baselines3
    import gymnasium as gym
except ImportError:
    print("Installing stable-baselines3 and gymnasium...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "stable-baselines3", "gymnasium", "xgboost", "mlflow"])
    import stable_baselines3
    import gymnasium as gym

import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
import mlflow.pytorch
from pyspark.sql import SparkSession
from sklearn.model_selection import train_test_split
from stable_baselines3 import DQN
from gymnasium import spaces

class FraudDetectionEnv(gym.Env):
    """
    Custom Environment that follows gym interface for Fraud Detection.
    State: XGBoost probability + original scaled features (amount, category_encoded).
    Action: 0 (Allow), 1 (Review), 2 (Block).
    Reward:
      - Block a fraud: +10
      - Allow a fraud: -100
      - Block a valid: -10
      - Allow a valid: +1
      - Review any: -2 (cost of manual review), then true outcome is evaluated
    """
    metadata = {'render_modes': ['console']}

    def __init__(self, data_df):
        super(FraudDetectionEnv, self).__init__()
        self.data = data_df.reset_index(drop=True)
        self.current_step = 0
        self.max_steps = len(self.data) - 1
        
        # Action space: 0 = Allow, 1 = Review, 2 = Block
        self.action_space = spaces.Discrete(3)
        
        # Observation space: [amount, category_encoded, xgb_prob]
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)

    def step(self, action):
        row = self.data.iloc[self.current_step]
        is_fraud = row['is_fraud']
        
        reward = 0
        if action == 0: # Allow
            reward = -100 if is_fraud else 1
        elif action == 1: # Review
            # Pay a fixed review penalty, but then successfully block if fraud or allow if valid
            reward = -2 + (10 if is_fraud else 1)
        elif action == 2: # Block
            reward = 10 if is_fraud else -10
            
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        
        obs = self._next_observation()
        info = {}
        
        return obs, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        return self._next_observation(), {}

    def _next_observation(self):
        if self.current_step >= len(self.data):
            return np.zeros(3, dtype=np.float32)
        row = self.data.iloc[self.current_step]
        # Normalize for observation
        amount_norm = min(row['amount'] / 2000.0, 1.0) # Cap at 1.0 for NN stability
        return np.array([amount_norm, row['category_encoded'], row['xgb_prob']], dtype=np.float32)

def main():
    spark = SparkSession.builder.getOrCreate()
    
    catalog = "main"
    schema = "fraud_detection_dev"
    table_name = "transactions"
    full_table_name = f"{catalog}.{schema}.{table_name}"
    
    print(f"Reading data from {full_table_name}...")
    df = spark.read.table(full_table_name).toPandas()
    
    # Feature Engineering
    df['category_encoded'] = df['category'].astype('category').cat.codes
    # Normalize category for NN
    df['category_encoded'] = df['category_encoded'] / max(1, df['category_encoded'].max())
    
    features = ['amount', 'category_encoded']
    target = 'is_fraud'
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training XGBoost Model...")
    mlflow.set_experiment("/Users/abhirajkumar/fraud_detection_experiment")
    
    with mlflow.start_run(run_name="Hybrid_XGB_DQN") as run:
        # 1. Train XGBoost
        xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
        xgb_model.fit(X_train, y_train)
        
        # Log XGBoost Model
        mlflow.xgboost.log_model(
            xgb_model, 
            artifact_path="xgb_model", 
            registered_model_name=f"{catalog}.{schema}.xgb_fraud_model"
        )
        
        # 2. Get probabilities for DQN State
        df['xgb_prob'] = xgb_model.predict_proba(df[features])[:, 1]
        
        # 3. Train DQN
        print("Training DQN Model...")
        train_df = df.iloc[X_train.index]
        env = FraudDetectionEnv(train_df)
        
        dqn_model = DQN("MlpPolicy", env, verbose=0, learning_rate=1e-3, buffer_size=10000)
        # Train for a few timesteps (adjust in prod)
        dqn_model.learn(total_timesteps=10000)
        
        # Save DQN model locally to log it
        dqn_model.save("/tmp/dqn_fraud_model")
        
        # Log DQN as an MLflow PyTorch artifact
        mlflow.log_artifact("/tmp/dqn_fraud_model.zip", artifact_path="dqn_model")
        
        # NOTE: To register DQN directly to Unity Catalog, we typically write a custom pyfunc wrapper.
        # For simplicity here, we log the artifact which can be downloaded at inference time.
        
        print("Training complete. Models logged to MLflow and registered to Unity Catalog.")

if __name__ == "__main__":
    main()
