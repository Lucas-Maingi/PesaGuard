import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import joblib

class PesaGuardFeaturePipeline(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Learned statistics from training set
        self.merchant_risk_scores_ = {}
        # We will initialize a default risk score for unknown categories
        self.default_merchant_risk_ = 0.0

    def fit(self, X, y=None):
        """
        Fits the pipeline on a historical training set to learn static parameters 
        such as merchant/destination category risk scores.
        """
        df_fit = X.copy()
        if y is not None:
            df_fit['isFraud'] = y
        else:
            df_fit['isFraud'] = 0

        # Learn merchant risk scores based on destination account prefix
        # In PaySim, nameDest starting with 'M' is a Merchant, 'C' is Customer
        df_fit['dest_prefix'] = df_fit['nameDest'].str[0].fillna('C')
        
        # Calculate fraud rate per destination category prefix
        merchant_stats = df_fit.groupby('dest_prefix')['isFraud'].agg(['count', 'sum'])
        for prefix, row in merchant_stats.iterrows():
            rate = row['sum'] / row['count'] if row['count'] > 0 else 0.0
            self.merchant_risk_scores_[prefix] = float(rate)
            
        self.default_merchant_risk_ = float(df_fit['isFraud'].mean())
        return self

    def transform(self, X):
        """
        Batch processing (Offline Mode): Performs full feature engineering 
        for model training and batch evaluations on a dataset.
        """
        df_trans = X.copy()
        # Sort chronologically to simulate chronological processing
        df_trans = df_trans.sort_values(by='step')
        
        engineered_rows = []
        # Local state dictionary to calculate historical metrics per user chronologically
        user_history = {}

        for idx, row in df_trans.iterrows():
            user = row['nameOrig']
            dest = row['nameDest']
            amt = float(row['amount'])
            step = int(row['step'])
            balance_orig = float(row['oldbalanceOrg'])
            
            # 1. Base Static Features
            hour_of_day = int(step % 24)
            is_round_amount = 1 if (amt % 100 == 0 or amt % 1000 == 0 or amt % 10000 == 0) and amt > 0 else 0
            cross_border = 1 if dest.startswith('M') else 0
            amount_to_balance_ratio = float(amt / (balance_orig + 1e-5))

            # Merchant category risk score
            dest_prefix = dest[0] if len(dest) > 0 else 'C'
            merchant_risk_score = self.merchant_risk_scores_.get(dest_prefix, self.default_merchant_risk_)

            # 2. Dynamic User History Features
            user_amts = []
            user_steps = []
            
            if user in user_history:
                user_amts = user_history[user]['amounts']
                user_steps = user_history[user]['steps']

            # Extract 30-day window (30 days = 720 steps/hours)
            recent_amts = [a for a, s in zip(user_amts, user_steps) if step - s <= 720]
            recent_steps = [s for s in user_steps if step - s <= 720]

            if len(recent_amts) > 0:
                # amount_deviation: Z-score compared to historical average
                avg_amt = float(np.mean(recent_amts))
                std_amt = float(np.std(recent_amts))
                amount_deviation = float((amt - avg_amt) / (std_amt + 1e-5))

                # velocity_1h: number of transactions in last 1 hour
                velocity_1h = int(sum(1 for s in recent_steps if step - s <= 1) + 1) # count current
                
                # velocity_24h: number of transactions in last 24 hours
                velocity_24h = int(sum(1 for s in recent_steps if step - s <= 24) + 1) # count current

                # time_since_last_transaction
                time_since_last_transaction = float(step - user_steps[-1])

                # amount_percentile
                if len(recent_amts) >= 2:
                    amount_percentile = float(sum(1 for a in recent_amts if amt >= a) / len(recent_amts))
                else:
                    amount_percentile = 0.5
            else:
                # First transaction in history
                amount_deviation = 0.0
                velocity_1h = 1
                velocity_24h = 1
                time_since_last_transaction = 999.0 # Flag for first-time transaction
                amount_percentile = 0.5

            # 3. Model Feature Record
            feat_row = {
                'amount_deviation': amount_deviation,
                'velocity_1h': velocity_1h,
                'velocity_24h': velocity_24h,
                'amount_to_balance_ratio': amount_to_balance_ratio,
                'hour_of_day': hour_of_day,
                'is_round_amount': is_round_amount,
                'merchant_risk_score': merchant_risk_score,
                'cross_border': cross_border,
                'time_since_last_transaction': time_since_last_transaction,
                'amount_percentile': amount_percentile,
                'amount': amt,
                'oldbalanceOrg': balance_orig,
                'newbalanceOrig': float(row['newbalanceOrig']),
                'oldbalanceDest': float(row['oldbalanceDest']),
                'newbalanceDest': float(row['newbalanceDest']),
                'is_transfer': 1 if row['type'] == 'TRANSFER' else 0,
                'is_cash_out': 1 if row['type'] == 'CASH_OUT' else 0
            }
            engineered_rows.append(feat_row)

            # Update historical state
            if user not in user_history:
                user_history[user] = {'amounts': [], 'steps': []}
            user_history[user]['amounts'].append(amt)
            user_history[user]['steps'].append(step)

        return pd.DataFrame(engineered_rows)

    def transform_realtime(self, transaction: dict, user_history_df: pd.DataFrame) -> pd.DataFrame:
        """
        Online Mode: Process a single incoming transaction in real-time.
        
        Args:
            transaction: dict containing the keys: 'step', 'type', 'amount', 
                         'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 
                         'nameDest', 'oldbalanceDest', 'newbalanceDest'
            user_history_df: DataFrame of user's historical transactions 
                             containing columns: 'step', 'amount'
        """
        step = int(transaction['step'])
        amt = float(transaction['amount'])
        dest = transaction['nameDest']
        balance_orig = float(transaction['oldbalanceOrg'])

        # 1. Base Static Features
        hour_of_day = int(step % 24)
        is_round_amount = 1 if (amt % 100 == 0 or amt % 1000 == 0 or amt % 10000 == 0) and amt > 0 else 0
        cross_border = 1 if dest.startswith('M') else 0
        amount_to_balance_ratio = float(amt / (balance_orig + 1e-5))

        dest_prefix = dest[0] if len(dest) > 0 else 'C'
        merchant_risk_score = self.merchant_risk_scores_.get(dest_prefix, self.default_merchant_risk_)

        # 2. Dynamic User History Features
        if user_history_df is not None and not user_history_df.empty:
            # Filter history to the 30-day window (720 hours)
            recent_tx = user_history_df[step - user_history_df['step'] <= 720]
            
            recent_amts = recent_tx['amount'].tolist()
            recent_steps = recent_tx['step'].tolist()
            
            if len(recent_amts) > 0:
                avg_amt = float(np.mean(recent_amts))
                std_amt = float(np.std(recent_amts))
                amount_deviation = float((amt - avg_amt) / (std_amt + 1e-5))
                
                velocity_1h = int(sum(1 for s in recent_steps if step - s <= 1) + 1)
                velocity_24h = int(sum(1 for s in recent_steps if step - s <= 24) + 1)
                
                time_since_last_transaction = float(step - recent_steps[-1])
                
                if len(recent_amts) >= 2:
                    amount_percentile = float(sum(1 for a in recent_amts if amt >= a) / len(recent_amts))
                else:
                    amount_percentile = 0.5
            else:
                amount_deviation = 0.0
                velocity_1h = 1
                velocity_24h = 1
                time_since_last_transaction = 999.0
                amount_percentile = 0.5
        else:
            amount_deviation = 0.0
            velocity_1h = 1
            velocity_24h = 1
            time_since_last_transaction = 999.0
            amount_percentile = 0.5

        # 3. Assemble Feature DataFrame (single row)
        feat_row = {
            'amount_deviation': amount_deviation,
            'velocity_1h': velocity_1h,
            'velocity_24h': velocity_24h,
            'amount_to_balance_ratio': amount_to_balance_ratio,
            'hour_of_day': hour_of_day,
            'is_round_amount': is_round_amount,
            'merchant_risk_score': merchant_risk_score,
            'cross_border': cross_border,
            'time_since_last_transaction': time_since_last_transaction,
            'amount_percentile': amount_percentile,
            'amount': amt,
            'oldbalanceOrg': balance_orig,
            'newbalanceOrig': float(transaction['newbalanceOrig']),
            'oldbalanceDest': float(transaction['oldbalanceDest']),
            'newbalanceDest': float(transaction['newbalanceDest']),
            'is_transfer': 1 if transaction['type'] == 'TRANSFER' else 0,
            'is_cash_out': 1 if transaction['type'] == 'CASH_OUT' else 0
        }
        
        return pd.DataFrame([feat_row])
