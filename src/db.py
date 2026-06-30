import os
import sqlite3
import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")

def get_connection():
    """
    Returns a database connection. Connects to Supabase PostgreSQL if
    SUPABASE_DB_URL is set, otherwise falls back to a local SQLite database.
    """
    if SUPABASE_DB_URL:
        import psycopg2
        try:
            conn = psycopg2.connect(SUPABASE_DB_URL)
            return conn, "postgresql"
        except Exception as e:
            print(f"Failed to connect to Supabase PostgreSQL, falling back to local SQLite. Error: {e}")
            
    # SQLite fallback
    os.makedirs("data", exist_ok=True)
    sqlite_path = os.path.join("data", "pesaguard.db")
    conn = sqlite3.connect(sqlite_path)
    return conn, "sqlite"

def init_db():
    """
    Initializes the database schema. Creates tables if they do not exist.
    Supports both SQLite and PostgreSQL syntax.
    """
    conn, db_type = get_connection()
    cursor = conn.cursor()
    
    if db_type == "sqlite":
        # Create transactions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT UNIQUE,
                step INTEGER,
                type TEXT,
                amount REAL,
                nameOrig TEXT,
                oldbalanceOrg REAL,
                newbalanceOrig REAL,
                nameDest TEXT,
                oldbalanceDest REAL,
                newbalanceDest REAL,
                fraud_probability REAL,
                anomaly_score REAL,
                ensemble_score REAL,
                risk_tier TEXT,
                recommendation TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create feedback table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT,
                is_fraud_feedback INTEGER,
                feedback_notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    else:  # postgresql
        # Create transactions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id SERIAL PRIMARY KEY,
                transaction_id VARCHAR(100) UNIQUE,
                step INTEGER,
                type VARCHAR(20),
                amount DOUBLE PRECISION,
                nameOrig VARCHAR(50),
                oldbalanceOrg DOUBLE PRECISION,
                newbalanceOrig DOUBLE PRECISION,
                nameDest VARCHAR(50),
                oldbalanceDest DOUBLE PRECISION,
                newbalanceDest DOUBLE PRECISION,
                fraud_probability DOUBLE PRECISION,
                anomaly_score DOUBLE PRECISION,
                ensemble_score DOUBLE PRECISION,
                risk_tier VARCHAR(20),
                recommendation VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create feedback table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id SERIAL PRIMARY KEY,
                transaction_id VARCHAR(100),
                is_fraud_feedback INTEGER,
                feedback_notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
    conn.commit()
    conn.close()
    print(f"Database initialized successfully using [{db_type.upper()}] engine.")

def log_transaction(tx_data: dict, scores: dict):
    """
    Logs a scored transaction into the database.
    """
    conn, db_type = get_connection()
    cursor = conn.cursor()
    
    # Generate a unique transaction id if not provided
    tx_id = tx_data.get("transaction_id") or f"TX_{tx_data['nameOrig']}_{tx_data['step']}_{int(datetime.datetime.now().timestamp())}"
    
    placeholders = "?" if db_type == "sqlite" else "%s"
    sql = f"""
        INSERT OR IGNORE INTO transactions (
            transaction_id, step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
            nameDest, oldbalanceDest, newbalanceDest, fraud_probability, anomaly_score,
            ensemble_score, risk_tier, recommendation
        ) VALUES (
            {', '.join([placeholders]*15)}
        )
    """
    if db_type == "postgresql":
        # PostgreSQL doesn't support "INSERT OR IGNORE" in standard SQL, use "ON CONFLICT DO NOTHING"
        sql = sql.replace("INSERT OR IGNORE", "INSERT").strip() + " ON CONFLICT (transaction_id) DO NOTHING"
        
    values = (
        tx_id,
        int(tx_data["step"]),
        tx_data["type"],
        float(tx_data["amount"]),
        tx_data["nameOrig"],
        float(tx_data["oldbalanceOrg"]),
        float(tx_data["newbalanceOrig"]),
        tx_data["nameDest"],
        float(tx_data["oldbalanceDest"]),
        float(tx_data["newbalanceDest"]),
        float(scores["fraud_probability"]),
        float(scores["anomaly_score"]),
        float(scores["ensemble_score"]),
        scores["risk_tier"],
        scores["recommendation"]
    )
    
    try:
        cursor.execute(sql, values)
        conn.commit()
    except Exception as e:
        print(f"Error logging transaction: {e}")
        conn.rollback()
    finally:
        conn.close()
    
    return tx_id

def log_feedback(transaction_id: str, is_fraud_feedback: int, feedback_notes: str = ""):
    """
    Logs human feedback for a transaction.
    """
    conn, db_type = get_connection()
    cursor = conn.cursor()
    
    placeholders = "?" if db_type == "sqlite" else "%s"
    sql = f"""
        INSERT INTO feedback (transaction_id, is_fraud_feedback, feedback_notes)
        VALUES ({placeholders}, {placeholders}, {placeholders})
    """
    
    try:
        cursor.execute(sql, (transaction_id, int(is_fraud_feedback), feedback_notes))
        conn.commit()
    except Exception as e:
        print(f"Error logging feedback: {e}")
        conn.rollback()
    finally:
        conn.close()

def get_user_history(name_orig: str):
    """
    Fetches historical transaction steps and amounts for a given user nameOrig.
    Used for real-time feature engineering.
    """
    import pandas as pd
    conn, db_type = get_connection()
    
    # We only need 'step' and 'amount' from transactions originated by this user, ordered chronologically
    placeholders = "?" if db_type == "sqlite" else "%s"
    query = f"""
        SELECT step, amount FROM transactions 
        WHERE nameOrig = {placeholders} 
        ORDER BY step ASC
    """
    
    try:
        df = pd.read_sql_query(query, conn, params=(name_orig,))
        return df
    except Exception as e:
        print(f"Error fetching user history: {e}")
        return pd.DataFrame(columns=["step", "amount"])
    finally:
        conn.close()

def get_db_stats():
    """
    Fetches running stats for the Streamlit dashboard:
    - Total processed transactions
    - Total fraud cases flagged (risk_tier in HIGH/CRITICAL)
    - Total monetary value blocked (fraud value)
    - Total feedback loops processed (FPR calibration)
    """
    conn, db_type = get_connection()
    cursor = conn.cursor()
    
    stats = {
        "total_transactions": 0,
        "fraud_flagged": 0,
        "value_blocked": 0.0,
        "false_positives": 0,
        "true_positives": 0,
        "total_feedback": 0
    }
    
    try:
        # 1. Base transaction counts
        cursor.execute("""
            SELECT 
                COUNT(*), 
                SUM(CASE WHEN risk_tier IN ('HIGH', 'CRITICAL') THEN 1 ELSE 0 END),
                SUM(CASE WHEN risk_tier IN ('HIGH', 'CRITICAL') THEN amount ELSE 0.0 END)
            FROM transactions
        """)
        row = cursor.fetchone()
        if row and row[0] > 0:
            stats["total_transactions"] = row[0]
            stats["fraud_flagged"] = row[1] or 0
            stats["value_blocked"] = float(row[2] or 0.0)
            
        # 2. Feedback statistics
        cursor.execute("SELECT COUNT(*) FROM feedback")
        stats["total_feedback"] = cursor.fetchone()[0] or 0
        
        # Compare flags vs human labels to calculate False Positive Rate
        cursor.execute("""
            SELECT 
                SUM(CASE WHEN f.is_fraud_feedback = 0 AND t.risk_tier IN ('HIGH', 'CRITICAL') THEN 1 ELSE 0 END),
                SUM(CASE WHEN f.is_fraud_feedback = 1 AND t.risk_tier IN ('HIGH', 'CRITICAL') THEN 1 ELSE 0 END)
            FROM feedback f
            JOIN transactions t ON f.transaction_id = t.transaction_id
        """)
        row_feedback = cursor.fetchone()
        if row_feedback:
            stats["false_positives"] = row_feedback[0] or 0
            stats["true_positives"] = row_feedback[1] or 0
            
    except Exception as e:
        print(f"Error fetching stats: {e}")
    finally:
        conn.close()
        
    return stats
