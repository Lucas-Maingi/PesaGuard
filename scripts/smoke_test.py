import time
import subprocess
import requests
import sys

def main():
    print("="*60)
    # Start FastAPI server using uvicorn in a background process
    print("Starting FastAPI server in the background...")
    server_process = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "src.api:app", 
            "--host", "127.0.0.1", "--port", "8000", "--log-level", "warning"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to start
    print("Waiting for server to initialize...")
    health_url = "http://127.0.0.1:8000/health"
    score_url = "http://127.0.0.1:8000/score"
    
    server_ready = False
    for i in range(15):
        try:
            r = requests.get(health_url)
            if r.status_code == 200:
                server_ready = True
                print("Server is healthy and ready!")
                break
        except Exception:
            pass
        time.sleep(0.5)

    if not server_ready:
        print("Error: Server failed to start in 7.5 seconds.")
        # Print stderr logs
        stdout, stderr = server_process.communicate()
        print("Uvicorn Output:", stdout.decode())
        print("Uvicorn Error:", stderr.decode())
        server_process.kill()
        return

    # Define mock high-risk transaction payload
    # Let's mock a TRANSFER with a large amount that empties the account balance
    # (amount = balance, which is suspicious in fraud, and is a TRANSFER)
    payload = {
        "step": 11,
        "type": "TRANSFER",
        "amount": 450000.0,
        "nameOrig": "C99988877",
        "oldbalanceOrg": 450000.0,
        "newbalanceOrig": 0.0,
        "nameDest": "C88877766",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0
    }

    print("\nSending mock transaction payload to /score...")
    print(f"Payload: {payload}")
    
    try:
        t0 = time.perf_counter()
        response = requests.post(score_url, json=payload)
        t1 = time.perf_counter()
        
        latency_requests = int((t1 - t0) * 1000)
        
        print(f"\nResponse Code: {response.status_code}")
        if response.status_code == 200:
            res_data = response.json()
            print("Response Data:")
            import json
            print(json.dumps(res_data, indent=2))
            
            # Assertions
            print("\nAPI LATENCY VERIFICATION:")
            print(f"  - Latency calculated by client requests library: {latency_requests} ms")
            print(f"  - Latency reported by API internal timer: {res_data['response_time_ms']} ms")
            
            if res_data["response_time_ms"] < 200:
                print("  => SUCCESS: API response time is UNDER the 200ms limit.")
            else:
                print("  => FAILURE: API response time exceeded 200ms limit.")
                
            print("\nRISK EVALUATION CHECK:")
            print(f"  - Risk Tier: {res_data['risk_tier']}")
            print(f"  - Recommendation: {res_data['recommendation']}")
            print(f"  - Top Explanation Signals:")
            for signal in res_data["top_signals"]:
                print(f"    * [{signal['signal']}] (Impact: {signal['impact']}) - {signal['description']}")
        else:
            print("Failed to get 200 OK. Response body:")
            print(response.text)
            
    except Exception as e:
        print(f"Request failed: {e}")
        
    finally:
        # Shutdown server
        print("\nShutting down FastAPI server...")
        server_process.terminate()
        server_process.wait()
        print("Server shutdown completed.")
        print("="*60)

if __name__ == "__main__":
    main()
