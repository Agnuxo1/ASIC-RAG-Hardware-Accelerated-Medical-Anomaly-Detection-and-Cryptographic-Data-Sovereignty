import requests
import json
import time

IP_ADDRESS = "192.168.0.15"
BASE_URL = f"http://{IP_ADDRESS}"

def unlock_update():
    print(f"[INFO] Connecting to {IP_ADDRESS}...")
    
    # 1. Unlock
    print("[INFO] Setting allowUpdate=1...")
    try:
        # Note: AxeOS sometimes requires allowUpdate to be an integer 1, not boolean true
        payload = {"allowUpdate": 1}
        resp = requests.patch(f"{BASE_URL}/api/system", json=payload)
        print(f"[PATCH] Status: {resp.status_code}, Response: {resp.text}")
    except Exception as e:
        print(f"[ERROR] PATCH failed: {e}")
        return

    # 2. Verify
    print("[INFO] Verifying status...")
    try:
        resp = requests.get(f"{BASE_URL}/api/system/info")
        data = resp.json()
        allow_update = data.get("allowUpdate")
        print(f"[VERIFY] allowUpdate is currently: {allow_update}")
        
        if allow_update == 1 or allow_update is True:
            print("\n[SUCCESS] Firmware update is UNLOCKED! proceeding to upload...")
        else:
            print("\n[FAIL] Firmware update is still LOCKED.")
            
    except Exception as e:
        print(f"[ERROR] GET info failed: {e}")

if __name__ == "__main__":
    unlock_update()
