import requests
import os
import sys

# Configuration
IP_ADDRESS = "192.168.0.15"
FIRMWARE_PATH = r"D:\ASIC_ANOMALY_Medical_Research_Memo\Firmware_ASIC_LV06\esp-miner-factory-lv06-v2.6.0.bin"
UPDATE_URL = f"http://{IP_ADDRESS}/update"

def force_update():
    print(f"[INFO] Target: {UPDATE_URL}")
    print(f"[INFO] Firmware: {FIRMWARE_PATH}")
    
    if not os.path.exists(FIRMWARE_PATH):
        print("[ERROR] Firmware file not found!")
        return

    try:
        # standard ESPAsyncWebServer OTA handler usually expects 'update' or 'firmware' field
        files = {
            'update': ('firmware.bin', open(FIRMWARE_PATH, 'rb'), 'application/octet-stream')
        }
        
        # Some implementations need specific headers or fields
        # verify if we need to pass MD5 or similar? usually not for simple handlers.
        
        print("[INFO] Starting upload... this may take 1-2 minutes...")
        resp = requests.post(UPDATE_URL, files=files, timeout=120)
        
        print(f"[UPLOAD] Status: {resp.status_code}")
        print(f"[UPLOAD] Response: {resp.text}")
        
        if resp.status_code == 200:
            print("[SUCCESS] Upload appears successful! Device should reboot.")
        else:
            print("[FAIL] Upload received non-200 status.")
            
    except Exception as e:
        print(f"[ERROR] Upload failed: {e}")

if __name__ == "__main__":
    force_update()
