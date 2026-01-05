import os
import sys
import time
import subprocess
import requests
import psutil
from pathlib import Path

# Configuration
MINER_IP = "192.168.0.15"  # Update this if needed, or read from config
MINER_REBOOT_URL = f"http://{MINER_IP}/api/system/reboot"
WAIT_FOR_BOOT_SECONDS = 30
MAX_RETRIES = 10

def kill_other_python_processes():
    """Kills all Python processes except the current one."""
    current_pid = os.getpid()
    print(f"[SAFETY] Current PID: {current_pid}")
    print("[SAFETY] Scanning for other Python processes...")
    
    killed_count = 0
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if 'python' in proc.info['name'].lower() and proc.info['pid'] != current_pid:
                print(f"[SAFETY] Killing process {proc.info['pid']} ({proc.info['name']})")
                proc.kill()
                killed_count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
            
    print(f"[SAFETY] Cleanup complete. Killed {killed_count} processes.")

def reboot_miner():
    """Attempts to reboot the miner via API."""
    print(f"[SAFETY] Attempting to reboot miner at {MINER_IP}...")
    try:
        # Try POST first (common for actions)
        response = requests.post(MINER_REBOOT_URL, timeout=5)
        if response.status_code == 200:
            print("[SAFETY] Reboot command sent successfully (POST).")
            return True
    except:
        pass

    try:
        # Try GET as fallback
        response = requests.get(MINER_REBOOT_URL, timeout=5)
        if response.status_code == 200:
            print("[SAFETY] Reboot command sent successfully (GET).")
            return True
    except Exception as e:
        print(f"[SAFETY] Failed to send reboot command: {e}")
        
    print("[SAFETY] WARNING: Automatic reboot failed.")
    print("[SAFETY] Please manually reboot the LV06 miner now.")
    input("[SAFETY] Press Enter once the miner is rebooting...")
    return False

def wait_for_miner():
    """Waits for the miner to be responsive."""
    print(f"[SAFETY] Waiting for miner to come online ({WAIT_FOR_BOOT_SECONDS}s initial sleep)...")
    time.sleep(WAIT_FOR_BOOT_SECONDS)
    
    for i in range(MAX_RETRIES):
        try:
            response = requests.get(f"http://{MINER_IP}/api/system/info", timeout=5)
            if response.status_code == 200:
                print("[SAFETY] Miner is ONLINE and responsive!")
                return True
        except:
            pass
        
        print(f"[SAFETY] Waiting for miner... (Attempt {i+1}/{MAX_RETRIES})")
        time.sleep(5)
    
    print("[SAFETY] ERROR: Miner did not come online.")
    return False

def run_experiment():
    """Runs the main experiment script."""
    print("[SAFETY] Launching main experiment...")
    print("="*60)
    
    # Forward all arguments to main.py
    # If no arguments provided, default to FULL benchmark with ASIC enabled
    args = sys.argv[1:]
    if not args:
        print("[SAFETY] No arguments provided. Defaulting to: --mode full --enable-asic")
        args = ["--mode", "full", "--enable-asic"]
        
    cmd = [sys.executable, "main.py"] + args
    
    try:
        # Use Popen to stream output
        process = subprocess.Popen(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stderr, # Capture stderr too
            universal_newlines=True,
            bufsize=1 # Line buffering
        )
        
        # Wait for completion
        return_code = process.wait()
        
        if return_code == 0:
            print("\n" + "="*60)
            print("[SAFETY] Experiment completed SUCCESSFULLY.")
        else:
            print("\n" + "="*60)
            print(f"[SAFETY] Experiment failed with exit code {return_code}.")
            
    except Exception as e:
        print(f"[SAFETY] Failed to launch experiment: {e}")

if __name__ == "__main__":
    print("="*60)
    print("ASIC HYBRID BENCHMARK - SAFETY PROTOCOL RUNNER")
    print("="*60)
    
    # 1. Clean environment
    kill_other_python_processes()
    
    # 2. Reboot hardware
    reboot_miner()
    
    # 3. Verify hardware state
    if wait_for_miner():
        # 4. Run experiment
        run_experiment()
    else:
        print("[SAFETY] ABORTING: Miner not ready.")
        sys.exit(1)
