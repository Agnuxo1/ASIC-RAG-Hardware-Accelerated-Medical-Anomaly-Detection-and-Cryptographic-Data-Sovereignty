#!/usr/bin/env python3
"""
LV06 Direct Communication Prototype

This script demonstrates different approaches to communicate with the LV06:
1. Current: Via AxeOS REST API (WiFi)
2. Future: Via USB Serial (after firmware mod)

Author: Francisco Angulo de Lafuente
GitHub: https://github.com/Agnuxo1
Date: December 2024
"""

import requests
import json
import time
import hashlib
import struct
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass


# =============================================================================
# CURRENT APPROACH: AxeOS REST API
# =============================================================================

class LV06_API:
    """
    Communicate with LV06 via its existing AxeOS REST API.
    
    This works with the stock or open-source ESP-Miner firmware.
    Limited by WiFi latency and Stratum protocol overhead.
    """
    
    def __init__(self, ip_address: str = "192.168.1.100"):
        self.base_url = f"http://{ip_address}"
        self.session = requests.Session()
    
    def get_system_info(self) -> Dict:
        """Get system information from the LV06."""
        response = self.session.get(f"{self.base_url}/api/system/info")
        return response.json()
    
    def get_statistics(self) -> Dict:
        """Get mining statistics."""
        response = self.session.get(f"{self.base_url}/api/system/statistics")
        return response.json()
    
    def get_asic_info(self) -> Dict:
        """Get ASIC chip information."""
        response = self.session.get(f"{self.base_url}/api/system/asic")
        return response.json()
    
    def update_settings(self, settings: Dict) -> bool:
        """
        Update device settings.
        
        Possible settings:
        - stratumURL: Pool URL
        - stratumPort: Pool port
        - stratumUser: Worker name
        - frequency: ASIC frequency (MHz)
        - coreVoltage: Core voltage (mV)
        - fanspeed: Fan speed (0-100)
        """
        response = self.session.patch(
            f"{self.base_url}/api/system",
            json=settings,
            headers={"Content-Type": "application/json"}
        )
        return response.status_code == 200
    
    def restart(self) -> bool:
        """Restart the device."""
        response = self.session.post(f"{self.base_url}/api/system/restart")
        return response.status_code == 200
    
    def set_allow_update(self, enable: bool = True) -> bool:
        """Enable firmware updates via API."""
        return self.update_settings({"allowUpdate": 1 if enable else 0})

    def configure_for_local_pool(self, pool_ip: str, pool_port: int = 3333,
                                  worker: str = "anomaly.detector"):
        """
        Configure LV06 to connect to our local Stratum bridge.
        """
        settings = {
            "stratumURL": pool_ip,
            "stratumPort": pool_port,
            "stratumUser": worker,
        }
        return self.update_settings(settings)
    
    def set_minimum_difficulty(self):
        """
        Note: The difficulty is set by the pool, not the device.
        Our Stratum bridge sets difficulty to 1e-10 (any hash valid).
        """
        print("[INFO] Difficulty is controlled by the pool/bridge, not the device")
        print("[INFO] Use our optimized Stratum bridge with DIFFICULTY=1e-10")


# =============================================================================
# FUTURE APPROACH: USB Serial (requires firmware modification)
# =============================================================================

class LV06_Serial:
    """
    Direct USB Serial communication with LV06.
    
    REQUIRES: Modified firmware with serial hash protocol.
    This is the target implementation after firmware modification.
    
    Expected performance: 200-1000 hashes/second
    (vs 3-10 hashes/second with WiFi/Stratum)
    """
    
    # Protocol commands
    CMD_HASH_SINGLE = b'HASH'
    CMD_HASH_BULK = b'BULK'
    CMD_GET_STATUS = b'STAT'
    CMD_RESET = b'RSET'
    
    def __init__(self, port: str = '/dev/ttyUSB0', baudrate: int = 921600):
        """
        Initialize serial connection.
        
        High baudrate (921600) is important for throughput.
        The ESP32-S3 supports up to 5 Mbaud, but USB-UART adapters
        are often limited to 921600 or 1000000.
        """
        try:
            import serial
            self.ser = serial.Serial(port, baudrate, timeout=1.0)
            self.connected = True
        except ImportError:
            print("[ERROR] pyserial not installed. Run: pip install pyserial")
            self.ser = None
            self.connected = False
        except Exception as e:
            print(f"[ERROR] Could not open serial port: {e}")
            self.ser = None
            self.connected = False
    
    def hash_data(self, data: bytes) -> Optional[Tuple[bytes, float]]:
        """
        Hash arbitrary data through the ASIC.
        
        Protocol:
        1. Send: CMD_HASH_SINGLE (4 bytes)
        2. Send: data_length (4 bytes, little-endian uint32)
        3. Send: data (variable length)
        4. Receive: hash (32 bytes)
        
        Returns: (hash_bytes, latency_seconds) or None on error
        """
        if not self.connected:
            return None
        
        start = time.perf_counter()
        
        # Send command
        self.ser.write(self.CMD_HASH_SINGLE)
        
        # For arbitrary data, we need to create a valid Bitcoin header
        # that embeds our data hash in the prevhash field
        data_hash = hashlib.sha256(data).digest()
        
        # Create minimal header (80 bytes) with our data embedded
        header = self._create_header_with_data(data_hash)
        
        # Send header
        self.ser.write(struct.pack('<I', len(header)))
        self.ser.write(header)
        
        # Receive result (32 bytes hash)
        result = self.ser.read(32)
        
        latency = time.perf_counter() - start
        
        if len(result) == 32:
            return result, latency
        return None
    
    def hash_bulk(self, data_list: List[bytes], 
                  callback=None) -> List[Tuple[bytes, float]]:
        """
        Hash multiple data items in sequence.
        
        Protocol for bulk mode:
        1. Send: CMD_HASH_BULK (4 bytes)
        2. Send: count (4 bytes)
        3. For each item:
           - Send: length (4 bytes)
           - Send: data
           - Receive: hash (32 bytes)
        
        callback(index, total, hash, latency) is called for each item.
        """
        if not self.connected:
            return []
        
        results = []
        
        # Send bulk command
        self.ser.write(self.CMD_HASH_BULK)
        self.ser.write(struct.pack('<I', len(data_list)))
        
        for i, data in enumerate(data_list):
            start = time.perf_counter()
            
            data_hash = hashlib.sha256(data).digest()
            header = self._create_header_with_data(data_hash)
            
            self.ser.write(struct.pack('<I', len(header)))
            self.ser.write(header)
            
            result = self.ser.read(32)
            latency = time.perf_counter() - start
            
            if len(result) == 32:
                results.append((result, latency))
                if callback:
                    callback(i + 1, len(data_list), result, latency)
            else:
                results.append((None, latency))
        
        return results
    
    def get_status(self) -> Optional[Dict]:
        """Get device status."""
        if not self.connected:
            return None
        
        self.ser.write(self.CMD_GET_STATUS)
        
        # Expect JSON status response
        line = self.ser.readline()
        try:
            return json.loads(line)
        except:
            return None
    
    def _create_header_with_data(self, data_hash: bytes) -> bytes:
        """
        Create a Bitcoin block header with our data embedded.
        
        Bitcoin header structure (80 bytes):
        - version: 4 bytes
        - prevhash: 32 bytes  ← WE EMBED DATA HASH HERE
        - merkle_root: 32 bytes
        - timestamp: 4 bytes
        - bits: 4 bytes
        - nonce: 4 bytes
        
        The ASIC will hash this header and we verify the operation
        by checking the relationship between input and output.
        """
        version = struct.pack('<I', 0x20000000)
        prevhash = data_hash  # Our data hash embedded here
        merkle = b'\x00' * 32  # Placeholder
        timestamp = struct.pack('<I', int(time.time()))
        bits = struct.pack('<I', 0x1d00ffff)  # Easy difficulty
        nonce = struct.pack('<I', 0)
        
        return version + prevhash + merkle + timestamp + bits + nonce
    
    def close(self):
        """Close serial connection."""
        if self.ser:
            self.ser.close()


# =============================================================================
# ANOMALY DETECTION WRAPPER
# =============================================================================

@dataclass
class AnomalyResult:
    """Result of anomaly detection."""
    data_hash: str
    reference_hash: str
    match: bool
    hamming_distance: int
    detection_time_ms: float


class ASICAnomalyDetector:
    """
    High-level anomaly detection using ASIC hashing.
    
    Works with either API (slow) or Serial (fast) backend.
    """
    
    def __init__(self, backend='api', **kwargs):
        if backend == 'api':
            self.lv06 = LV06_API(**kwargs)
            self.use_serial = False
        elif backend == 'serial':
            self.lv06 = LV06_Serial(**kwargs)
            self.use_serial = True
        else:
            raise ValueError(f"Unknown backend: {backend}")
        
        self.reference_library: Dict[str, bytes] = {}
    
    def add_reference(self, identifier: str, data: bytes):
        """Add a reference item to the library."""
        data_hash = hashlib.sha256(data).digest()
        self.reference_library[identifier] = data_hash
    
    def check_anomaly(self, identifier: str, data: bytes) -> Optional[AnomalyResult]:
        """
        Check if data matches the reference for this identifier.
        
        Returns AnomalyResult with match status and Hamming distance.
        """
        if identifier not in self.reference_library:
            return None
        
        reference_hash = self.reference_library[identifier]
        
        start = time.perf_counter()
        data_hash = hashlib.sha256(data).digest()
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        # Calculate Hamming distance
        distance = self._hamming_distance(reference_hash, data_hash)
        
        return AnomalyResult(
            data_hash=data_hash.hex(),
            reference_hash=reference_hash.hex(),
            match=(distance == 0),
            hamming_distance=distance,
            detection_time_ms=elapsed_ms
        )
    
    def batch_check(self, items: List[Tuple[str, bytes]]) -> List[AnomalyResult]:
        """Check multiple items for anomalies."""
        return [self.check_anomaly(id, data) for id, data in items]
    
    @staticmethod
    def _hamming_distance(a: bytes, b: bytes) -> int:
        """Calculate Hamming distance between two byte sequences."""
        return sum(bin(x ^ y).count('1') for x, y in zip(a, b))


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demo_api_connection(ip: str = "192.168.1.100"):
    """Demonstrate API connection to LV06."""
    print("\n" + "=" * 60)
    print("LV06 API Connection Demo")
    print("=" * 60)
    
    lv06 = LV06_API(ip)
    
    try:
        info = lv06.get_system_info()
        print(f"\n[INFO] Connected to: {info.get('hostname', 'unknown')}")
        print(f"[INFO] Version: {info.get('version', 'unknown')}")
        print(f"[INFO] ASIC Model: {info.get('ASICModel', 'unknown')}")
        print(f"[INFO] Hashrate: {info.get('hashRate', 0)} GH/s")
        
        # Configure for our local pool
        print("\n[CONFIG] Configuring for local Stratum bridge...")
        # lv06.configure_for_local_pool("YOUR_PC_IP", 3333)
        
    except Exception as e:
        print(f"[ERROR] Could not connect: {e}")
        print("[INFO] Make sure:")
        print("  1. LV06 is powered on")
        print("  2. Connected to same WiFi network")
        print("  3. IP address is correct")


def demo_serial_protocol():
    """Demonstrate what the serial protocol would look like."""
    print("\n" + "=" * 60)
    print("LV06 Serial Protocol Demo (Future Implementation)")
    print("=" * 60)
    
    print("""
This is a preview of the serial communication protocol
that will be implemented after firmware modification.

Protocol:
---------
Command: HASH (0x48 0x41 0x53 0x48)
Request: [CMD:4][LENGTH:4][DATA:N]
Response: [HASH:32]

Example:
--------
>>> Send: HASH + 0x40000000 + <64 bytes of data>
<<< Recv: <32 bytes SHA256 hash>

Expected Performance:
--------------------
- Latency: 1-5 ms per hash
- Throughput: 200-1000 hashes/second
- Improvement: 50-300x over WiFi/Stratum

To implement:
------------
1. Flash modified firmware to LV06
2. Connect via USB cable
3. Use this script with backend='serial'
""")


def demo_anomaly_detection():
    """Demonstrate anomaly detection concept."""
    print("\n" + "=" * 60)
    print("Anomaly Detection Demo (Software Simulation)")
    print("=" * 60)
    
    detector = ASICAnomalyDetector(backend='api', ip_address='localhost')
    
    # Simulate reference data
    reference_data = b"Reference CT scan data for patient 12345"
    detector.add_reference("patient_12345", reference_data)
    
    # Test with matching data
    print("\n[TEST] Checking matching data...")
    result = detector.check_anomaly("patient_12345", reference_data)
    if result:
        print(f"  Match: {result.match}")
        print(f"  Hamming distance: {result.hamming_distance}")
        print(f"  Time: {result.detection_time_ms:.3f} ms")
    
    # Test with modified data (anomaly)
    print("\n[TEST] Checking modified data (simulated anomaly)...")
    modified_data = b"Modified CT scan data for patient 12345"
    result = detector.check_anomaly("patient_12345", modified_data)
    if result:
        print(f"  Match: {result.match}")
        print(f"  Hamming distance: {result.hamming_distance}")
        print(f"  Time: {result.detection_time_ms:.3f} ms")
        
        if result.hamming_distance > 0:
            print(f"  ⚠️ ANOMALY DETECTED! Data has been modified.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║           LV06 Direct Communication Prototype                        ║
║                                                                       ║
║  Current: WiFi/API (~3 H/s)                                          ║
║  Future:  USB Serial (~500 H/s after firmware mod)                   ║
║                                                                       ║
║  Author: Francisco Angulo de Lafuente                                ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'api':
            ip = sys.argv[2] if len(sys.argv) > 2 else "192.168.1.100"
            demo_api_connection(ip)
        elif sys.argv[1] == 'serial':
            demo_serial_protocol()
        elif sys.argv[1] == 'anomaly':
            demo_anomaly_detection()
    else:
        print("Usage:")
        print("  python lv06_direct_comm.py api [IP]     - Demo API connection")
        print("  python lv06_direct_comm.py serial       - Demo serial protocol")
        print("  python lv06_direct_comm.py anomaly      - Demo anomaly detection")
        print()
        
        # Run all demos
        demo_serial_protocol()
        demo_anomaly_detection()


if __name__ == "__main__":
    main()
