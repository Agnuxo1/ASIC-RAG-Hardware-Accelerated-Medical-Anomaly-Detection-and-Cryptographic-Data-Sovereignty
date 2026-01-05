# LV06 Quick Reference Card

## Flash Firmware (5 minutes)

### Web Flasher Method
1. Download firmware: `esp-miner-factory-lv06-v2.6.0.bin`
   - From: https://github.com/matlen67/LuckyMiner-LV06/releases
2. Open: https://un-painted-org.github.io/bitaxe-web-flasher/
3. Connect USB → Click Connect → Select file → Flash

### Command Line Method
```bash
pip install bitaxetool
bitaxetool --firmware ./esp-miner-factory-lv06-v2.6.0.bin
```

---

## Configure WiFi (2 minutes)

1. Connect to WiFi: **"Bitaxe"** (no password)
2. Open: http://192.168.4.1
3. Enter your WiFi credentials → Save → Restart

---

## Configure for Local Bridge (1 minute)

```bash
curl -X PATCH http://LV06_IP/api/system \
  -H "Content-Type: application/json" \
  -d '{"stratumURL":"YOUR_PC_IP","stratumPort":3333}'
```

Or via web interface:
- Stratum URL: `YOUR_PC_IP`
- Stratum Port: `3333`
- Worker: `anomaly.detector`

---

## Run Experiment

```bash
python asic_throughput_experiment_lv06.py
```

Wait for: `[BRIDGE] Authorized - ready for high-speed hashing!`

---

## Expected Results

| Before | After |
|--------|-------|
| 9.45s latency | 0.05s latency |
| 0.11 H/s | 10-20 H/s |
| 60+ hours for 512×512 | 30 minutes |

---

## Useful Commands

```bash
# Check status
curl http://LV06_IP/api/system/info

# Restart device
curl -X POST http://LV06_IP/api/system/restart

# Check firewall (Linux)
sudo ufw allow 3333/tcp
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Can't flash | Try different USB cable |
| No WiFi AP | Wait 2 min, then power cycle |
| Won't connect | Check firewall port 3333 |
| Slow hashes | Verify difficulty = 1e-10 |

---

## Resources

- Firmware: https://github.com/matlen67/LuckyMiner-LV06
- Web Flasher: https://un-painted-org.github.io/bitaxe-web-flasher/
- Discord: https://discord.gg/osmu
