import subprocess
import time
import os

os.makedirs('../logs', exist_ok=True)
output_file = '../logs/gpu_telemetry.csv'

print(f"Logging GPU telemetry to {output_file}...")

with open(output_file, 'w') as f:
    cmd = ['nvidia-smi', '--query-gpu=timestamp,utilization.gpu,temperature.gpu,memory.used', '--format=csv', '-l', '1']
    proc = subprocess.Popen(cmd, stdout=f)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        proc.terminate()
        print("\nLogging stopped.")
