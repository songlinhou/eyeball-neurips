#!/usr/bin/env python3
"""
Monitor running experiments and display progress
"""

import os
import time
import json
from datetime import datetime

SAVE_DIR = "/content/drive/MyDrive/EyeballProject/classifier_experiment"

def get_latest_log():
    """Find the most recently modified log file"""
    logs_dir = os.path.join(SAVE_DIR, "logs")
    if not os.path.exists(logs_dir):
        return None
    
    log_files = [f for f in os.listdir(logs_dir) if f.endswith('.log')]
    if not log_files:
        return None
    
    latest = max([os.path.join(logs_dir, f) for f in log_files], 
                 key=os.path.getmtime)
    return latest

def tail_log(log_file, n=20):
    """Display last n lines of log file"""
    if not os.path.exists(log_file):
        return []
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    return lines[-n:]

def get_experiment_status():
    """Get status of all experiments"""
    logs_dir = os.path.join(SAVE_DIR, "logs")
    if not os.path.exists(logs_dir):
        return []
    
    experiments = []
    log_files = [f for f in os.listdir(logs_dir) if f.endswith('.log')]
    
    for log_file in sorted(log_files):
        log_path = os.path.join(logs_dir, log_file)
        exp_name = log_file.replace('.log', '')
        
        # Check if metrics file exists
        metrics_file = os.path.join(logs_dir, f"{exp_name}_metrics.json")
        
        status = {
            'name': exp_name,
            'log_file': log_file,
            'started': datetime.fromtimestamp(os.path.getctime(log_path)).strftime('%Y-%m-%d %H:%M:%S'),
            'last_update': datetime.fromtimestamp(os.path.getmtime(log_path)).strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                status['epochs_completed'] = len(metrics.get('train_loss', []))
                if metrics.get('val_acc'):
                    status['best_val_acc'] = max(metrics['val_acc'])
        else:
            status['epochs_completed'] = 0
        
        experiments.append(status)
    
    return experiments

def display_status():
    """Display current experiment status"""
    print("\n" + "="*80)
    print("EXPERIMENT MONITORING DASHBOARD")
    print("="*80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Save Directory: {SAVE_DIR}")
    print("="*80 + "\n")
    
    experiments = get_experiment_status()
    
    if not experiments:
        print("No experiments found yet. Waiting for experiments to start...")
        return
    
    print("EXPERIMENT STATUS:")
    print("-" * 80)
    for exp in experiments:
        print(f"\n{exp['name']}")
        print(f"  Started: {exp['started']}")
        print(f"  Last Update: {exp['last_update']}")
        print(f"  Epochs Completed: {exp.get('epochs_completed', 0)}")
        if 'best_val_acc' in exp:
            print(f"  Best Val Acc: {exp['best_val_acc']:.2f}%")
    
    print("\n" + "-" * 80)
    
    # Show latest log
    latest_log = get_latest_log()
    if latest_log:
        print(f"\nLATEST LOG ({os.path.basename(latest_log)}):")
        print("-" * 80)
        lines = tail_log(latest_log, n=15)
        for line in lines:
            print(line.rstrip())
    
    print("\n" + "="*80)

def monitor_loop(interval=60):
    """Continuously monitor experiments"""
    try:
        while True:
            os.system('clear' if os.name == 'posix' else 'cls')
            display_status()
            print(f"\nRefreshing in {interval} seconds... (Ctrl+C to stop)")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--once':
        display_status()
    else:
        print("Starting continuous monitoring (refresh every 60 seconds)...")
        print("Press Ctrl+C to stop\n")
        time.sleep(2)
        monitor_loop(interval=60)
