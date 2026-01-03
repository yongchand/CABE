#!/usr/bin/env python3
"""
Monitor ongoing ablation experiments
"""
import os
import sys
from pathlib import Path
import json
from datetime import datetime
import time

def get_latest_log_lines(log_file, n=10):
    """Get last n lines from log file"""
    if not os.path.exists(log_file):
        return []
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            return lines[-n:] if len(lines) >= n else lines
    except:
        return []

def parse_epoch_from_log(log_file):
    """Extract current epoch from log file"""
    lines = get_latest_log_lines(log_file, 50)
    for line in reversed(lines):
        if 'Epoch ' in line and '/' in line:
            try:
                # Extract "Epoch X/Y"
                parts = line.split('Epoch ')[1].split('/')[0].strip()
                return int(parts)
            except:
                continue
    return 0

def get_experiment_status(exp_dir):
    """Get status of an experiment"""
    log_file = Path(exp_dir) / 'training.log'
    
    if not log_file.exists():
        return {'status': 'not_started', 'epoch': 0}
    
    # Check if log file is empty
    if os.path.getsize(log_file) == 0:
        return {'status': 'initializing', 'epoch': 0}
    
    # Parse current epoch
    current_epoch = parse_epoch_from_log(log_file)
    
    # Check if completed
    lines = get_latest_log_lines(log_file, 20)
    for line in lines:
        if 'Final evaluation on test set' in line or 'Test Results:' in line:
            return {'status': 'completed', 'epoch': current_epoch}
    
    # Check if running
    if current_epoch > 0:
        return {'status': 'running', 'epoch': current_epoch}
    
    return {'status': 'initializing', 'epoch': 0}

def monitor_experiments(exp_dir, total_epochs):
    """Monitor all experiments in directory"""
    exp_path = Path(exp_dir)
    
    if not exp_path.exists():
        print(f"❌ Experiment directory not found: {exp_dir}")
        return
    
    # Find all experiment subdirectories
    exp_dirs = [d for d in exp_path.iterdir() if d.is_dir() and 'seed' in d.name]
    exp_dirs = sorted(exp_dirs, key=lambda x: x.name)
    
    if not exp_dirs:
        print(f"⏳ No experiments started yet in {exp_dir}")
        print(f"   Waiting for experiments to begin...")
        return
    
    print("="*80)
    print(f"EXPERIMENT MONITOR - {exp_dir}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    status_counts = {'not_started': 0, 'initializing': 0, 'running': 0, 'completed': 0}
    total_progress = 0
    
    for exp_dir in exp_dirs:
        status_info = get_experiment_status(exp_dir)
        status = status_info['status']
        epoch = status_info['epoch']
        
        status_counts[status] += 1
        
        # Progress bar
        if total_epochs > 0:
            progress = (epoch / total_epochs) * 100
            total_progress += progress
            bar_length = 30
            filled = int(bar_length * epoch // total_epochs)
            bar = '█' * filled + '░' * (bar_length - filled)
        else:
            progress = 0
            bar = '░' * 30
        
        # Status emoji
        if status == 'completed':
            emoji = '✅'
        elif status == 'running':
            emoji = '🔄'
        elif status == 'initializing':
            emoji = '⏳'
        else:
            emoji = '⏸️'
        
        print(f"\n{emoji} {exp_dir.name}")
        print(f"   Status: {status.upper()}")
        print(f"   Progress: [{bar}] {epoch}/{total_epochs} epochs ({progress:.1f}%)")
        
        # Show last few log lines if running
        if status == 'running':
            log_file = exp_dir / 'training.log'
            recent_lines = get_latest_log_lines(log_file, 3)
            if recent_lines:
                print(f"   Recent:")
                for line in recent_lines:
                    line = line.strip()
                    if line and not line.startswith('='):
                        # Truncate long lines
                        if len(line) > 70:
                            line = line[:67] + '...'
                        print(f"     {line}")
    
    # Overall summary
    total_exps = len(exp_dirs)
    overall_progress = total_progress / total_exps if total_exps > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*80}")
    print(f"Total Experiments: {total_exps}")
    print(f"  ✅ Completed:    {status_counts['completed']}")
    print(f"  🔄 Running:      {status_counts['running']}")
    print(f"  ⏳ Initializing: {status_counts['initializing']}")
    print(f"  ⏸️  Not Started:  {status_counts['not_started']}")
    print(f"\nOverall Progress: {overall_progress:.1f}%")
    
    # Time estimate
    if status_counts['running'] > 0 or status_counts['completed'] > 0:
        # Rough estimate: ~30-40 minutes per experiment for 150 epochs
        remaining = total_exps - status_counts['completed']
        if remaining > 0:
            est_minutes = remaining * 35  # 35 min average
            est_hours = est_minutes / 60
            print(f"Estimated Time Remaining: ~{est_hours:.1f} hours ({est_minutes:.0f} minutes)")
    
    print(f"{'='*80}\n")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor ablation experiments')
    parser.add_argument('--exp_dir', type=str, 
                       default='experiments_improved_comparison',
                       help='Experiment directory to monitor')
    parser.add_argument('--epochs', type=int, default=150,
                       help='Total epochs per experiment')
    parser.add_argument('--watch', action='store_true',
                       help='Continuously watch (refresh every 30 seconds)')
    parser.add_argument('--interval', type=int, default=30,
                       help='Refresh interval in seconds (default: 30)')
    
    args = parser.parse_args()
    
    if args.watch:
        try:
            while True:
                # Clear screen (works on Unix-like systems)
                os.system('clear' if os.name == 'posix' else 'cls')
                monitor_experiments(args.exp_dir, args.epochs)
                print(f"🔄 Refreshing in {args.interval} seconds... (Press Ctrl+C to stop)")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n\n👋 Monitoring stopped.")
            sys.exit(0)
    else:
        monitor_experiments(args.exp_dir, args.epochs)

