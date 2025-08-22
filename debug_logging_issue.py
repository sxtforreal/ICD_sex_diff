#!/usr/bin/env python3
"""
Debug Script for SeqSetVAE Logging Issue
========================================

This script helps identify exactly which two metrics are being logged 
with the same name "val/focal_loss" but different arguments.
"""

import re
import os

def find_duplicate_logging(model_file_path):
    """
    Find all self.log calls with 'val/focal_loss' in the model file
    """
    print(f"Analyzing: {model_file_path}")
    print("=" * 60)
    
    if not os.path.exists(model_file_path):
        print(f"ERROR: File not found: {model_file_path}")
        print("Please update the path to your model.py file")
        return
    
    with open(model_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    focal_loss_logs = []
    
    # Pattern to match self.log calls with val/focal_loss
    patterns = [
        r'self\.log\s*\(\s*["\']val/focal_loss["\']',
        r'self\.log_dict\s*\([^)]*["\']val/focal_loss["\']',
        r'log\s*\(\s*["\']val/focal_loss["\']'
    ]
    
    for i, line in enumerate(lines, 1):
        for pattern in patterns:
            if re.search(pattern, line):
                focal_loss_logs.append({
                    'line_num': i,
                    'line': line.strip(),
                    'context': get_context(lines, i-1, 3)
                })
    
    print(f"Found {len(focal_loss_logs)} potential logging calls for 'val/focal_loss':")
    print()
    
    for idx, log_info in enumerate(focal_loss_logs, 1):
        print(f"OCCURRENCE #{idx}:")
        print(f"  Line {log_info['line_num']}: {log_info['line']}")
        print("  Context:")
        for ctx_line in log_info['context']:
            print(f"    {ctx_line}")
        print()
    
    if len(focal_loss_logs) >= 2:
        print("ANALYSIS:")
        print("=" * 40)
        print("Found multiple logging calls for 'val/focal_loss'.")
        print("This confirms the duplicate logging issue.")
        print()
        print("LIKELY CAUSES:")
        print("1. Same focal_loss calculated and logged in both _step() and validation_step()")
        print("2. Different focal_loss calculations (main + splitdata) using same key name")
        print("3. Conditional logging with different parameters")
        print()
        print("RECOMMENDED FIXES:")
        print("1. Use different key names: 'val/focal_loss_main' vs 'val/focal_loss_split'")
        print("2. Collect all metrics in _step() and log once in validation_step()")
        print("3. Ensure all self.log() calls use identical parameters")
    
    return focal_loss_logs

def get_context(lines, center_idx, context_size):
    """Get context lines around the center line"""
    start = max(0, center_idx - context_size)
    end = min(len(lines), center_idx + context_size + 1)
    
    context = []
    for i in range(start, end):
        marker = ">>>" if i == center_idx else "   "
        context.append(f"{marker} {i+1:4d}: {lines[i].rstrip()}")
    
    return context

def analyze_method_structure(model_file_path):
    """
    Analyze the structure of _step and validation_step methods
    """
    print("\nMETHOD STRUCTURE ANALYSIS:")
    print("=" * 40)
    
    if not os.path.exists(model_file_path):
        return
    
    with open(model_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find _step method
    step_match = re.search(r'def _step\(self[^:]*\):(.*?)(?=def|\Z)', content, re.DOTALL)
    if step_match:
        step_content = step_match.group(1)
        step_logs = re.findall(r'self\.log[^(]*\([^)]*focal_loss[^)]*\)', step_content)
        print(f"_step method contains {len(step_logs)} focal_loss logging calls:")
        for log in step_logs:
            print(f"  - {log}")
        print()
    
    # Find validation_step method  
    val_match = re.search(r'def validation_step\(self[^:]*\):(.*?)(?=def|\Z)', content, re.DOTALL)
    if val_match:
        val_content = val_match.group(1)
        val_logs = re.findall(r'self\.log[^(]*\([^)]*focal_loss[^)]*\)', val_content)
        print(f"validation_step method contains {len(val_logs)} focal_loss logging calls:")
        for log in val_logs:
            print(f"  - {log}")
        print()

def check_splitdata_module(project_path):
    """
    Check if splitdata module exists and might be causing conflicts
    """
    print("\nSPLITDATA MODULE CHECK:")
    print("=" * 40)
    
    # Look for splitdata related files
    splitdata_files = []
    if os.path.exists(project_path):
        for root, dirs, files in os.walk(project_path):
            for file in files:
                if 'split' in file.lower() and file.endswith('.py'):
                    splitdata_files.append(os.path.join(root, file))
    
    if splitdata_files:
        print(f"Found {len(splitdata_files)} potential splitdata files:")
        for file in splitdata_files:
            print(f"  - {file}")
            
            # Check if they contain focal_loss logging
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'focal_loss' in content and 'self.log' in content:
                        print(f"    WARNING: {file} contains focal_loss logging!")
            except:
                pass
    else:
        print("No obvious splitdata files found.")

if __name__ == "__main__":
    # Update these paths to match your setup
    MODEL_FILE = "/home/sunx/data/aiiih/projects/sunx/projects/SeqSetVAE/main/model.py"
    PROJECT_PATH = "/home/sunx/data/aiiih/projects/sunx/projects/SeqSetVAE"
    
    print("SeqSetVAE Duplicate Logging Debugger")
    print("=" * 60)
    print()
    
    # Main analysis
    focal_loss_logs = find_duplicate_logging(MODEL_FILE)
    
    # Method structure analysis
    analyze_method_structure(MODEL_FILE)
    
    # Check for splitdata conflicts
    check_splitdata_module(PROJECT_PATH)
    
    print("\nNEXT STEPS:")
    print("=" * 40)
    print("1. Run this script: python debug_logging_issue.py")
    print("2. Check the output to see exactly where val/focal_loss is logged")
    print("3. Apply the fixes from the solution files I provided")
    print("4. Use different key names or consolidate logging calls")