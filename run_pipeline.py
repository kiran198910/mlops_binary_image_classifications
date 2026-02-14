#!/usr/bin/env python
"""
Full Pipeline Script
Runs the complete MLOps pipeline from data preparation to model serving.
"""

import os
import sys
import argparse
import subprocess


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"📌 {description}")
    print(f"{'='*60}")
    
    result = subprocess.run(command, shell=True)
    
    if result.returncode != 0:
        print(f"❌ Failed: {description}")
        return False
    
    print(f"✅ Completed: {description}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run full MLOps pipeline")
    parser.add_argument("--skip-data", action="store_true", 
                        help="Skip data preparation")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip evaluation")
    parser.add_argument("--sample-data", action="store_true",
                        help="Use sample data instead of full dataset")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--serve", action="store_true",
                        help="Start API server after pipeline")
    args = parser.parse_args()
    
    print("\n" + "🐱🐶 Cats vs Dogs MLOps Pipeline" + "\n")
    
    # Step 1: Data Preparation
    if not args.skip_data:
        data_cmd = "python src/data/download_data.py"
        if args.sample_data:
            data_cmd += " --sample-only"
        
        if not run_command(data_cmd, "Data Preparation"):
            print("Pipeline failed at data preparation stage!")
            return 1
    
    # Step 2: Model Training
    if not args.skip_train:
        train_cmd = f"python src/models/train.py --epochs {args.epochs}"
        
        if not run_command(train_cmd, "Model Training"):
            print("Pipeline failed at training stage!")
            return 1
    
    # Step 3: Model Evaluation
    if not args.skip_eval:
        if not run_command("python src/models/evaluate.py", "Model Evaluation"):
            print("Pipeline failed at evaluation stage!")
            return 1
    
    print("\n" + "="*60)
    print("🎉 Pipeline completed successfully!")
    print("="*60)
    
    # Optional: Start API Server
    if args.serve:
        print("\nStarting API server...")
        run_command("python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000", 
                   "API Server")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
