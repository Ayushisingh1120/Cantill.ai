import os
import argparse
import time
from src.pipeline import SpendPipeline

def main():
    parser = argparse.ArgumentParser(description="Cantill.ai Spend Intelligence Pipeline & Classifier")
    parser.add_argument("--config", type=str, help="Path to company config YAML file (e.g. configs/company_A.yaml)")
    parser.add_argument("--all", action="store_true", help="Run end-to-end for all configured companies (A and B)")
    parser.add_argument("--user", type=str, default="Ayushisingh1120", help="Candidate name for output file naming")
    parser.add_argument("--zero-shot", action="store_true", help="Force zero-shot classification without using labels")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save output CSV files")
    
    args = parser.parse_args()
    
    # If no specific config is specified, default to running all
    run_all = args.all or (args.config is None)
    
    configs_to_run = []
    if run_all:
        configs_to_run = [
            ("configs/company_A.yaml", "A"),
            ("configs/company_B.yaml", "B")
        ]
    else:
        # Determine tag from config name
        base_name = os.path.basename(args.config)
        tag = "A" if "_A" in base_name else ("B" if "_B" in base_name else "C")
        configs_to_run = [(args.config, tag)]
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    start_total = time.time()
    for cfg_path, tag in configs_to_run:
        if not os.path.exists(cfg_path):
            print(f"Error: Config file {cfg_path} not found!")
            continue
            
        print(f"\n{'='*60}")
        print(f"Starting run for {cfg_path} -> Output Tag: {tag}")
        print(f"{'='*60}")
        
        t0 = time.time()
        pipeline = SpendPipeline(cfg_path, use_zero_shot=args.zero_shot)
        df_out = pipeline.run()
        
        output_filename = f"{args.user}__{tag}.csv"
        output_filepath = os.path.join(args.output_dir, output_filename)
        df_out.to_csv(output_filepath, index=False)
        
        elapsed = time.time() - t0
        print(f"Saved {len(df_out)} rows to {output_filepath} in {elapsed:.1f}s")
        
    print(f"\nAll tasks completed in {time.time() - start_total:.1f}s.")

if __name__ == "__main__":
    main()
