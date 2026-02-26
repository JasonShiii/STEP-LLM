# calculate Complete-Ratio of the ckpt's generation
import os
import pandas as pd

def calculate_complete_ratio(base_dir, ckpt_path, save_csv_path):
    total_non_empty = 0
    complete_count = 0

    data_folders = [os.path.join(base_dir, folder) for folder in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, folder))]

    for data_dir in data_folders:
        gen_dir = os.path.join(data_dir, "gen")
        gen_files = [f for f in os.listdir(gen_dir) if f.endswith(".step")]

        if len(gen_files) == 0:
            continue  # Skip if no gen step file found

        gen_file_path = os.path.join(gen_dir, gen_files[0])

        # Check if the gen step file is empty
        if os.path.getsize(gen_file_path) == 0:
            continue  # Skip empty file

        total_non_empty += 1

        with open(gen_file_path, "r") as f:
            lines = f.readlines()
            if len(lines) == 0:
                continue
            last_line = lines[-1].strip()

        if last_line == "END-ISO-10303-21;":
            complete_count += 1

    if total_non_empty == 0:
        cr = 0.0
    else:
        cr = complete_count / total_non_empty

    print(f"Total Non-Empty Data: {total_non_empty}")
    print(f"Complete Data: {complete_count}")
    print(f"Complete Ratio (CR): {cr:.4f}")

    # Save result to CSV
    if os.path.exists(save_csv_path):
        df = pd.read_csv(save_csv_path)
    else:
        df = pd.DataFrame(columns=["ckpt_path", "CR"])

    df = pd.concat([df, pd.DataFrame([{"ckpt_path": ckpt_path, "CR": cr}])], ignore_index=True)
    df.to_csv(save_csv_path, index=False)
    print(f"Saved CR to {save_csv_path}")



if __name__ == "__main__":
    # UPDATE these paths to match your local setup:
    base_dir = "./data/STEP_generated/eval_output"       # directory with generated STEP files
    ckpt_path = "./checkpoints/step-llm-qwen3b"          # checkpoint path (for logging)
    save_csv_path = "./eval_ckpt/CR/CR_results.csv"      # output CSV

    calculate_complete_ratio(base_dir, ckpt_path, save_csv_path)
