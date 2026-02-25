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
    # base_dir = "/home/group/cad_codebased/data/STEP_generated/rag/eval_batch400_ckpt-3668"
    base_dir = "/home/group/cad_codebased/data/STEP_generated/rag_reorder_round/eval_ckpt-5400"
    # ckpt_path = "/home/group/cad_codebased/data/SFT_ckpt/ckpt_outputs_response_rag/10500/ckpt_epoch1_30/checkpoint-3668"
    ckpt_path = "/home/group/cad_codebased/data/SFT_ckpt/ckpt_outputs_reorder_round/checkpoint-5400"
    save_csv_path = "/home/group/cad_codebased/eval_ckpt/CR/CR_ckpts_rr_copy.csv"

    calculate_complete_ratio(base_dir, ckpt_path, save_csv_path)
