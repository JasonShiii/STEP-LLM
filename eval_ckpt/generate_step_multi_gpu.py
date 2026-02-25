import os
import json
import random
import torch
import torch.distributed as dist
from unsloth import FastLanguageModel
from transformers import TextStreamer


def setup_distributed():
    """Initialize distributed training environment"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        # Initialize the process group
        dist.init_process_group(backend='nccl')
        
        # Set the device for this process
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        
        return rank, world_size, local_rank, device
    else:
        # Single GPU mode
        return 0, 1, 0, torch.device('cuda:0')


def cleanup_distributed():
    """Clean up distributed training environment"""
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except:
        pass  # Ignore errors if not initialized


def load_model(ckpt_path, max_seq_length=25000, dtype=None, load_in_4bit=False, device=None):
    """Load model on the specified device"""
    # Set the device before loading if specified
    if device is not None:
        device_index = device.index if hasattr(device, 'index') else int(str(device).split(':')[-1])
        torch.cuda.set_device(device_index)
        # Set default device for new tensors
        with torch.cuda.device(device_index):
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=ckpt_path,
                max_seq_length=max_seq_length,
                dtype=dtype,
                load_in_4bit=load_in_4bit,
            )
    else:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ckpt_path,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    
    # Move model to the specified device BEFORE for_inference
    if device is not None:
        # First, move the model to device
        model = model.to(device)
        # Recursively move all modules to the correct device
        def move_to_device(module, target_device):
            """Recursively move all submodules to target device"""
            for name, child in module.named_children():
                move_to_device(child, target_device)
            # Move parameters
            for param in module.parameters(recurse=False):
                if param.device != target_device:
                    param.data = param.data.to(target_device)
            # Move buffers
            for buffer in module.buffers(recurse=False):
                if buffer.device != target_device:
                    buffer.data = buffer.data.to(target_device)
        
        # Move the entire model recursively
        move_to_device(model, device)
        # Also ensure base_model is moved if it exists (for PEFT models)
        if hasattr(model, 'base_model'):
            move_to_device(model.base_model, device)
        if hasattr(model, 'model'):  # Some models have a 'model' attribute
            move_to_device(model.model, device)
        if hasattr(model, 'transformer'):  # Some models have a 'transformer' attribute
            move_to_device(model.transformer, device)
    
    # Call for_inference after moving to device
    FastLanguageModel.for_inference(model)
    
    # After for_inference, ensure everything is still on the correct device
    if device is not None:
        # Force move again after for_inference (it might have changed some things)
        model = model.to(device)
        # Check and move any remaining parameters/buffers
        for name, param in model.named_parameters():
            if param.device != device:
                param.data = param.data.to(device)
        for name, buffer in model.named_buffers():
            if buffer.device != device:
                buffer.data = buffer.data.to(device)
    
    return model, tokenizer


def generate_step_files(ckpt_path, step_json, base_save_dir, num_gen, rank=0, world_size=1, device=None):
    """
    Generate STEP files using multi-GPU data parallelism
    
    Args:
        ckpt_path: Path to checkpoint
        step_json: Path to JSON file with data
        base_save_dir: Base directory to save results
        num_gen: Number of samples to generate
        rank: Current process rank (0 to world_size-1)
        world_size: Total number of processes/GPUs
        device: Device to use for model
    """
    # Load model on the specified device
    model, tokenizer = load_model(ckpt_path, device=device)
    EOS_TOKEN = tokenizer.eos_token

    header = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION( ( '' ), ' ' );
FILE_NAME( '/vol/tmp/translate-8579754438183730235/5ae5839f3947920fcf80d878.step', '2018-04-29T08:34:40', ( '' ), ( '' ), ' ', ' ', ' ' );
FILE_SCHEMA( ( 'AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }' ) );
ENDSEC;"""

    ABC_prompt = """You are a CAD model generation assistant trained to produce STEP (.step) files based on textual descriptions. Given the following object description and relevant retrieved CAD data, generate a STEP file that accurately represents the described object.


### caption:
{}

### retrieved relavant step file:
{}

### output:
{}"""

    # Load data
    with open(step_json, 'r') as file:
        data = json.load(file)

    # Set random seed for reproducibility (all processes use same seed)
    random.seed(42)
    random_data = random.sample(data, num_gen)
    
    # Distribute data across processes
    # Each process gets a subset: rank, rank+world_size, rank+2*world_size, ...
    process_data = [random_data[i] for i in range(rank, len(random_data), world_size)]
    
    if rank == 0:
        print(f"Total data: {len(random_data)}, distributed across {world_size} GPUs")
    print(f"[GPU {rank}] Processing {len(process_data)} samples (indices: {rank} to {len(random_data)-1} step {world_size})")

    for idx, item in enumerate(process_data):
        id_original = item['id_original']

        # skip already existing files
        save_dir_check = os.path.join(base_save_dir, id_original)
        if os.path.exists(save_dir_check):
            print(f"Skipping {id_original} (already exists).")
            continue

        id_retrieve = item['id_retrieve']
        caption = item['caption']
        rel_step_text = item['relavant_step_file']
        output_gt = item['output']

        full_step_file_gt = header + "\n" + output_gt
        full_step_file_re = header + "\n" + rel_step_text

        # 保存 ground truth 和 retrieved
        save_gt_dir = os.path.join(base_save_dir, id_original, "gt")
        save_re_dir = os.path.join(base_save_dir, id_original, "re")
        os.makedirs(save_gt_dir, exist_ok=True)
        os.makedirs(save_re_dir, exist_ok=True)

        with open(os.path.join(save_gt_dir, f"{id_original}.step"), "w") as f:
            f.write(full_step_file_gt)

        with open(os.path.join(save_re_dir, f"{id_retrieve}.step"), "w") as f:
            f.write(full_step_file_re)

        # 构建prompt
        formatted_prompt = ABC_prompt.format(caption, rel_step_text, "")

        # Move inputs to the correct device
        inputs = tokenizer([formatted_prompt], return_tensors="pt").to(device)
        streamer = TextStreamer(tokenizer)

        # Double-check that inputs are on the correct device
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to(device)

        print(f"[GPU {rank}] Generating STEP file for {id_original}...")
        
        # Ensure model is on the correct device before generation
        model = model.to(device)
        
        generated = model.generate(**inputs, max_new_tokens=14000)
        # generated = model.generate(**inputs, streamer=streamer, max_new_tokens=14800)
        output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        step_data = output_text.split("### output:")[-1].strip()

        full_step_file_gen = header + "\n" + step_data

        save_gen_dir = os.path.join(base_save_dir, id_original, "gen")
        os.makedirs(save_gen_dir, exist_ok=True)
        save_path = os.path.join(save_gen_dir, f"gen_{id_original}.step")
        with open(save_path, "w") as f:
            f.write(full_step_file_gen)

        print(f"[GPU {rank}] STEP file saved to {save_path}")

        if (idx + 1) % 10 == 0:
            print(f"[GPU {rank}] Completed {idx + 1} / {len(process_data)} data.")


if __name__ == "__main__":
    # Initialize distributed training
    rank, world_size, local_rank, device = setup_distributed()
    
    try:
    #ckpt_path = "/home/group/cad_codebased/data/SFT_ckpt/ckpt_outputs_dfs/checkpoint-1800"
    ckpt_path = "/home/group/cad_codebased/data/SFT_ckpt/ckpt_outputs_dfs_Qwen3B/checkpoint-2700"
    step_json = "/home/group/cad_codebased/data/abc_rag/20500_dfs/val.json"    # val.json (2056) has less data than test.json (4066)
    #base_save_dir = "/home/group/cad_codebased/data/STEP_generated/rag_dfs/eval_ckpt-1800"
    base_save_dir = "/home/group/cad_codebased/data/STEP_generated/rag_dfs_multi_gpu/eval_ckpt-2700"
    num_gen = 2047

        if rank == 0:
            print(f"Starting multi-GPU inference with {world_size} GPUs")
            print(f"Checkpoint: {ckpt_path}")
            print(f"Data file: {step_json}")
            print(f"Output directory: {base_save_dir}")
            print(f"Total samples to generate: {num_gen}")
            print("-" * 50)

        generate_step_files(ckpt_path, step_json, base_save_dir, num_gen, 
                          rank=rank, world_size=world_size, device=device)
        
        if rank == 0:
            print("-" * 50)
            print("All processes completed!")
            
    finally:
        cleanup_distributed()


# (cad_llm2) xiangyu@ideas2:/home/group/cad_codebased$ python /home/group/cad_codebased/eval_ckpt/generate_step_initial.py
# 🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
# 🦥 Unsloth Zoo will now patch everything to make training faster!
# Traceback (most recent call last):
#   File "/home/group/cad_codebased/eval_ckpt/generate_step_initial.py", line 10, in <module>
#     import faiss
# ModuleNotFoundError: No module named 'faiss'

# clone the cad_llm2 to cad_llm3 for further experiments