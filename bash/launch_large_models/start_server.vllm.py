from openai import OpenAI
import subprocess
import sys
import json
import os
import uuid
from typing import Optional

def generate_uuid() -> str:
    """Generate a random UUID string."""
    return str(uuid.uuid4())

def get_server_host() -> str:
    """Get server host from environment variable or use default."""
    return os.getenv('SERVER_HOST', 'localhost')

def build_client(
    path_to_model: str,
    path_to_chat_template: str,
    api_key: str,
    gpu: str,
    port: int = 8000,
    gpu_utilize: float = 0.9,
    stop_tokens: Optional[str] = None
) -> None:
    """Build and start the model server."""
    host = get_server_host()
    
    try:
        client = OpenAI(
            base_url=f"http://{host}:{port}/v1",
            api_key=api_key,
        )
        client.chat.completions.create(
            model=path_to_model,
            messages=[{"role": "user", "content": "Hi!"}],
            max_tokens=20,
            temperature=0.7,
            top_p=0.9,
            stop="<|eot_id|>",
            n=1,
        )
        print('Start the server successfully.')
    except Exception as e:
        print(f"{e}")
    
    gpu_num = len(gpu.split(','))
    
    command = f"CUDA_VISIBLE_DEVICES={gpu} nohup python -m vllm.entrypoints.openai.api_server " \
           f"--host {host} " \
           f"--model {path_to_model} --dtype auto " \
           f"--api-key {api_key} " \
           f"--port {port} --chat-template {path_to_chat_template} --disable-log-stats --tensor-parallel-size {gpu_num}  --gpu_memory_utilization {gpu_utilize} --trust-remote-code > log.launch &"
    
    print(f'\n\n{command}\n\n')
    
    process = subprocess.Popen(command, shell=True)

    client = OpenAI(
        base_url=f"http://{host}:{port}/v1",
        api_key=api_key,
    )
    
    print('Start the server successfully.')

def write_config(config: dict, model_name: str, api_key: str, port: int, GPU: str, gpu_utilize: float, root_to_save: str) -> None:
    """Write configuration to file."""
    host = get_server_host()
    
    new_config = {
        "model_name": model_name,
        "config": {
            "path_to_model": config['path_to_model'],
            "path_to_chat_template": config['path_to_chat_template'],
            "stop_tokens": config['stop_tokens'],
            "api_key": api_key,
            "port": port,
            "host": host,
            "GPU": GPU,
            "gpu_utilize": gpu_utilize
        }
    }
    
    gpus = '_'.join(GPU.split(','))
    
    # Use a more generic file naming convention
    file_name = f"model_{model_name}_gpu_{gpus}_port_{port}.json"
    path_to_save = os.path.join(root_to_save, file_name)
    json.dump(new_config, open(path_to_save, 'w'), indent=4)
    
    print(f"config saved to ---> {path_to_save}")

if __name__ == "__main__":
    print("Please specify:\n1. path_to_config\n2. root_to_save\n3. GPU\n4. port: make sure models use different port number in the same node.\n5. gpu_utilize")
    
    path_to_config = sys.argv[1]
    root_to_save = sys.argv[2]
    GPU = sys.argv[3]
    port = int(sys.argv[4])
    gpu_utilize = float(sys.argv[5])
    
    with open(path_to_config, 'r') as f:
        config = json.load(f)
        print(f"\n\n{config}\n\n")
        model_name = list(config['policy_model'].keys())[0]
        print(f"\n\n{model_name}\n\n")
    
    # Generate a random API key
    api_key = generate_uuid()
    
    # Start the model
    build_client(
        path_to_model=config['policy_model'][model_name]['path_to_model'],
        path_to_chat_template=config['policy_model'][model_name]['path_to_chat_template'],
        api_key=api_key,
        gpu=GPU,
        port=port,
        gpu_utilize=gpu_utilize,
        stop_tokens=config['policy_model'][model_name]['stop_tokens'],
    )
    
    write_config(config['policy_model'][model_name], model_name, api_key, port, GPU, gpu_utilize, root_to_save)

    
    
    
    
    
  
    
    
