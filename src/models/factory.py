import importlib
from transformers import pipeline
from .llm import ChatLLM, FineTunedLLM

def load_prompts(prompt_configs):
    """Загружает все промпты из файлов в память"""
    prompts = {}
    for key, path in prompt_configs.items():
        with open(path, 'r', encoding='utf-8') as f:
            prompts[key] = f.read().strip()
    return prompts

def create_model(config, prompts=None):
    """Универсальная фабрика моделей"""
    
    if config['type'] == 'sklearn':
        module = importlib.import_module(config['module'])
        model_class = getattr(module, config['class'])
        return model_class(**config.get('params', {}))
    
    elif config['type'] == 'llm_pipeline':
        device = config.get('params', {}).get('device', -1)
        return pipeline(
            task="text-classification",
            model=config['model_name'],
            tokenizer=config['model_name'],
            device=device,
            return_all_scores=True
        )
    
    elif config['type'] == 'llm_chat':
        device = config.get('device', 0)
        prompt_key = config['prompt_key']
        system_prompt = prompts[prompt_key]
        return ChatLLM(config['model_name'], system_prompt, device)
    
    elif config['type'] == 'llm_finetune':
        device = config.get('device', 0)
        return FineTunedLLM(config['model_name'], device)
    
    else:
        raise ValueError(f"Unknown model type: {config['type']}")
