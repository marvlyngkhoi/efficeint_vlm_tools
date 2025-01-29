from transformers.models.idefics3.modeling_idefics3 import Idefics3VisionAttention
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers import AutoProcessor, AutoModelForVision2Seq

import torch
import torch.nn as nn



class Hook:
    def __init__(self):
        self.attention_maps = {}
        self.hooks = []
    def hook_fn(self,module, input, output):
        try:
            attention_output, attention_weight = output
            self.attention_maps[module.name] = attention_weight.to('cpu')
        except:
            attention_output, attention_weight,_ = output # output, attention maps , hidden states output
            self.attention_maps[module.name] = attention_weight.to('cpu')

    def apply_hook(self,model):
        # hooks = []
        for name, module in model.model.named_modules():
            if isinstance(module,(Idefics3VisionAttention,LlamaAttention)):
                module.name = name  # Assign a name to the module for identification
                hook = module.register_forward_hook(self.hook_fn)
                self.hooks.append(hook)
    def get_attention_maps(self):
        return self.attention_maps
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        


def load_model(model_path="HuggingFaceTB/SmolVLM-Base",device='cpu'):
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForVision2Seq.from_pretrained(
        "HuggingFaceTB/SmolVLM-Base",
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2" if device == "cuda" else "eager",
         output_attentions=True,
         return_dict_in_generate=True,
    ).to(device)
    return model, processor
    
