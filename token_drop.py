# Load model directly
from transformers import AutoProcessor, AutoModelForImageTextToText, set_seed
import torch
import gc
import requests
import json
from tqdm import tqdm
import time 
import numpy as np
import time
from sklearn.cluster import KMeans 
import torch.nn.functional as F

def seed_(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False
    set_seed(seed)
    
seed_()
    

# Func to check if url exist
def check_link(url):
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


# Function to load dataset 
def load_textvqa_val(path='TextVQA_0.5.1_val.json',sample_num=500,sys_prompt =  'Answer the question using a single word or phrase.'):
    with open(path,'r') as f:
        data = json.load(f)['data']

    sample_data =[]
    ctr=0
    for sample in tqdm(data):
        if check_link(sample['flickr_original_url']):
            # temp = {
            #     'ans':sample['answers'],
            #     'ques':sample['question'],
            #     'qid':sample['question_id'],
            #     'image':sample['flickr_original_url']
            # }

            msg = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": sys_prompt}]
                },
                {
                    "role": "user",
                    "content": [
                            {"type": "image", "image":sample['flickr_original_url'] },
                            {"type": "text", "text": sample['question']}
                        ]
                }
            ]
            ctr+=1
            sample_data.append([msg,sample['question_id'],sample['answers'],sample['question']])
        if ctr==sample_num:
            return sample_data








def bench(msg):
    
    """Benchmark LLM generation with precise time and memory tracking."""
    # 1. Clear CUDA cache and get baseline memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline_mem = torch.cuda.memory_allocated()  # Memory before processing
    
    # 2. Preprocessing (tokenization, moving to GPU)
    start_preprocess = time.perf_counter()
    
    inputs = processor.apply_chat_template(
        msg, add_generation_prompt=True, 
        tokenize=True, return_dict=True, 
        return_tensors="pt"
    ).to(dtype=torch.bfloat16).to("cuda")
    
    input_len = inputs["input_ids"].shape[-1]
    preprocess_time = time.perf_counter() - start_preprocess
    
    # 3. Generation (track peak memory)
    start_generate = time.perf_counter()
    
    with torch.inference_mode():
        generation = model.generate(
            **inputs, 
            do_sample=False,
            #max_new_tokens=100,  # Optional: Set a fixed token limit
        )
        generation = generation[0][input_len:]
    
    generate_time = time.perf_counter() - start_generate
    
    # 4. Calculate memory usage (GB)
    peak_mem = torch.cuda.max_memory_allocated()  # Peak during generation
    memory_used_gb = (peak_mem - baseline_mem) / (1024 ** 3)  # Delta in GB
    
    # 5. Decode and return results
    decoded = processor.decode(generation, skip_special_tokens=True)
    
    return {
        "output": decoded,
        "metrics": {
            "preprocess_time_sec": round(preprocess_time, 4),
            "generation_time_sec": round(generate_time, 4),
            "total_time_sec": round(preprocess_time + generate_time, 4),
            "memory_used_gb": round(memory_used_gb, 4),  # Total memory consumed
            "peak_memory_gb": round(peak_mem / (1024 ** 3), 4),  # Absolute peak (optional)
            "tokens_generated": len(generation),
            "tokens_per_sec": round(len(generation) / generate_time, 2) if generate_time > 0 else 0,
        }
    }





    
# def random_drop(inputs,ques,sample_seq_len = 40):
#     model.to('cpu')
#     gc.collect()
#     torch.cuda.empty_cache()
    
#     inputs = processor.apply_chat_template(
#         messages, add_generation_prompt=True, tokenize=True,
#             return_dict=True, return_tensors="pt"
#     ).to(dtype=torch.bfloat16)


#     text_inputs_final = processor.tokenizer(
#                 "<bos><start_of_turn>user\nAnswer the question using a single word or phrase.\n\n<start_of_image><end_of_image>" +
#                 ques + "<end_of_turn>\n",
#                 return_tensors='pt'
#             ).to('cpu')

#     with torch.no_grad():
#         vision_outputs = model.vision_tower.to('cuda')(pixel_values=inputs.pixel_values.to('cuda') ).last_hidden_state
#         image_features = model.multi_modal_projector.to('cuda')(vision_outputs).to('cpu')        
#         inputs_embeds = model.language_model.model.embed_tokens.to('cuda')(inputs.input_ids.to('cuda')).to('cpu')

        
#         idx_1 = torch.where(input_ids_final == 255999)[1].item()
#         idx_2 = torch.where(input_ids_final == 256000)[1].item()


#         total_seq_len = image_features.shape[1]  

#         sampled_indices = torch.randperm(total_seq_len)[:sample_seq_len].sort().values
#         filtered_token = image_features[:, sampled_indices, :]
           
#         Text_before = inputs_embeds[:, :idx_1, :]
#         Text_middle = inputs_embeds[:, idx_1:idx_2, :]
#         Text_after = inputs_embeds[:, idx_2:, :]
#         combined = torch.cat([Text_before, filtered_token, Text_middle, Text_after], dim=1)


#     model.to('cpu')
#     gc.collect()
#     torch.cuda.empty_cache()  

#     new_inputs = {'inputs_embeds': combined.to('cuda')}

#     with torch.inference_mode():
#             generation = model.to('cuda').generate(**new_inputs, do_sample=False)



#     decoded = processor.decode(generation, skip_special_tokens=True)


def random_drop(msg, ques, model, processor, sample_seq_len=40):
    """Optimized random drop with full memory cleanup, GC, and AMP."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = model.to(device).eval()  # keep model on GPU once, in eval mode

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device)
    base_mem = torch.cuda.memory_allocated(device)

    with torch.no_grad():#, torch.cuda.amp.autocast():  # disable gradients and use mixed precision
        # --------- Preprocessing ---------
        start_pre = time.perf_counter()
        inputs = (
            processor
            .apply_chat_template(msg, add_generation_prompt=True, tokenize=True,
                                 return_dict=True, return_tensors="pt")
            .to(dtype=torch.bfloat16)
        )
        text_inputs = processor.tokenizer(
            f"<bos><start_of_turn>user\nAnswer in one word or phrase.\n\n"
            f"<start_of_image><end_of_image>{ques}<end_of_turn>\n",
            return_tensors='pt'
        )
        input_ids = text_inputs.input_ids
        idx_1 = torch.where(input_ids == 255999)[1].item()
        idx_2 = torch.where(input_ids == 256000)[1].item()
        preprocess_time = time.perf_counter() - start_pre

        # --------- Feature Extraction ---------
        start_feat = time.perf_counter()
        pixel_values = inputs.pixel_values.to(device, non_blocking=True)
        vision_out = model.vision_tower(pixel_values)
        image_feats = model.multi_modal_projector(vision_out.last_hidden_state)

        del vision_out, pixel_values, inputs
        gc.collect()
        torch.cuda.empty_cache()

        input_ids = input_ids.to(device, non_blocking=True)
        embed_layer = model.language_model.model.embed_tokens
        text_embeds = embed_layer(input_ids)

        del embed_layer, input_ids
        gc.collect()
        torch.cuda.empty_cache()

        # Random image token sampling
        sampled_idx = torch.randperm(image_feats.size(1), device=device)[:sample_seq_len]
        sampled_idx, _ = torch.sort(sampled_idx)
        filtered_img = image_feats[:, sampled_idx, :]

        combined = torch.cat([
            text_embeds[:, :idx_1],
            filtered_img,
            text_embeds[:, idx_1:idx_2],
            text_embeds[:, idx_2:]
        ], dim=1)

        del image_feats, filtered_img, text_embeds
        gc.collect()
        torch.cuda.empty_cache()

        feature_time = time.perf_counter() - start_feat

        # --------- Generation ---------
        start_gen = time.perf_counter()
        outputs = model.generate(inputs_embeds=combined, do_sample=False, max_new_tokens=20)
        torch.cuda.synchronize()
        generate_time = time.perf_counter() - start_gen

    # --------- Postprocessing ---------
    peak_mem = torch.cuda.max_memory_allocated(device)
    decoded = processor.decode(outputs[0].cpu(), skip_special_tokens=True)
    total_time = preprocess_time + feature_time + generate_time

    return {
        "output": decoded,
        "metrics": {
            "preprocess_time_sec": round(preprocess_time, 4),
            "feature_time_sec": round(feature_time, 4),
            "generation_time_sec": round(generate_time, 4),
            "total_time_sec": round(total_time, 4),
            "memory_used_gb": round((peak_mem - base_mem) / (1024**3), 4),
            "peak_memory_gb": round(peak_mem / (1024**3), 4),
            "tokens_generated": outputs.shape[-1],
            "tokens_per_sec": round(outputs.shape[-1] / generate_time, 2) if generate_time > 0 else 0,
            "sampled_tokens": sample_seq_len
        }
    }

    

def k_means_merge(msg, ques, model, processor, sample_seq_len=40):
    """
    Merge the most similar vision tokens via k‑means clustering,
    track precise timings, memory, and input/output token counts.
    """
    # 1) Setup device & model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()

    # 2) Clear caches & record baseline memory
    torch.cuda.empty_cache()      # free cached GPU memory :contentReference[oaicite:3]{index=3}
    gc.collect()                 # free Python‑side garbage :contentReference[oaicite:4]{index=4}
    torch.cuda.reset_peak_memory_stats(device)
    base_mem = torch.cuda.memory_allocated(device)

    with torch.no_grad():  # disable gradient tracking entirely :contentReference[oaicite:5]{index=5}
        # --- Preprocessing: text + vision inputs ---
        start_pre = time.perf_counter()
        inputs = (
            processor
            .apply_chat_template(msg, add_generation_prompt=True, tokenize=True,
                                 return_dict=True, return_tensors="pt")
            .to(dtype=torch.bfloat16)
        )
        text_in = processor.tokenizer(
            f"<bos><start_of_turn>user\nAnswer in one word or phrase.\n\n"
            f"<start_of_image><end_of_image>{ques}<end_of_turn>\n",
            return_tensors='pt'
        )
        input_ids = text_in.input_ids
        idx_1 = torch.where(input_ids == 255999)[1].item()
        idx_2 = torch.where(input_ids == 256000)[1].item()
        preprocess_time = time.perf_counter() - start_pre

        # --- Feature extraction: Vision tower + projection ---
        start_feat = time.perf_counter()
        pix = inputs.pixel_values.to(device, non_blocking=True)
        vout = model.vision_tower(pix)
        img_feats = model.multi_modal_projector(vout.last_hidden_state)  # (1, N, C)
        # cleanup immediately after use
        del vout, pix, inputs  
        gc.collect(); torch.cuda.empty_cache()  # free GPU cache :contentReference[oaicite:6]{index=6}

        # --- K‑means clustering on CPU ---
        feats_np = img_feats[0].cpu().numpy()  # (N, C) :contentReference[oaicite:7]{index=7}
        kmeans = KMeans(n_clusters=sample_seq_len, n_init=10).fit(feats_np)  # :contentReference[oaicite:8]{index=8}
        centroids = torch.tensor(kmeans.cluster_centers_, device=device)    # (k, C)
        merged_tokens = centroids.unsqueeze(0)                             # (1, k, C)
        del img_feats, feats_np, kmeans  
        gc.collect(); torch.cuda.empty_cache()  # free after clustering :contentReference[oaicite:9]{index=9}
        feature_time = time.perf_counter() - start_feat

        # --- Text embeddings ---
        tid = input_ids.to(device, non_blocking=True)
        embed = model.language_model.model.embed_tokens
        text_embeds = embed(tid)
        del embed, tid  
        gc.collect(); torch.cuda.empty_cache()

        # --- Combine text & merged vision tokens ---
        combined = torch.cat([
            text_embeds[:, :idx_1],
            merged_tokens,
            text_embeds[:, idx_1:idx_2],
            text_embeds[:, idx_2:]
        ], dim=1)
        input_length = combined.shape[-2]  # record prompt length
        del text_embeds, merged_tokens  
        gc.collect(); torch.cuda.empty_cache()

        # --- Generation ---
        start_gen = time.perf_counter()
        out = model.generate(
            inputs_embeds=combined,
            do_sample=False,
            max_new_tokens=20,
            # return_dict_in_generate=True  # optional for extra info
        )
        torch.cuda.synchronize()  # ensure all kernels complete before timing
        generate_time = time.perf_counter() - start_gen

    # --- Metrics & decode output ---
    peak_mem = torch.cuda.max_memory_allocated(device)
    decoded = processor.decode(out[0].cpu(), skip_special_tokens=True)
    total_time = preprocess_time + feature_time + generate_time
    tokens_generated = out.shape[-1] - input_length

    return {
        "output": decoded,
        "metrics": {
            "preprocess_time_sec":   round(preprocess_time, 4),
            "feature_time_sec":      round(feature_time, 4),
            "generation_time_sec":   round(generate_time, 4),
            "total_time_sec":        round(total_time, 4),
            "memory_used_gb":        round((peak_mem - base_mem)/(1024**3), 4),
            "peak_memory_gb":        round(peak_mem/(1024**3), 4),
            "sampled_tokens":        sample_seq_len,
            "input_length":          input_length,
            "tokens_generated":      tokens_generated,
            "tokens_per_sec":        round(tokens_generated / generate_time, 2)
        }
    }

    



def cosine_sampler(msg, ques, model, processor, sample_seq_len=40):
    """
    Samples text generation by selecting image tokens most similar to the last question token.

    Args:
        msg: previous chat messages/context object
        ques: question string
        model: multimodal model with vision_tower, multi_modal_projector, language_model, and generate()
        processor: tokenizer/processor with apply_chat_template() and decode()
        sample_seq_len: number of visual tokens to sample based on cosine similarity

    Returns:
        dict with "output" (decoded text) and "metrics" (timing & memory stats)
    """
    # Setup device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()

    # Clear caches & track base memory
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device)
    base_mem = torch.cuda.memory_allocated(device)

    with torch.no_grad():
        # --- Preprocessing ---
        start_pre = time.perf_counter()
        inputs = (
            processor
            .apply_chat_template(msg, add_generation_prompt=True, tokenize=True,
                                 return_dict=True, return_tensors="pt")
            .to(dtype=torch.bfloat16)
        )
        text_in = processor.tokenizer(
            f"<bos><start_of_turn>user\nAnswer in one word or phrase.\n\n"
            f"<start_of_image><end_of_image>{ques}<end_of_turn>\n",
            return_tensors='pt'
        )
        input_ids = text_in.input_ids
        # find insertion indices for visual tokens
        idx_1 = torch.where(input_ids == processor.tokenizer.convert_tokens_to_ids('<start_of_image>'))[1].item()
        idx_2 = torch.where(input_ids == processor.tokenizer.convert_tokens_to_ids('<end_of_image>'))[1].item()
        preprocess_time = time.perf_counter() - start_pre

        # --- Vision features ---
        start_feat = time.perf_counter()
        pix = inputs.pixel_values.to(device, non_blocking=True)
        vout = model.vision_tower(pix)
        img_feats = model.multi_modal_projector(vout.last_hidden_state)  # (1, N, C)
        del vout, pix, inputs
        gc.collect(); torch.cuda.empty_cache()

        # --- Text embeddings ---
        tid = input_ids.to(device, non_blocking=True)
        text_embeds = model.language_model.model.embed_tokens(tid)  # (1, T, C)
        # use only last token embedding from the question
        last_token_embed = text_embeds[0, -1, :].unsqueeze(0)  # (1, C)

        # --- Cosine similarity selection ---
        vision_tokens = img_feats[0]  # (N, C)
        # expand last_token_embed to match vision_tokens shape
        lt_expanded = last_token_embed.expand(vision_tokens.size(0), -1)
        sims = F.cosine_similarity(vision_tokens, lt_expanded, dim=1)  # (N,)
        topk_indices = torch.topk(sims, sample_seq_len).indices
        selected_tokens = vision_tokens[topk_indices].unsqueeze(0)  # (1, k, C)

        del img_feats, text_embeds, vision_tokens, sims
        gc.collect(); torch.cuda.empty_cache()
        feature_time = time.perf_counter() - start_feat

        # --- Combine embeddings ---
        embed = model.language_model.model.embed_tokens
        text_embeds = embed(tid)
        combined = torch.cat([
            text_embeds[:, :idx_1],
            selected_tokens,
            text_embeds[:, idx_1:idx_2],
            text_embeds[:, idx_2:]
        ], dim=1)
        input_length = combined.shape[-2]

        del selected_tokens, text_embeds, embed
        gc.collect(); torch.cuda.empty_cache()

        # --- Generate output ---
        start_gen = time.perf_counter()
        out = model.generate(
            inputs_embeds=combined,
            do_sample=False,
            max_new_tokens=20,
        )
        torch.cuda.synchronize()
        generate_time = time.perf_counter() - start_gen

    # --- Decode & metrics ---
    peak_mem = torch.cuda.max_memory_allocated(device)
    decoded = processor.decode(out[0].cpu(), skip_special_tokens=True)
    total_time = preprocess_time + feature_time + generate_time
    tokens_generated = out.shape[-1] - input_length

    return {
        "output": decoded,
        "metrics": {
            "preprocess_time_sec":   round(preprocess_time, 4),
            "feature_time_sec":      round(feature_time, 4),
            "generation_time_sec":   round(generate_time, 4),
            "total_time_sec":        round(total_time, 4),
            "memory_used_gb":        round((peak_mem - base_mem)/(1024**3), 4),
            "peak_memory_gb":        round(peak_mem/(1024**3), 4),
            "sampled_tokens":        sample_seq_len,
            "input_length":          input_length,
            "tokens_generated":      tokens_generated,
            "tokens_per_sec":        round(tokens_generated / generate_time, 2)
        }
    }


    
def kl_sampler(msg, ques, model, processor, sample_seq_len=40):
    """
    Samples text generation by selecting image tokens most similar to the last question token
    based on KL-divergence between their softmax distributions.

    Args:
        msg: previous chat messages/context object
        ques: question string
        model: multimodal model with vision_tower, multi_modal_projector, language_model, and generate()
        processor: tokenizer/processor with apply_chat_template() and decode()
        sample_seq_len: number of visual tokens to sample based on minimal KL divergence

    Returns:
        dict with "output" (decoded text) and "metrics" (timing & memory stats)
    """
    # Setup device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()

    # Clear caches & track base memory
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device)
    base_mem = torch.cuda.memory_allocated(device)

    with torch.no_grad():
        # --- Preprocessing ---
        start_pre = time.perf_counter()
        inputs = (
            processor
            .apply_chat_template(msg, add_generation_prompt=True, tokenize=True,
                                 return_dict=True, return_tensors="pt")
            .to(dtype=torch.bfloat16)
        )
        text_in = processor.tokenizer(
            f"<bos><start_of_turn>user\nAnswer in one word or phrase.\n\n"
            f"<start_of_image><end_of_image>{ques}<end_of_turn>\n",
            return_tensors='pt'
        )
        input_ids = text_in.input_ids
        idx_1 = torch.where(input_ids == processor.tokenizer.convert_tokens_to_ids('<start_of_image>'))[1].item()
        idx_2 = torch.where(input_ids == processor.tokenizer.convert_tokens_to_ids('<end_of_image>'))[1].item()
        preprocess_time = time.perf_counter() - start_pre

        # --- Vision features ---
        start_feat = time.perf_counter()
        pix = inputs.pixel_values.to(device, non_blocking=True)
        vout = model.vision_tower(pix)
        img_feats = model.multi_modal_projector(vout.last_hidden_state)  # (1, N, C)
        del vout, pix, inputs
        gc.collect(); torch.cuda.empty_cache()

        # --- Text embeddings ---
        tid = input_ids.to(device, non_blocking=True)
        text_embeds = model.language_model.model.embed_tokens(tid)  # (1, T, C)
        last_token_embed = text_embeds[0, -1, :].unsqueeze(0)  # (1, C)

        # --- KL divergence selection ---
        vision_tokens = img_feats[0]  # (N, C)
        # convert embeddings to distributions
        # add small value for numerical stability
        vt_log_probs = F.log_softmax(vision_tokens, dim=1)  # (N, C)
        lt_probs = F.softmax(last_token_embed, dim=1).expand_as(vision_tokens)  # (N, C)
        # compute KL divergence for each vision token: KL(lt || vt) or KL(vt || lt)? use vt_log_probs 
        # D_{KL}(lt||vt) = sum lt_probs * (log(lt_probs) - log(vt_probs))
        lt_log = torch.log(lt_probs + 1e-12)
        kl_div = (lt_probs * (lt_log - vt_log_probs)).sum(dim=1)  # (N,)
        # select tokens with smallest divergence
        topk_indices = torch.topk(-kl_div, sample_seq_len).indices
        selected_tokens = vision_tokens[topk_indices].unsqueeze(0)  # (1, k, C)

        del img_feats, text_embeds, vision_tokens, vt_log_probs, lt_probs, lt_log, kl_div
        gc.collect(); torch.cuda.empty_cache()
        feature_time = time.perf_counter() - start_feat

        # --- Combine embeddings ---
        embed = model.language_model.model.embed_tokens
        text_embeds = embed(tid)
        combined = torch.cat([
            text_embeds[:, :idx_1],
            selected_tokens,
            text_embeds[:, idx_1:idx_2],
            text_embeds[:, idx_2:]
        ], dim=1)
        input_length = combined.shape[-2]

        del selected_tokens, text_embeds, embed
        gc.collect(); torch.cuda.empty_cache()

        # --- Generate output ---
        start_gen = time.perf_counter()
        out = model.generate(
            inputs_embeds=combined,
            do_sample=False,
            max_new_tokens=20,
        )
        torch.cuda.synchronize()
        generate_time = time.perf_counter() - start_gen

    # --- Decode & metrics ---
    peak_mem = torch.cuda.max_memory_allocated(device)
    decoded = processor.decode(out[0].cpu(), skip_special_tokens=True)
    total_time = preprocess_time + feature_time + generate_time
    tokens_generated = out.shape[-1] - input_length

    return {
        "output": decoded,
        "metrics": {
            "preprocess_time_sec":   round(preprocess_time, 4),
            "feature_time_sec":      round(feature_time, 4),
            "generation_time_sec":   round(generate_time, 4),
            "total_time_sec":        round(total_time, 4),
            "memory_used_gb":        round((peak_mem - base_mem)/(1024**3), 4),
            "peak_memory_gb":        round(peak_mem/(1024**3), 4),
            "sampled_tokens":        sample_seq_len,
            "input_length":          input_length,
            "tokens_generated":      tokens_generated,
            "tokens_per_sec":        round(tokens_generated / generate_time, 2)
        }
    }

    

def merge_token(inputs):
    pass


dataset = load_textvqa_val(sample_num=5)





sampled_token = 100

def run_exp(dataset,opt=1):
    #load model
    processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")
    model = AutoModelForImageTextToText.from_pretrained(
        "google/gemma-3-4b-it",
        device_map='cuda'
    ).eval()

    results = []
    for sample in tqdm(dataset):        
        msg, qid,ans,ques = sample        
        if opt==1:
            pred = bench(msg)

            result_entry = {
                "qid": qid,
                "input": msg,
                "reference_answer": ans,
                "model_output": pred["output"],
                "metrics": pred["metrics"]
                }
        
            results.append(result_entry)
        if opt==2:
            pred = random_drop(msg,ques=ques,model=model,processor=processor,sample_seq_len=sampled_token)

            result_entry = {
                "qid": qid,
                "input": msg,
                "reference_answer": ans,
                "model_output": pred["output"],
                "metrics": pred["metrics"]
                }
        
            results.append(result_entry)
        if opt==3:
            pred =  k_means_merge(msg,ques=ques,model=model,processor=processor,sample_seq_len=sampled_token)

            result_entry = {
                "qid": qid,
                "input": msg,
                "reference_answer": ans,
                "model_output": pred["output"],
                "metrics": pred["metrics"]
                }
        
            results.append(result_entry)
        if opt==4:
            pred =  cosine_sampler(msg,ques=ques,model=model,processor=processor,sample_seq_len=sampled_token)

            result_entry = {
                "qid": qid,
                "input": msg,
                "reference_answer": ans,
                "model_output": pred["output"],
                "metrics": pred["metrics"]
                }
        
            results.append(result_entry)
        if opt==5:
            pred =  kl_sampler(msg,ques=ques,model=model,processor=processor,sample_seq_len=sampled_token)

            result_entry = {
                "qid": qid,
                "input": msg,
                "reference_answer": ans,
                "model_output": pred["output"],
                "metrics": pred["metrics"]
                }
        
            results.append(result_entry)
    
    if opt==1:
        with open('eval/'+'benchmark.json', "w") as f:
            json.dump(results, f, indent=2)
    if opt==2:
        with open('eval/'+'random_drop_'+str(sampled_token)+'.json', "w") as f:
            json.dump(results, f, indent=2)
    if opt==3:
        with open('eval/'+'k_means_'+str(sampled_token)+'.json', "w") as f:
            json.dump(results, f, indent=2)
    if opt==4:
        with open('eval/'+'cos_'+str(sampled_token)+'.json', "w") as f:
            json.dump(results, f, indent=2)
    if opt==5:
        with open('eval/'+'kl_'+str(sampled_token)+'.json', "w") as f:
            json.dump(results, f, indent=2)

run_exp(dataset,opt=5)


    



