from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F


def logit_lens(recettes, model, tokenizer, preserved_layers=[0, 1, -1], model_name='default'):
    """
    Take input sentences, encode them to get hidden_state layers and their unencoded version from logit lens approach to keep track of vocabulary divergence
    preserved_layers : Range you want to keep in the hideen states -- most interesting are direct embedding -- value =0 and semantic representation value = 1
    """
    #LANGUAGE_MODE = 0  # InputMode.LANGUAGE
    top_k = 10
    all_layers = []
    layer_sent = []
    #Memory optimization 
    model.eval()
    use_half = (model.dtype in (torch.float16, torch.bfloat16))

    for recette in recettes:
        inputs = tokenizer(recette, return_tensors="pt").to(model.device)
        #with torch.no_grad():
        with torch.inference_mode():  # lighter than no_grad
            # autocast only if model supports it
            #cm = (torch.cuda.amp.autocast(dtype=model.dtype) if use_half else torch.cuda.amp.autocast(enabled=False))
            if use_half:
                cm = torch.amp.autocast("cuda", dtype=model.dtype)
            else:
                cm = torch.amp.autocast("cuda", enabled=False)

            with cm:
                if model_name == 'microsoft':
                    LANGUAGE_MODE_TENSOR = torch.tensor([0], device=inputs["input_ids"].device, dtype=torch.long)
                    inputs.pop("input_mode", None)           # remove any existing bad value
                    inputs["input_mode"] = LANGUAGE_MODE_TENSOR
                    outputs = model(
                        **inputs,
                        output_hidden_states=True,
                        output_attentions=False,
                        use_cache=False)
                        # tell Phi-4 MM we're doing pure text:
                        #input_mode=LANGUAGE_MODE)
                else:
                    outputs = model(
                        **inputs,
                        output_hidden_states=True,
                        output_attentions=False,
                        use_cache=False  # avoid extra KV cache memory
                    )

            outputs = model(**inputs, output_hidden_states=True, output_attentions=False)
            layers = outputs.hidden_states   #[full_n_layers, seq_len, hidden_dim]
            #selected_layers =  [n_layers, seq_len, hidden_dim]
            selected_layers = [layers[i][0] for i in preserved_layers if -len(layers) <= i < len(layers)]   
            # Stack into one tensor instead of list -- keep track of layers if needed
            stacked = torch.stack(selected_layers, dim=0)

            ### Applying logit lens here # selected_logits =  [n_layers, seq_len, vocab_size]
            if model_name == 'gemma-3':
                ### A essayer for gemma-3 : model.language_model.model.norm(...)
                selected_probs = [F.softmax(model.lm_head(model.model.language_model.norm(layers[i][0])), dim=-1) for i in preserved_layers if -len(layers) <= i < len(layers)]
            else:
                selected_probs = [F.softmax(model.lm_head(model.model.norm(layers[i][0])), dim=-1) for i in preserved_layers if -len(layers) <= i < len(layers)]
            ########################################
            selected_topk = [torch.topk(probs, k=top_k, dim=-1) for probs in selected_probs]
            token_probs = [topk.values for topk in selected_topk]
            token_ids = [topk.indices for topk in selected_topk]

            # Choose best-matching token in intersection with highest prob (per position)
            selected_token_ids = []
            for j in range(len(token_ids)):   # For each layer first
                token_id_layer = token_ids[j]
                token_probs_layer = token_probs[j]
                layer_selected_ids = []
                for i in range(token_id_layer.size(0)):  # for each token position
                    candidates = token_id_layer[i]        # shape: (top_k,)
                    probs = token_probs_layer[i]     # shape: (top_k,)
                    #Decode candidates to string tokens
                    decoded_tokens = tokenizer.convert_ids_to_tokens(candidates.tolist())    # shape: (top_k,)
                    #Match tokens in original string and select most probable one. If no match then select the most probable
                    matched = [(tok_id.item(), prob.item()) for tok_id, tok_str, prob in zip(candidates, decoded_tokens, probs) if tok_str in recette]
                    if matched:
                        best_token = max(matched, key=lambda x: x[1])[0]
                    else:
                        best_token = candidates[0].item()
                    layer_selected_ids.append(best_token)

                selected_token_ids.append(layer_selected_ids)   ## [n_alyers, seq_len]

            decoded_sentences = [tokenizer.decode(token_ids, skip_special_tokens=True) for token_ids in selected_token_ids]

            del outputs, layers, inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        #List of for all recipe [[n_layers, seq_len, vocab_size]] of len = n_recipe
        all_layers.append(stacked)
        layer_sent.append(decoded_sentences) #[[n_layers, seq_len]]  # List of decoded sentence for all recipes



    return all_layers, layer_sent

def add_falcon_compat_aliases(model):
    # Expose `.model` like some libs expect
    if hasattr(model, "transformer") and not hasattr(model, "model"):
        model.model = model.transformer

    tr = model.model if hasattr(model, "model") else getattr(model, "transformer", None)
    if tr is None:
        return model  # not a Falcon-like model

    # LLaMA-style aliases expected by many utilities
    if not hasattr(tr, "norm") and hasattr(tr, "ln_f"):
        tr.norm = tr.ln_f
    if not hasattr(tr, "embed_tokens") and hasattr(tr, "word_embeddings"):
        tr.embed_tokens = tr.word_embeddings
    if not hasattr(tr, "layers") and hasattr(tr, "h"):
        tr.layers = tr.h

    return model