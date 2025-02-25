import torch

def generate_kv_with_id(model, input_ids):
    input_ids = input_ids.to(model.device)

    with torch.no_grad():
        out = model(input_ids)
        past_key_values = out.past_key_values

    return past_key_values

def generate_kv_with_connect(model, input_ids):

    input_ids = input_ids.to(model.device)

    with torch.no_grad():
        out = model(input_ids)
        past_key_values = out.past_key_values

    return past_key_values

def generate_kv_with_position(model, input_ids, position_ids):

    input_ids = input_ids.to(model.device)
    position_ids = position_ids.to(model.device)

    with torch.no_grad():
        out = model(input_ids = input_ids, position_ids = position_ids)
        past_key_values = out.past_key_values

    return past_key_values

def append_kv(kv_list, d):  #d=0 batch size; d=2 sequence length
    num_layers = len(kv_list[0])
    concatenated_past_key_values = ()

    for layer in range(num_layers):
        keys_list = [kv[layer][0].detach() for kv in kv_list]
        values_list = [kv[layer][1].detach() for kv in kv_list]

        concatenated_keys = torch.cat(keys_list, dim=d)
        concatenated_values = torch.cat(values_list, dim=d)
        concatenated_past_key_values += ((concatenated_keys, concatenated_values),)

    return concatenated_past_key_values

def concat_kv(split_kv, num_memory):  
    '''
    This function convert a batched splited memory KV cache into a batched concatenated memory KV cache
    split_kv: ((batch_size * num_memory, num_heads, seq_len, hidden_dims),(...))
    
    final_past_key_values: ((batch_size, num_heads, seq_len * num_memory, hidden_dims),(...))
    '''
    num_layers = len(split_kv)
    split_batch_size = split_kv[0][0].size(0)
    final_past_key_values = ()

    for layer_idx in range(num_layers):
        key_cache, value_cache = split_kv[layer_idx]

        concatenated_keys_list = []
        concatenated_values_list = []

        for i in range(0, split_batch_size, num_memory):
            
            key_group = key_cache[i:i+num_memory]
            key_list = torch.split(key_group, 1, dim=0)
            value_group = value_cache[i:i+num_memory]
            value_list = torch.split(value_group, 1, dim=0)

            concatenated_key = torch.cat(key_list, dim=2)
            concatenated_value = torch.cat(value_list, dim=2)

            concatenated_keys_list.append(concatenated_key)
            concatenated_values_list.append(concatenated_value)

        layer_concatenated_keys = torch.cat(concatenated_keys_list, dim=0)  # Concatenate along batch dimension
        layer_concatenated_values = torch.cat(concatenated_values_list, dim=0)

        final_past_key_values += ((layer_concatenated_keys, layer_concatenated_values),)
        
    return final_past_key_values