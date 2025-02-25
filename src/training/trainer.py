import torch
import matplotlib.pyplot as plt
import json
import random
import time
from transformers import Trainer
from torch.nn import CrossEntropyLoss
from src.utils.cache import generate_kv_with_id, concat_kv, append_kv, generate_kv_with_position
from src.data.attention import construct_biased_attention_matrix
# from transformers import LlamaModel
class CustomTrainer(Trainer):
    def __init__(self, *args, data_loader, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_loader = data_loader
        self.train_loss_history = []

    def get_train_dataloader(self):
        return self.data_loader

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        memory_ids = input_ids[:, 1:501]
        remaining_ids_batch = input_ids[:, 501:]
        # kv_list = [generate_kv_with_id(self.model, self.tokenizer("", return_tensors="pt").input_ids)]
        split_input_ids = memory_ids.reshape(-1, 50)

        split_past_key_values = generate_kv_with_id(self.model, split_input_ids)
        kv_s = generate_kv_with_id(self.model, self.tokenizer("", return_tensors="pt").input_ids)

        num_memory = int((501 - 1) / 50)
        past_key_values_batch = concat_kv(split_past_key_values, num_memory)
        kv_s_concat = append_kv([kv_s] * past_key_values_batch[0][0].size(0),0)
        past_key_values_batch = append_kv([kv_s_concat, past_key_values_batch],2)

        # print(remaining_ids_batch.shape, attention_mask.shape, len(past_key_values_batch), past_key_values_batch[0][0].shape)
        outputs = self.model(input_ids=remaining_ids_batch, attention_mask=attention_mask, labels = remaining_ids_batch, past_key_values=past_key_values_batch, use_cache=True)

        logits = outputs.logits  

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = remaining_ids_batch[..., 1:].contiguous()

        loss_fct = CrossEntropyLoss(reduction='none')
        losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        losses = losses.view(shift_logits.size(0), -1)
        
        mask = attention_mask[:, 1:].clone()
        mask = mask[:, 501:]
        masked_losses = losses * mask

        masked_losses_sum = masked_losses.sum()
        valid_positions = mask.sum()

        batch_loss = masked_losses_sum / valid_positions
        print(batch_loss)

        return batch_loss

    def save_training_curve(self, output_dir):
        # Save the loss history to a JSON file
        with open(f"{output_dir}/train_loss_history.json", "w") as f:
            json.dump(self.train_loss_history, f)

        # Plot and save the training curve as an image
        plt.plot(self.train_loss_history, label='Train Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.savefig(f"{output_dir}/train_loss_curve.png")
        plt.show()

class CustomTrainerConnect(Trainer):
    def __init__(self, *args, data_loader, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_loader = data_loader
        self.train_loss_history = []

    def get_train_dataloader(self):
        return self.data_loader

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        memory_ids = input_ids[:, 1:501]
        remaining_ids_batch = input_ids[:, 501:]

        memory_attention = attention_mask[:, 1:501]
        split_memory_attention = memory_attention.reshape(-1, 50)
        split_memory_attention = torch.cat((split_memory_attention, torch.zeros(split_memory_attention.shape[0], 1).to(split_memory_attention.device)), dim=1)
        memory_attention = split_memory_attention.reshape(attention_mask.shape[0], 510)

        # print(attention_mask[:, 0].shape, memory_attention.shape, attention_mask[:, 501].shape)
        attention_mask = torch.cat((attention_mask[:, :1], memory_attention, attention_mask[:, 501:]), dim=1)

        # kv_list = [generate_kv_with_id(self.model, self.tokenizer("", return_tensors="pt").input_ids)]
        split_input_ids = memory_ids.reshape(-1, 50)

        split_past_key_values = generate_kv_with_id(self.model.base_model, split_input_ids)

        # print(remaining_ids_batch.shape, attention_mask.shape, len(past_key_values_batch), past_key_values_batch[0][0].shape)
        outputs = self.model(input_ids=remaining_ids_batch, attention_mask=attention_mask, labels = remaining_ids_batch, past_key_values=split_past_key_values, use_cache=True)

        logits = outputs.logits  

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = remaining_ids_batch[..., 1:].contiguous()

        loss_fct = CrossEntropyLoss(reduction='none')
        losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        losses = losses.view(shift_logits.size(0), -1)
        
        mask = attention_mask[:, 1:].clone()
        mask = mask[:, 511:]
        masked_losses = losses * mask

        masked_losses_sum = masked_losses.sum()
        valid_positions = mask.sum()

        batch_loss = masked_losses_sum / valid_positions
        # print(batch_loss)

        return batch_loss

class CustomTrainerNormal(Trainer):
    def __init__(self, *args, data_loader, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_loader = data_loader
        self.train_loss_history = []

    def get_train_dataloader(self):
        return self.data_loader

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # print(remaining_ids_batch.shape, attention_mask.shape, len(past_key_values_batch), past_key_values_batch[0][0].shape)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels = input_ids)

        logits = outputs.logits  

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        loss_fct = CrossEntropyLoss(reduction='none')
        losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        losses = losses.view(shift_logits.size(0), -1)
        losses = losses[:, 501:]

        mask = attention_mask[:, 1:].clone()
        mask = mask[:, 501:]
        masked_losses = losses * mask

        masked_losses_sum = masked_losses.sum()
        valid_positions = mask.sum()

        batch_loss = masked_losses_sum / valid_positions
        print(batch_loss)

        return batch_loss

    def save_training_curve(self, output_dir):
        # Save the loss history to a JSON file
        with open(f"{output_dir}/train_loss_history.json", "w") as f:
            json.dump(self.train_loss_history, f)

        # Plot and save the training curve as an image
        plt.plot(self.train_loss_history, label='Train Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.savefig(f"{output_dir}/train_loss_curve.png")
        plt.show()

    def save_training_curve(self, output_dir):
        # Save the loss history to a JSON file
        with open(f"{output_dir}/train_loss_history.json", "w") as f:
            json.dump(self.train_loss_history, f)

        # Plot and save the training curve as an image
        plt.plot(self.train_loss_history, label='Train Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.savefig(f"{output_dir}/train_loss_curve.png")
        plt.show()

class CustomTrainerCheat(Trainer):
    def __init__(self, *args, data_loader, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_loader = data_loader
        self.train_loss_history = []

    def get_train_dataloader(self):
        return self.data_loader

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        num_memory = random.randint(5, 10)
        each_mem_len = random.randint(30, 80)
        mum_len = num_memory * each_mem_len

        memory_ids = input_ids[:, 1:mum_len + 1]
        remaining_ids_batch = input_ids[:, mum_len + 1:]
        # kv_list = [generate_kv_with_id(self.model, self.tokenizer("", return_tensors="pt").input_ids)]
        split_input_ids = memory_ids.reshape(-1, each_mem_len)

        memory_position = torch.arange(1, 1 + mum_len).unsqueeze(0)
        memory_positions = torch.cat([memory_position] * input_ids.size(0))
        memory_position_batch = memory_positions.reshape(-1, each_mem_len)

        split_past_key_values = generate_kv_with_position(self.model, split_input_ids, position_ids = memory_position_batch)
        kv_s = generate_kv_with_position(self.model, self.tokenizer("", return_tensors="pt").input_ids, position_ids = torch.tensor([[0]]))

        past_key_values_batch = concat_kv(split_past_key_values, num_memory)
        kv_s_concat = append_kv([kv_s] * past_key_values_batch[0][0].size(0),0)
        past_key_values_batch = append_kv([kv_s_concat, past_key_values_batch],2)

        # print(remaining_ids_batch.shape, attention_mask.shape, len(past_key_values_batch), past_key_values_batch[0][0].shape)
        outputs = self.model(input_ids=remaining_ids_batch, attention_mask=attention_mask, labels = remaining_ids_batch, past_key_values=past_key_values_batch, use_cache=True)

        logits = outputs.logits  

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = remaining_ids_batch[..., 1:].contiguous()

        loss_fct = CrossEntropyLoss(reduction='none')
        losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        losses = losses.view(shift_logits.size(0), -1)
        
        mask = attention_mask[:, 1:].clone()
        mask = mask[:, mum_len + 1:]
        masked_losses = losses * mask

        masked_losses_sum = masked_losses.sum()
        valid_positions = mask.sum()

        batch_loss = masked_losses_sum / valid_positions
        print(batch_loss)

        return batch_loss

    def save_training_curve(self, output_dir):
        # Save the loss history to a JSON file
        with open(f"{output_dir}/train_loss_history.json", "w") as f:
            json.dump(self.train_loss_history, f)

        # Plot and save the training curve as an image
        plt.plot(self.train_loss_history, label='Train Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.savefig(f"{output_dir}/train_loss_curve.png")
        plt.show()

class CustomTrainerCombine(Trainer):
    def __init__(self, *args, data_loader, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_loader = data_loader
        self.train_loss_history = []

    def get_train_dataloader(self):
        return self.data_loader

    def compute_loss(self, model, inputs, return_outputs=False):
        dataset_id = inputs["dataset_id"]
        # print(dataset_id)

        if dataset_id[0] == 'text':
            input_ids = inputs["input_ids"][0].unsqueeze(0)
            attention_mask = inputs["attention_mask"][0].unsqueeze(0)

            num_memory = random.randint(5, 10)
            each_mem_len = random.randint(30, 80)
            mum_len = num_memory * each_mem_len

            memory_ids = input_ids[:, 1:mum_len + 1]
            remaining_ids_batch = input_ids[:, mum_len + 1:]
            # kv_list = [generate_kv_with_id(self.model, self.tokenizer("", return_tensors="pt").input_ids)]
            split_input_ids = memory_ids.reshape(-1, each_mem_len)

            memory_position = torch.arange(1, 1 + mum_len).unsqueeze(0)
            memory_positions = torch.cat([memory_position] * input_ids.size(0))
            memory_position_batch = memory_positions.reshape(-1, each_mem_len)

            split_past_key_values = generate_kv_with_position(self.model, split_input_ids, position_ids = memory_position_batch)
            kv_s = generate_kv_with_position(self.model, self.tokenizer("", return_tensors="pt").input_ids, position_ids = torch.tensor([[0]]))

            past_key_values_batch = concat_kv(split_past_key_values, num_memory)
            kv_s_concat = append_kv([kv_s] * past_key_values_batch[0][0].size(0),0)
            past_key_values_batch = append_kv([kv_s_concat, past_key_values_batch],2)

            # print(remaining_ids_batch.shape, attention_mask.shape, len(past_key_values_batch), past_key_values_batch[0][0].shape)
            outputs = self.model(input_ids=remaining_ids_batch, attention_mask=attention_mask, labels = remaining_ids_batch, past_key_values=past_key_values_batch, use_cache=True)

            logits = outputs.logits  

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = remaining_ids_batch[..., 1:].contiguous()

            loss_fct = CrossEntropyLoss(reduction='none')
            losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            losses = losses.view(shift_logits.size(0), -1)
            
            mask = attention_mask[:, 1:].clone()
            mask = mask[:, mum_len + 1:]
            masked_losses = losses * mask

            masked_losses_sum = masked_losses.sum()
            valid_positions = mask.sum()

            batch_loss1 = masked_losses_sum / valid_positions
            # print("loss1: ",batch_loss1)
        
        if dataset_id[1] == 'sft':
            input_ids = inputs["input_ids"][1].unsqueeze(0)
            attention_mask = inputs["attention_mask"][1].unsqueeze(0)
            mask = inputs["loss_mask"][1].unsqueeze(0)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels = input_ids)

            logits = outputs.logits  

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            loss_fct = CrossEntropyLoss(reduction='none')
            losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            losses = losses.view(shift_logits.size(0), -1)

            mask = mask[:, 1:]
            masked_losses = losses * mask
            # print(masked_losses)
            masked_losses_sum = masked_losses.sum()
            valid_positions = mask.sum()

            batch_loss2 = masked_losses_sum / valid_positions
            # print("loss2: ",batch_loss2)
        
        return (batch_loss1 + batch_loss2) / 2

class CustomTrainerSpecial(Trainer):
    def __init__(self, *args, data_loader, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_loader = data_loader
        self.train_loss_history = []

    def get_train_dataloader(self):
        return self.data_loader

    def compute_loss(self, model, inputs, return_outputs=False):
        dataset_id = inputs["dataset_id"]
        # print(dataset_id)

        if dataset_id[0] == 'text':
            input_ids = inputs["input_ids"][0].unsqueeze(0)
            attention_mask = inputs["attention_mask"][0].unsqueeze(0)

            num_memory = random.randint(5, 10)
            each_mem_len = random.randint(30, 80)
            mum_len = num_memory * each_mem_len

            input_len = input_ids.size(1)
            input_ids = input_ids[:, :input_len - num_memory]

            attention_mask_len = attention_mask.size(1)
            attention_mask = attention_mask[:, :attention_mask_len - num_memory]
            attention_mask = torch.cat([torch.tensor([[1] * num_memory]).to(attention_mask.device), attention_mask], dim=1)

            memory_ids = input_ids[:, 1:mum_len + 1]
            remaining_ids_batch = input_ids[:, mum_len + 1:]

            split_input_ids = memory_ids.reshape(-1, each_mem_len)
            split_input_ids = torch.cat([split_input_ids, torch.tensor([[32000]]*split_input_ids.size(0)).to(split_input_ids.device)], dim=1)

            memory_position = torch.arange(1, 1 + mum_len + num_memory).unsqueeze(0)
            memory_positions = torch.cat([memory_position] * input_ids.size(0))
            memory_position_batch = memory_positions.reshape(-1, each_mem_len + 1)

            split_past_key_values = generate_kv_with_position(self.model, split_input_ids, position_ids = memory_position_batch)
            kv_s = generate_kv_with_position(self.model, self.tokenizer("", return_tensors="pt").input_ids, position_ids = torch.tensor([[0]]))

            past_key_values_batch = concat_kv(split_past_key_values, num_memory)
            kv_s_concat = append_kv([kv_s] * past_key_values_batch[0][0].size(0),0)
            past_key_values_batch = append_kv([kv_s_concat, past_key_values_batch],2)

            # print(remaining_ids_batch.shape, attention_mask.shape,  past_key_values_batch[0][0].shape)
            outputs = self.model(input_ids=remaining_ids_batch, attention_mask=attention_mask, labels = remaining_ids_batch, past_key_values=past_key_values_batch, use_cache=True)

            logits = outputs.logits  

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = remaining_ids_batch[..., 1:].contiguous()

            loss_fct = CrossEntropyLoss(reduction='none')
            losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            losses = losses.view(shift_logits.size(0), -1)
            
            mask = attention_mask[:, 1:].clone()
            mask = mask[:, mum_len  + num_memory + 1:]
            masked_losses = losses * mask

            masked_losses_sum = masked_losses.sum()
            valid_positions = mask.sum()

            batch_loss1 = masked_losses_sum / valid_positions
            # print("loss1: ",batch_loss1)
        
        if dataset_id[1] == 'sft':
            input_ids = inputs["input_ids"][1].unsqueeze(0)
            attention_mask = inputs["attention_mask"][1].unsqueeze(0)
            mask = inputs["loss_mask"][1].unsqueeze(0)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels = input_ids)

            logits = outputs.logits  

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            loss_fct = CrossEntropyLoss(reduction='none')
            losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            losses = losses.view(shift_logits.size(0), -1)

            mask = mask[:, 1:]
            masked_losses = losses * mask
            # print(masked_losses)
            masked_losses_sum = masked_losses.sum()
            valid_positions = mask.sum()

            batch_loss2 = masked_losses_sum / valid_positions
            # print("loss2: ",batch_loss2)
        
        return (batch_loss1 + batch_loss2) / 2

class CustomTrainerMixSpecial(Trainer):
    def __init__(self, *args, data_loader, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_loader = data_loader
        self.train_loss_history = []

    def get_train_dataloader(self):
        return self.data_loader

    def textinst_loss(self, input_ids, labels, split_input_ids, memory_positions, sys_tokens):
        num_memory = memory_positions.size(0)

        split_past_key_values = generate_kv_with_position(self.model, split_input_ids, position_ids = memory_positions)
        memory_key_values = concat_kv(split_past_key_values, num_memory)


        sys_key_values = generate_kv_with_id(self.model, sys_tokens)
        past_key_values = append_kv([sys_key_values, memory_key_values], 2)

        outputs = self.model(input_ids=input_ids, labels = labels, past_key_values = past_key_values, use_cache = True)
        return outputs.loss
    
    def sftmem_loss(self, input_ids, labels, memory_ids, memory_positions, sys_tokens):
        num_memory = len(memory_positions)
        sys_key_values = generate_kv_with_id(self.model, sys_tokens)
        kv_list = [sys_key_values]

        for idx in range(num_memory):
            kv_list.append(generate_kv_with_position(self.model, torch.tensor([memory_ids[idx]]), position_ids = torch.tensor([memory_positions[idx]])))
        past_key_values = append_kv(kv_list, 2)

        outputs = self.model(input_ids=input_ids, labels = labels, past_key_values = past_key_values, use_cache = True)
        return outputs.loss
    
    def text_loss(self, input_ids, labels):
        outputs = self.model(input_ids=input_ids, labels = labels)
        return outputs.loss
    
    def textmem_loss(self, input_ids, labels, memory_ids, memory_positions, sys_tokens):
        num_memory = memory_positions.size(0)

        split_past_key_values = generate_kv_with_position(self.model, memory_ids, position_ids = memory_positions)
        # print(memory_ids.size(), memory_positions.size())
        memory_key_values = concat_kv(split_past_key_values, num_memory)

        sys_key_values = generate_kv_with_id(self.model, sys_tokens)
        past_key_values = append_kv([sys_key_values, memory_key_values], 2)

        outputs = self.model(input_ids=input_ids, labels = labels, past_key_values = past_key_values, use_cache = True)
        return outputs.loss

    def sft_loss(self, input_ids, labels):
        outputs = self.model(input_ids=input_ids, labels = labels)
        return outputs.loss

    def nqmem_loss(self, input_ids, labels, memory_ids, memory_positions, sys_tokens):
        num_memory = len(memory_positions)
        sys_key_values = generate_kv_with_id(self.model, sys_tokens)
        kv_list = [sys_key_values]

        for idx in range(num_memory):
            kv_list.append(generate_kv_with_position(self.model, torch.tensor([memory_ids[idx]]), position_ids = torch.tensor([memory_positions[idx]])))
        past_key_values = append_kv(kv_list, 2)
        # print("kv", past_key_values[0][0].size())
        outputs = self.model(input_ids=input_ids, labels = labels, past_key_values = past_key_values, use_cache = True)
        return outputs.loss

    def xsum_loss(self, input_ids, labels, memory_ids, memory_positions, sys_tokens):
        num_memory = len(memory_positions)
        sys_key_values = generate_kv_with_id(self.model, sys_tokens)
        kv_list = [sys_key_values]

        for idx in range(num_memory):
            kv_list.append(generate_kv_with_position(self.model, torch.tensor([memory_ids[idx]]), position_ids = torch.tensor([memory_positions[idx]])))
        past_key_values = append_kv(kv_list, 2)

        outputs = self.model(input_ids=input_ids, labels = labels, past_key_values = past_key_values, use_cache = True)
        return outputs.loss
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # loss_list = []
        final_loss = torch.tensor(0)
        for i in range(len(inputs["dataset_id"])):
            if inputs["dataset_id"][i] == 'textinst':
                loss = self.textinst_loss(inputs["input_ids"][i].unsqueeze(0), inputs["labels"][i].unsqueeze(0), torch.tensor(inputs["split_memory_id"][i]), torch.tensor(inputs["memory_position"][i]), inputs["sys_id"][i])
                final_loss = final_loss.to(loss.device) + loss
            elif inputs["dataset_id"][i] == 'sftmem':
                loss = self.sftmem_loss(inputs["input_ids"][i].unsqueeze(0), inputs["labels"][i].unsqueeze(0), inputs["split_memory_id"][i], inputs["memory_position"][i], inputs["sys_id"][i])
                final_loss = final_loss.to(loss.device) + loss
            elif inputs["dataset_id"][i] == 'text':
                loss = self.text_loss(inputs["input_ids"][i].unsqueeze(0), inputs["labels"][i].unsqueeze(0))
                final_loss = final_loss.to(loss.device) + loss
            elif inputs["dataset_id"][i] == 'textmem':
                loss = self.textmem_loss(inputs["input_ids"][i].unsqueeze(0), inputs["labels"][i].unsqueeze(0), torch.tensor(inputs["split_memory_id"][i]), torch.tensor(inputs["memory_position"][i]), inputs["sys_id"][i])
                final_loss = final_loss.to(loss.device) + loss
            elif inputs["dataset_id"][i] == 'sft':
                loss = self.sft_loss(inputs["input_ids"][i].unsqueeze(0), inputs["labels"][i].unsqueeze(0))
                final_loss = final_loss.to(loss.device) + loss
            elif inputs["dataset_id"][i] == 'nqmem':
                loss = self.nqmem_loss(inputs["input_ids"][i].unsqueeze(0), inputs["labels"][i].unsqueeze(0), inputs["split_memory_id"][i], inputs["memory_position"][i], inputs["sys_id"][i])
                final_loss = final_loss.to(loss.device) + loss
            elif inputs["dataset_id"][i] == 'xsum':
                loss = self.xsum_loss(inputs["input_ids"][i].unsqueeze(0), inputs["labels"][i].unsqueeze(0), inputs["split_memory_id"][i], inputs["memory_position"][i], inputs["sys_id"][i])
                final_loss = final_loss.to(loss.device) + loss
        return final_loss / len(inputs["dataset_id"])
    
# class CustomTrainerMixBaseline(Trainer):
#     def __init__(self, *args, data_loader, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.data_loader = data_loader
#         self.train_loss_history = []

#     def get_train_dataloader(self):
#         return self.data_loader

#     def textinst_loss(self, input_ids, labels):
#         outputs = self.model(input_ids=input_ids, labels = labels)
#         return outputs.loss
    
#     def sftmem_loss(self, input_ids, labels):
#         outputs = self.model(input_ids=input_ids, labels = labels)
#         return outputs.loss
    
#     def text_loss(self, input_ids, labels):
#         outputs = self.model(input_ids=input_ids, labels = labels)
#         return outputs.loss
    
#     def textmem_loss(self, input_ids, labels):
#         outputs = self.model(input_ids=input_ids, labels = labels)
#         return outputs.loss

#     def sft_loss(self, input_ids, labels):
#         outputs = self.model(input_ids=input_ids, labels = labels)
#         return outputs.loss

    # def nqmem_loss(self, input_ids, labels, memory_ids, memory_positions, sys_tokens):
    #     num_memory = len(memory_positions)
    #     sys_key_values = generate_kv_with_id(self.model, sys_tokens)
    #     kv_list = [sys_key_values]

    #     for idx in range(num_memory):
    #         kv_list.append(generate_kv_with_position(self.model, torch.tensor([memory_ids[idx]]), position_ids = torch.tensor([memory_positions[idx]])))
    #     past_key_values = append_kv(kv_list, 2)
    #     # print("kv", past_key_values[0][0].size())
    #     outputs = self.model(input_ids=input_ids, labels = labels, past_key_values = past_key_values, use_cache = True)
    #     return outputs.loss

    # def compute_loss(self, model, inputs, return_outputs=False):
    #     # loss_list = []
    #     final_loss = torch.tensor(0)
    #     for i in range(len(inputs["dataset_id"])):
    #         if inputs["dataset_id"][i] == 'textinst':
    #             loss = self.textinst_loss(inputs["input_ids"][i].unsqueeze(0), inputs["labels"][i].unsqueeze(0))
    #             final_loss = final_loss.to(loss.device) + loss
    #         elif inputs["dataset_id"][i] == 'sftmem':
    #             loss = self.sftmem_loss(inputs["input_ids"][i].unsqueeze(0), inputs["labels"][i].unsqueeze(0))
    #             final_loss = final_loss.to(loss.device) + loss
    #         elif inputs["dataset_id"][i] == 'text':
    #             loss = self.text_loss(inputs["input_ids"][i].unsqueeze(0), inputs["labels"][i].unsqueeze(0))
    #             final_loss = final_loss.to(loss.device) + loss
    #         elif inputs["dataset_id"][i] == 'textmem':
    #             loss = self.textmem_loss(inputs["input_ids"][i].unsqueeze(0), inputs["labels"][i].unsqueeze(0))
    #             final_loss = final_loss.to(loss.device) + loss
    #         elif inputs["dataset_id"][i] == 'sft':
    #             loss = self.sft_loss(inputs["input_ids"][i].unsqueeze(0), inputs["labels"][i].unsqueeze(0))
    #             final_loss = final_loss.to(loss.device) + loss
    #         # elif inputs["dataset_id"][i] == 'nqmem':
    #         #     loss = self.nqmem_loss(inputs["input_ids"][i].unsqueeze(0), inputs["labels"][i].unsqueeze(0), inputs["split_memory_id"][i], inputs["memory_position"][i], inputs["sys_id"][i])
    #         #     final_loss = final_loss.to(loss.device) + loss
    #     return final_loss / len(inputs["dataset_id"])
    
class CustomTrainerMixSpecial_Batch(Trainer):
    def __init__(self, *args, data_loader, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_loader = data_loader
        self.train_loss_history = []

    def get_train_dataloader(self):
        return self.data_loader
    
    def compute_loss(self, model, inputs, return_outputs=False):

        # Get input ids
        input_ids_batch = inputs['batch_input_ids']
        batch_size = input_ids_batch.size(0)
        input_length = input_ids_batch.size(1)

        # Get labels
        labels_batch = inputs['labels_batch']

        # Get KV Cache
        split_memory_ids_batch = inputs['split_memory_ids_batch']
        split_memory_position_batch = inputs['split_memory_position_batch']
        split_past_key_values = generate_kv_with_position(self.model, split_memory_ids_batch, position_ids = split_memory_position_batch)

        num_memory_each_sample = split_memory_ids_batch.size(0) // batch_size
        past_key_values_batch = concat_kv(split_past_key_values, num_memory_each_sample)

        # Get Attention Masks
        attention_mask_batch = inputs['attention_mask_batch']

        # Get Loss
        outputs = self.model(input_ids = input_ids_batch, attention_mask = attention_mask_batch, labels = labels_batch, past_key_values = past_key_values_batch, use_cache = True)

        return outputs.loss

class CustomTrainerBiasAttn(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        attention_matrices = []
        max_length = max(inputs['input_length'])
        for idx in range(len(inputs['input_ids'])):
            mem_num = inputs['mem_num'][idx]
            attention_matrices.append(
                construct_biased_attention_matrix(
                    inputs['input_length'][idx],
                    inputs['biased_index'][idx][:mem_num],
                    max_length,
                    inputs['input_ids'].device
                ).unsqueeze(0)
            )

        outputs = model(input_ids = inputs['input_ids'], attention_mask = torch.stack(attention_matrices), labels = inputs['labels'])

        return (outputs.loss, outputs) if return_outputs else outputs.loss
