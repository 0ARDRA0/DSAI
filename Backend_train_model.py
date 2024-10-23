from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    Trainer,
    TrainingArguments,
    AdamW,
)
import json
import torch
from torch.utils.data import Dataset
import torch.quantization as quant


# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, data_file):
        with open(data_file, "r") as f:
            self.data = json.load(f)
        self.tokenizer = LlamaTokenizer.from_pretrained("openlm-research/open_llama_3b")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = " ".join(item["patterns"])
        output_text = " ".join(item["responses"])
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids
        output_ids = self.tokenizer(output_text, return_tensors="pt").input_ids
        return {"input_ids": input_ids.squeeze(), "labels": output_ids.squeeze()}


# Load Dataset
dataset = CustomDataset("data/intents.json")

# Load Model
model = LlamaForCausalLM.from_pretrained("openlm-research/open_llama_3b")

# Define the AdamW Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    save_steps=10_000,
    save_total_limit=2,
    optim="adamw_torch",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    optimizers=(
        optimizer,
        None,
    ),
)

# Train the model
trainer.train()


# Quantization
def quantize_model(model):
    model.eval()
    # Prepare model for quantization
    model.qconfig = quant.default_qconfig
    quant.prepare(model, inplace=True)
    quant.convert(model, inplace=True)
    return model


# Quantize the model
quantized_model = quantize_model(model)

# Save the quantized model
quantized_model.save_pretrained("./trained_llama_model")
