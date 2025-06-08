# Falcon-7B Q&A Chatbot

This project implements a question-answering chatbot built on the Falcon-7B language model. It uses 4-bit quantization and LoRA fine-tuning for efficient training and inference. The chatbot is trained on custom Q&A data and generates human-like responses to user queries.

---

## Features

- Uses **tiiuae/falcon-7b** pretrained model with 4-bit quantization for low memory usage
- Applies **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning
- Supports training on custom question-answer datasets using Hugging Face `Trainer`
- Implements dataset balancing with oversampling to handle class imbalance
- Generates concise, relevant answers to user questions
- Easy-to-use generation function with configurable temperature, top-p, and max tokens

---

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Datasets
- PEFT (Parameter-Efficient Fine-Tuning)
- BitsAndBytes (for quantization)
- Imbalanced-learn (for oversampling)
- Pandas

Install dependencies:

```bash
pip install torch transformers datasets peft bitsandbytes imbalanced-learn pandas
````

---

## Setup & Usage

### 1. Load Model and Tokenizer

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
import torch

MODEL_NAME = "tiiuae/falcon-7b"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
```

### 2. Prepare Dataset

Format your data as a list of dictionaries with `"Query"` and `"Answer"` keys. Then convert it to a Hugging Face dataset and tokenize:

```python
from datasets import Dataset

hf_dataset = Dataset.from_dict({
    "Query": [item["Query"] for item in data],
    "Answer": [item["Answer"] for item in data]
})

def generate_prompt(data_point):
    return f"<human>: {data_point['Query']}\n<assistant>: {data_point['Answer']}"

def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(data_point)
    return tokenizer(full_prompt, padding=True, truncation=True)

tokenized_dataset = hf_dataset.train_test_split(test_size=0.2)
tokenized_dataset = tokenized_dataset.map(generate_and_tokenize_prompt)
```

### 3. (Optional) Balance Dataset with Oversampling

If your dataset is imbalanced across categories, oversample minority classes using `RandomOverSampler`:

```python
from imblearn.over_sampling import RandomOverSampler
import pandas as pd

oversampler = RandomOverSampler(sampling_strategy={
    'Food & Cooking': 3247,
    'Arts & Crafts': 3247,
    'Home & Lifestyle': 3247,
    'Vehicles': 1500,
    'Pets & Animals': 1500,
    'Holidays & Traditions': 1000,
    'Technology': 1000,
}, random_state=42)

X_balanced, y_balanced = oversampler.fit_resample(
    df[['title', 'transcript']],
    df['reduced_category']
)

df_balanced = pd.DataFrame({
    'title': X_balanced['title'],
    'transcript': X_balanced['transcript'],
    'reduced_category': y_balanced
})
```

### 4. Train the Model

Set up the `Trainer` and start fine-tuning:

```python
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,
    save_total_limit=3,
    logging_steps=1,
    output_dir="chatbot",
    max_steps=150,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    report_to="tensorboard",
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
)

trainer.train()
```

### 5. Generate Responses

Use this helper function to generate answers from the fine-tuned model:

```python
def generate_response(prompt, model, tokenizer, device="cuda:0"):
    model.config.use_cache = False
    formatted_prompt = f"<human>: {prompt}\n<assistant>:"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.7,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        use_cache=False,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("<assistant>:")[-1].strip()
```

---

## Notes

* This model uses LoRA and 4-bit quantization to reduce VRAM usage.
* Ensure your environment supports bitsandbytes and CUDA for best performance.
* Customize hyperparameters (learning rate, batch size, epochs) to fit your dataset and hardware.
* Dataset should be well-structured for optimal results.

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

* [Falcon-7B by TII UAE](https://huggingface.co/tiiuae/falcon-7b)
* Hugging Face Transformers & Datasets
* BitsAndBytes for quantization
* PEFT for efficient fine-tuning

