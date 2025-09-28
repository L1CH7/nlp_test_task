from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, Trainer, TrainingArguments, pipeline
import torch
import pandas as pd
from datasets import Dataset

class ChatLLM:
    def __init__(self, model_name, system_prompt, device=0):
        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() and device >= 0 else "cpu")
        
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = 2
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, config=config, ignore_mismatched_sizes=True
        ).to(self.device)
        
        self.system_prompt = system_prompt
    
    def predict(self, X):
        results = []
        self.model.eval()
        
        for row in X:
            csv_data = ','.join(map(str, row))
            prompt = f"{self.system_prompt}\n\nДанные: {csv_data}\nОтвет:"
            
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                prediction = torch.argmax(outputs.logits, dim=-1).item()
            
            results.append(prediction)
        
        return results

class FineTunedLLM:
    def __init__(self, model_name, device=0):
        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() and device >= 0 else "cpu")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = None  # Будет создан при fit
    
    def fit(self, X_train, y_train, X_dev, y_dev, system_prompt):
        # Создаем тексты для обучения
        train_texts = [f"{system_prompt}\n\nДанные: {','.join(map(str, row))}\nОтвет:" for row in X_train]
        dev_texts = [f"{system_prompt}\n\nДанные: {','.join(map(str, row))}\nОтвет:" for row in X_dev]
        
        # Создаем датасеты
        train_data = Dataset.from_dict({
            'text': train_texts,
            'labels': y_train.tolist()
        })
        dev_data = Dataset.from_dict({
            'text': dev_texts, 
            'labels': y_dev.tolist()
        })
        
        def tokenize(batch):
            return self.tokenizer(batch['text'], truncation=True, padding=True, max_length=512)
        
        train_data = train_data.map(tokenize, batched=True)
        dev_data = dev_data.map(tokenize, batched=True)
        
        # Создаем модель
        config = AutoConfig.from_pretrained(self.model_name)
        config.num_labels = 2
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, config=config
        ).to(self.device)
        
        # Обучение
        training_args = TrainingArguments(
            output_dir='./ft_output',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            learning_rate=2e-5,
            evaluation_strategy='epoch',
            save_strategy='no',
            logging_steps=10,
            fp16=torch.cuda.is_available()
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=dev_data,
            tokenizer=self.tokenizer
        )
        
        trainer.train()
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not fitted yet!")
            
        results = []
        self.model.eval()
        
        for row in X:
            csv_data = ','.join(map(str, row))
            inputs = self.tokenizer(
                csv_data, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                prediction = torch.argmax(outputs.logits, dim=-1).item()
            
            results.append(prediction)
        
        return results
