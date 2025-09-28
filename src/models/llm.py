from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, Trainer, TrainingArguments
import torch
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score

class ChatLLM:
    def __init__(self, model_name, system_prompt, device=0):
        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() and device >= 0 else "cpu")
        self.model_name = model_name
        self.system_prompt = system_prompt
        
        # Используем готовую fine-tuned модель для текстовой классификации
        from transformers import pipeline
        try:
            self.classifier = pipeline(
                "text-classification", 
                model=model_name,
                tokenizer=model_name,
                device=0 if torch.cuda.is_available() and device >= 0 else -1,
                top_k=None  # Исправлено! Вместо return_all_scores=True
            )
        except:
            # Fallback на прямой токенизатор+модель
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            config = AutoConfig.from_pretrained(model_name)
            config.num_labels = 2
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, config=config, ignore_mismatched_sizes=True
            ).to(self.device)
            self.classifier = None
    
    def _format_input(self, row):
        """Форматируем входные данные согласно промпту"""
        csv_data = ','.join(map(str, row))
        return f"{self.system_prompt}\n\nДанные: {csv_data}\nОтвет:"
    
    def predict(self, X):
        results = []
        
        if self.classifier:
            # Используем pipeline
            for row in X:
                text = self._format_input(row)
                
                try:
                    outputs = self.classifier(text)
                    
                    if isinstance(outputs, list) and len(outputs) > 0:
                        if isinstance(outputs[0], list):
                            best_pred = max(outputs[0], key=lambda x: x['score'])
                        else:
                            best_pred = outputs[0]
                        
                        if 'POSITIVE' in best_pred['label'] or 'LABEL_1' in best_pred['label']:
                            prediction = 1
                        else:
                            prediction = 0
                    else:
                        prediction = 0  # default
                except:
                    # Случайное предсказание если ошибка
                    prediction = 1 if len(str(row[0])) % 2 == 0 else 0
                
                results.append(prediction)
        else:
            # Используем прямую модель
            self.model.eval()
            for row in X:
                text = self._format_input(row)
                inputs = self.tokenizer(
                    text, 
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
        self.model = None
    
    def fit(self, X_train, y_train, X_dev, y_dev, system_prompt, training_params=None, save_model=False, save_path=None):
        print("Starting fine-tuning...")
        
        # Дефолтные параметры обучения
        default_params = {
            'num_train_epochs': 30,
            'per_device_train_batch_size': 8,
            'per_device_eval_batch_size': 8,
            'learning_rate': 2e-5,
            'weight_decay': 0.01,
            'warmup_steps': 0,
            'warmup_ratio': 0.0,
            'lr_scheduler_type': 'linear',
            'logging_steps': 10,
        }
        
        # Объединяем с пользовательскими параметрами
        if training_params:
            default_params.update(training_params)
            print(f"Using custom training params: {training_params}")
        
        # Создаем простые текстовые представления
        def format_text(row):
            return ','.join(map(str, row))
        
        train_texts = [format_text(row) for row in X_train]
        dev_texts = [format_text(row) for row in X_dev]
        
        print(f"Пример текста: {train_texts[0]}")
        print(f"Train size: {len(train_texts)}, Dev size: {len(dev_texts)}")
        print(f"Training epochs: {default_params['num_train_epochs']}, LR: {default_params['learning_rate']}")
        
        # Токенизация сразу
        train_encodings = self.tokenizer(
            train_texts,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        
        dev_encodings = self.tokenizer(
            dev_texts,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        
        # Создаем простой датасет
        class ECGDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = torch.tensor(labels, dtype=torch.long)
            
            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item['labels'] = self.labels[idx]
                return item
            
            def __len__(self):
                return len(self.labels)
        
        train_dataset = ECGDataset(train_encodings, y_train.tolist())
        dev_dataset = ECGDataset(dev_encodings, y_dev.tolist())
        
        # Создаем модель
        config = AutoConfig.from_pretrained(self.model_name)
        config.num_labels = 2
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, config=config, ignore_mismatched_sizes=True
        ).to(self.device)
        
        # Создаем папку для сохранения, если нужно
        if save_model and save_path:
            import os
            os.makedirs(save_path, exist_ok=True)
        
        # Training arguments с полным контролем
        training_args = TrainingArguments(
            output_dir=save_path if save_path else './ft_output',
            num_train_epochs=default_params['num_train_epochs'],
            per_device_train_batch_size=default_params['per_device_train_batch_size'],
            per_device_eval_batch_size=default_params['per_device_eval_batch_size'],
            learning_rate=default_params['learning_rate'],
            weight_decay=default_params['weight_decay'],
            warmup_steps=default_params['warmup_steps'],
            warmup_ratio=default_params['warmup_ratio'],
            lr_scheduler_type=default_params['lr_scheduler_type'],
            logging_steps=default_params['logging_steps'],
            eval_strategy='epoch',
            save_strategy='epoch' if save_model else 'no',
            fp16=torch.cuda.is_available(),
            load_best_model_at_end=save_model,
            metric_for_best_model='eval_loss' if save_model else None,
            greater_is_better=False,
            report_to=None,  # Отключаем wandb/tensorboard логирование
            dataloader_drop_last=True,
            remove_unused_columns=False
        )
        
        # Функция для вычисления метрик
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = predictions.argmax(axis=1)
            from sklearn.metrics import accuracy_score, f1_score
            return {
                'accuracy': accuracy_score(labels, predictions),
                'f1': f1_score(labels, predictions, average='weighted')
            }
        
        # Создаем trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            processing_class=self.tokenizer,
            compute_metrics=compute_metrics
        )
        
        # Обучаем
        print(f"Starting training with {len(train_dataset)} samples...")
        trainer.train()
        
        # Сохраняем модель, если требуется
        if save_model and save_path:
            print(f"Saving best model to {save_path}")
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            
            # Сохраняем информацию о конфигурации
            import json
            config_info = {
                'model_name': self.model_name,
                'training_params': default_params,
                'final_eval_metrics': trainer.evaluate()
            }
            with open(f"{save_path}/training_info.json", 'w') as f:
                json.dump(config_info, f, indent=2)
        
        print("Fine-tuning completed!")
        
        # Выводим финальные метрики
        final_metrics = trainer.evaluate()
        print(f"Final eval metrics: {final_metrics}")

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not fitted yet!")
        
        results = []
        self.model.eval()
        
        for row in X:
            text = ','.join(map(str, row))
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                prediction = torch.argmax(outputs.logits, dim=-1).item()
            
            results.append(prediction)
        
        return results

class SimpleRuleBasedClassifier:
    """Правило-основанный классификатор на основе медицинских норм"""
    def __init__(self):
        pass
    
    def predict(self, X):
        results = []
        for row in X:
            rr, p_start, p_end, qrs_start, qrs_end, t_end, p_axis, qrs_axis, t_axis = row[:9]
            
            # Подсчет аномалий
            anomaly_score = 0
            
            # RR интервал: норма 600-1000 (вес 2 - очень важный)
            if not (600 <= rr <= 1000):
                anomaly_score += 2
            
            # P-волна длительность: норма 60-120
            p_duration = p_end - p_start
            if not (60 <= p_duration <= 120):
                anomaly_score += 1
            
            # QRS длительность: норма 60-110 (вес 2 - очень важный)
            qrs_duration = qrs_end - qrs_start
            if not (60 <= qrs_duration <= 110):
                anomaly_score += 2
            
            # Электрические оси (каждая вес 1)
            if not (0 <= p_axis <= 75):
                anomaly_score += 1
            if not (-30 <= qrs_axis <= 90):
                anomaly_score += 1
            if not (0 <= t_axis <= 90):
                anomaly_score += 1
            
            # Если аномалий много, считаем нездоровым
            # Порог 3: учитывает, что может быть 1-2 небольших отклонения при норме
            results.append(1 if anomaly_score <= 2 else 0)
        
        return results
