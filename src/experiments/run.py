import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from src.models.factory import create_model, load_prompts

def run_experiments(config_path: str, dataset_key: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Загружаем промпты в память
    prompts = load_prompts(config['prompts'])

    # Данные
    data_cfg = config['data']
    data_path = data_cfg[f"{dataset_key}_path"]
    features = data_cfg['features']
    label = data_cfg['label']
    split_cfg = data_cfg['split']

    df = pd.read_csv(data_path)
    X = df[features]
    y = df[label]

    print(f"Загружены данные: {len(df)} строк")
    print(f"Распределение классов: {y.value_counts().to_dict()}")

    # Feature engineering
    X_enhanced = X.copy()
    X_enhanced['p_duration'] = X['p_end'] - X['p_onset']
    X_enhanced['qrs_duration'] = X['qrs_end'] - X['qrs_onset'] 
    X_enhanced['heart_rate'] = 60000 / X['rr_interval']

    # Разбиение
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_enhanced, y,
        test_size=split_cfg['test_size'] + split_cfg['dev_size'],
        stratify=y,
        random_state=split_cfg['random_state']
    )
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_temp, y_temp,
        test_size=split_cfg['test_size'] / (split_cfg['test_size'] + split_cfg['dev_size']),
        stratify=y_temp,
        random_state=split_cfg['random_state']
    )

    # Масштабирование для sklearn
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    # Классические методы
    for model_cfg in config['models']['sklearn_methods']:
        name = model_cfg['name']
        print(f"Запуск модели: {name}")
        try:
            model = create_model(model_cfg)
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            
            report = classification_report(y_test, preds, output_dict=True, zero_division=0)
            results.append({
                'model': name,
                'accuracy': report['accuracy'],
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1': report['weighted avg']['f1-score']
            })
            print(f"{name} F1: {report['weighted avg']['f1-score']:.3f}")

        except Exception as e:
            print(f"Ошибка в модели {name}: {e}")

    # LLM методы
    for model_cfg in config['models']['llm_methods']:
        name = model_cfg['name']
        print(f"Запуск модели: {name}")
        try:
            model = create_model(model_cfg, prompts)

            if model_cfg['type'] == 'llm_finetune':
                # Исправлено! Берем prompt_key из конфига модели
                prompt_key = model_cfg['prompt_key']
                system_prompt = prompts[prompt_key]
                
                model.fit(
                    X_train[features].values, y_train, 
                    X_dev[features].values, y_dev, 
                    system_prompt,
                    training_params=getattr(model, 'training_params', {}),
                    save_model=getattr(model, 'save_model', False),
                    save_path=getattr(model, 'save_path', None)
                )
                preds = model.predict(X_test[features].values)
            
            elif model_cfg['type'] == 'llm_chat':
                # Chat модели (остается как было)
                preds = model.predict(X_test[features].values)
            
            elif model_cfg['type'] == 'rule_based':
                # Rule-based (остается как было)
                preds = model.predict(X_test[features].values)

            report = classification_report(y_test, preds, output_dict=True, zero_division=0)
            results.append({
                'model': name,
                'accuracy': report['accuracy'],
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1': report['weighted avg']['f1-score']
            })
            print(f"{name} F1: {report['weighted avg']['f1-score']:.3f}")

        except Exception as e:
            print(f"Ошибка в модели {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'model': name, 'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0
            })

    # Сохраняем
    out_df = pd.DataFrame(results)
    out_df.to_csv(config['experiments']['results_csv'], index=False)
    print("\nИтоговые результаты:")
    print(out_df.round(3))
    print(f"Результаты сохранены в {config['experiments']['results_csv']}")
