import ray
from ray.train.lightgbm import LightGBMTrainer
from ray.air.config import ScalingConfig
import ray.data
import numpy as np
import pickle
from sklearn.metrics import accuracy_score

# Ray'i başlat ve cluster'a bağlan
ray.init(address='auto', ignore_reinit_error=True)

# Ray Data API kullanarak CSV dosyalarını yükleyelim
train_data = ray.data.read_csv('train_data.csv')
test_data = ray.data.read_csv('test_data.csv')

# Ray Dataset'ini LightGBM uyumlu numpy array'e dönüştürme
def preprocess_data(ray_dataset):
    # Batch'ler halinde verileri alıyoruz
    batches = list(ray_dataset.iter_batches(batch_size=None, batch_format="numpy"))
    features = np.concatenate([batch[:, :-1] for batch in batches], axis=0)
    target = np.concatenate([batch[:, -1] for batch in batches], axis=0)
    return features, target

# Eğitim ve test verilerini hazırlama
X_train, y_train = preprocess_data(train_data)
X_test, y_test = preprocess_data(test_data)

# Ray-LightGBM kullanarak model eğitimi yapıyoruz
def train_ray_lightgbm(X_train, y_train):
    # Eğitim verilerini LightGBMTrainer için uygun formata çevirme
    train_dataset = ray.data.from_numpy(np.column_stack((X_train, y_train)))

    # Ray-LightGBM Trainer kullanarak modeli dağıtık bir şekilde eğitiyoruz
    trainer = LightGBMTrainer(
        scaling_config=ScalingConfig(num_workers=2),  # 2 worker kullanıyoruz
        label_column=train_dataset.schema().names[-1],  # Son sütunu hedef olarak alıyoruz
        params={
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "verbose": -1
        },
        datasets={"train": train_dataset}
    )

    # Modeli eğit
    result = trainer.fit()
    model = result.best_model
    
    # Eğitilmiş modeli pickle formatında kaydediyoruz
    with open('lightgbm_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("Model başarıyla eğitildi ve 'lightgbm_model.pkl' olarak kaydedildi.")
    return model

# Ray remote decorator kullanarak işlemi dağıtık hale getirme
@ray.remote
def predict_and_score(X_test, y_test):
    # Kaydedilen modeli yükle
    with open('lightgbm_model.pkl', 'rb') as f:
        model = pickle.load(f)
        
    # Tahmin yapma
    y_pred = model.predict(X_test)
    y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_pred]
    
    # Accuracy hesaplama
    accuracy = accuracy_score(y_test, y_pred_binary)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Tahminleri ve gerçek değerleri CSV'ye kaydetme
    result = ray.data.from_items([{'true_value': tv, 'predicted_value': pv} for tv, pv in zip(y_test, y_pred_binary)])
    result.write_csv('scores.csv')
    print("Tahmin sonuçları 'scores.csv' olarak kaydedildi.")

# Modeli eğit ve tahminleri yap
if __name__ == '__main__':
    # Modeli Ray-LightGBM ile eğit
    model = train_ray_lightgbm(X_train, y_train)

    # Modeli Ray remote task ile test et
    ray.get(predict_and_score.remote(X_test, y_test))

    # Ray'i kapat
    ray.shutdown()
