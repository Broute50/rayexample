import ray
from ray.train.lightgbm import LightGBMTrainer
from ray.air.config import ScalingConfig
import ray.data
from sklearn.metrics import accuracy_score
import pickle

# Ray'i başlat ve cluster'a bağlan
ray.init(address='auto', ignore_reinit_error=True)

# Ray Data API kullanarak CSV dosyalarını yükleyelim
train_data = ray.data.read_csv('train_data.csv')
test_data = ray.data.read_csv('test_data.csv')

# Veriyi işlemeye başla
def preprocess_data(ray_dataset):
    # Özellikler ve hedefi ayırıyoruz
    features = ray_dataset.drop_columns(["target"])
    target = ray_dataset.select_columns(["target"])
    
    # Özellikler ve hedef verilerini numpy dizilerine dönüştür
    features_np = features.to_numpy()
    target_np = target.to_numpy().flatten()  # Tek boyutlu hale getir
    return features_np, target_np

# Eğitim ve test verilerini hazırlama
X_train, y_train = preprocess_data(train_data)
X_test, y_test = preprocess_data(test_data)

# Ray-LightGBM kullanarak model eğitimi yapıyoruz
def train_ray_lightgbm(X_train, y_train):
    # Eğitim verisini birleştiriyoruz
    train_dataset = ray.data.from_numpy(np.column_stack((X_train, y_train)))

    # Ray-LightGBM Trainer kullanarak modeli dağıtık bir şekilde eğitiyoruz
    trainer = LightGBMTrainer(
        scaling_config=ScalingConfig(num_workers=2),  # 2 worker kullanıyoruz
        label_column="target",
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
def predict_and_score(model, X_test, y_test):
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
    ray.get(predict_and_score.remote(model, X_test, y_test))

    # Ray'i kapat
    ray.shutdown()
