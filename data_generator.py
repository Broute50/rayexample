import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Rastgele bir dataset oluşturma
def generate_data():
    np.random.seed(42)
    data_size = 1000
    X = np.random.rand(data_size, 5)  # 5 özellikli veri
    y = (X[:, 0] + X[:, 1] * 2 + np.random.rand(data_size)) > 1.5  # Basit bir hedef değişken
    
    # Eğitim ve test verilerine ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Eğitim verisini CSV'ye kaydetme
    train_data = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(1, 6)])
    train_data['target'] = y_train
    train_data.to_csv('train_data.csv', index=False)
    
    # Test verisini CSV'ye kaydetme
    test_data = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(1, 6)])
    test_data['target'] = y_test
    test_data.to_csv('test_data.csv', index=False)

if __name__ == '__main__':
    generate_data()
    print("Train ve test verileri oluşturuldu ve CSV'ye kaydedildi.")
