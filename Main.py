import serial
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import messagebox

ARDUINO_PORT = 'COM13'
BAUD_RATE = 9600

DATASET_FILENAME = 'fire_detection_dataset_uno_only.csv'
KNN_MODEL_FILENAME = 'knn_fire_detector_uno_only.pkl'
RF_MODEL_FILENAME = 'rf_fire_detector_uno_only.pkl'

CRITERIA = {
    "Normal": 0,
    "Warning": 1,
    "Fire": 2
}

LABEL_TO_STATUS = {v: k for k, v in CRITERIA.items()}

try:
    ser = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print(f"Terhubung ke Arduino Uno di {ARDUINO_PORT}")
except serial.SerialException as e:
    print(f"Error: Tidak dapat terhubung ke Arduino Uno. Pastikan port sudah benar dan Arduino terhubung. ({e})")
    messagebox.showerror("Serial Error", f"Tidak dapat terhubung ke Arduino Uno.\nPastikan port sudah benar dan tidak digunakan.\n{e}")
    exit()

def send_command_to_arduino(command):
    try:
        ser.write(f"{command}\n".encode('utf-8'))
    except Exception as e:
        print(f"Gagal mengirim perintah '{command}' ke Arduino: {e}")

def train_and_save_models():
    global knn_accuracy, rf_accuracy

    try:
        df = pd.read_csv(DATASET_FILENAME)
        print(f"Dataset '{DATASET_FILENAME}' berhasil dimuat.")
        print("Preview Dataset:\n", df.head())
        if df.empty:
            messagebox.showerror("Dataset Error", "Dataset kosong setelah dimuat. Tidak bisa melatih model.")
            raise ValueError("Dataset kosong setelah dimuat. Tidak bisa melatih model.")
    except FileNotFoundError:
        messagebox.showerror("Dataset Error", f"Dataset '{DATASET_FILENAME}' tidak ditemukan.\nHarap jalankan program pengumpul data terlebih dahulu.")
        raise FileNotFoundError(f"Dataset '{DATASET_FILENAME}' tidak ditemukan.")
    except Exception as e:
        messagebox.showerror("Dataset Error", f"Error saat memuat dataset: {e}")
        raise e

    FEATURE_COLUMNS = ['infrared_value', 'mq2_value', 'humidity', 'temperature']
    X = df[FEATURE_COLUMNS]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nJumlah data training: {len(X_train)}")
    print(f"Jumlah data testing: {len(X_test)}")

    print("\nMelatih model K-Nearest Neighbors (KNN)...")
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)
    knn_accuracy = accuracy_score(y_test, y_pred_knn)
    print(f"Akurasi KNN pada data testing: {knn_accuracy:.2f}")
    print("Classification Report KNN:\n", classification_report(y_test, y_pred_knn, target_names=['Normal', 'Warning', 'Fire']))
    joblib.dump(knn_model, KNN_MODEL_FILENAME)
    print(f"Model KNN disimpan sebagai '{KNN_MODEL_FILENAME}'")

    print("\nMelatih model Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    print(f"Akurasi Random Forest pada data testing: {rf_accuracy:.2f}")
    print("Classification Report Random Forest:\n", classification_report(y_test, y_pred_rf, target_names=['Normal', 'Warning', 'Fire']))
    joblib.dump(rf_model, RF_MODEL_FILENAME)
    print(f"Model Random Forest disimpan sebagai '{RF_MODEL_FILENAME}'")

    return knn_model, rf_model

def load_or_train_models():
    global knn_accuracy, rf_accuracy

    try:
        knn_model = joblib.load(KNN_MODEL_FILENAME)
        rf_model = joblib.load(RF_MODEL_FILENAME)
        print(f"Model KNN ('{KNN_MODEL_FILENAME}') dan Random Forest ('{RF_MODEL_FILENAME}') berhasil dimuat.")
        
        df = pd.read_csv(DATASET_FILENAME)
        if df.empty:
            messagebox.showerror("Dataset Error", "Dataset kosong saat memuat model. Tidak bisa melakukan prediksi akurasi.")
            raise ValueError("Dataset kosong saat memuat model. Tidak bisa melakukan prediksi akurasi.")
        
        X = df[FEATURE_COLUMNS]
        y = df['label']
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        knn_accuracy = accuracy_score(y_test, knn_model.predict(X_test))
        rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test))
        
        print(f"Akurasi model yang dimuat: KNN={knn_accuracy:.2f}, Random Forest={rf_accuracy:.2f}")

        return knn_model, rf_model
    except FileNotFoundError:
        print("Model tidak ditemukan. Melatih model baru dari dataset...")
        return train_and_save_models()
    except Exception as e:
        messagebox.showerror("Model Error", f"Error saat memuat model: {e}.\nMelatih model baru...")
        return train_and_save_models()

knn_accuracy = 0.0
rf_accuracy = 0.0

FEATURE_COLUMNS = ['infrared_value', 'mq2_value', 'humidity', 'temperature']

root = tk.Tk()
root.title("Sistem Deteksi Kebakaran AI")
root.geometry("400x550")

root.tk_setPalette(background='#f0f0f0', foreground='#333333')
font_large = ('Arial', 14, 'bold')
font_medium = ('Arial', 12)
font_status = ('Arial', 16, 'bold')
font_prediction = ('Arial', 14, 'bold')

status_label = tk.Label(root, text="STATUS: N/A", font=font_status, pady=10)
status_label.pack(pady=10)

sensor_frame = tk.LabelFrame(root, text="Data Sensor Real-time", font=font_medium, padx=10, pady=10)
sensor_frame.pack(padx=20, pady=10, fill="x")

labels = {}
sensor_names = ["Infrared", "MQ-2", "Kelembaban", "Suhu"]
for name in sensor_names:
    row_frame = tk.Frame(sensor_frame)
    row_frame.pack(fill="x", pady=2)
    tk.Label(row_frame, text=f"{name}:", font=font_medium).pack(side="left")
    labels[name] = tk.Label(row_frame, text="N/A", font=font_medium, fg="blue")
    labels[name].pack(side="right")

prediction_frame = tk.LabelFrame(root, text="Prediksi AI", font=font_medium, padx=10, pady=10)
prediction_frame.pack(padx=20, pady=10, fill="x")

labels['KNN_Pred'] = tk.Label(prediction_frame, text="KNN: N/A", font=font_prediction, fg="purple")
labels['KNN_Pred'].pack(fill="x", pady=5)
labels['RF_Pred'] = tk.Label(prediction_frame, text="Random Forest: N/A", font=font_prediction, fg="darkgreen")
labels['RF_Pred'].pack(fill="x", pady=5)

control_frame = tk.Frame(root, pady=10)
control_frame.pack(pady=10)

off_button = tk.Button(control_frame, text="MATIKAN AKTULATOR", command=lambda: send_command_to_arduino("OFF"), bg="red", fg="white", font=font_medium, width=20, height=2)
off_button.pack(side="left", padx=5)

def on_closing():
    if messagebox.askokcancel("Keluar", "Anda yakin ingin keluar?"):
        ser.close()
        root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

def update_gui():
    global previous_status_to_arduino
    if ser.in_waiting > 0:
        line = ser.readline().decode('utf-8').strip()
        if line:
            try:
                parts = line.split(',')
                if len(parts) == 5:
                    infrared_val = int(parts[0])
                    mq2_val = int(parts[1])
                    humidity_val = float(parts[2])
                    temperature_val = float(parts[3])
                    status_arduino_raw = parts[4]

                    labels["Infrared"].config(text=str(infrared_val))
                    labels["MQ-2"].config(text=str(mq2_val))
                    labels["Kelembaban"].config(text=f"{humidity_val:.1f} %")
                    labels["Suhu"].config(text=f"{temperature_val:.1f} °C")

                    current_data = pd.DataFrame([[infrared_val, mq2_val, humidity_val, temperature_val]],
                                                  columns=FEATURE_COLUMNS)
                    
                    pred_knn = knn_model.predict(current_data)[0]
                    status_knn = LABEL_TO_STATUS.get(pred_knn, "UNKNOWN")
                    pred_rf = rf_model.predict(current_data)[0]
                    status_rf = LABEL_TO_STATUS.get(pred_rf, "UNKNOWN")

                    labels['KNN_Pred'].config(text=f"KNN: {status_knn}")
                    labels['RF_Pred'].config(text=f"Random Forest: {status_rf}")

                    final_ai_status = status_rf

                    status_label.config(text=f"STATUS: {final_ai_status}")
                    if final_ai_status == "NORMAL":
                        status_label.config(bg="#d4edda", fg="#155724")
                    elif final_ai_status == "WARNING":
                        status_label.config(bg="#fff3cd", fg="#856404")
                    elif final_ai_status == "FIRE":
                        status_label.config(bg="#f8d7da", fg="#721c24")
                    
                    if final_ai_status != previous_status_to_arduino:
                        send_command_to_arduino(final_ai_status)
                        previous_status_to_arduino = final_ai_status
                else:
                    print(f"Format data tidak sesuai dari Arduino: {line}")
            except (ValueError, IndexError) as e:
                print(f"Error parsing data or updating GUI: {e} - Raw: {line}")
            except Exception as e:
                print(f"Terjadi error tak terduga: {e}")
    
    root.after(50, update_gui)

try:
    knn_model, rf_model = load_or_train_models()
except (FileNotFoundError, ValueError) as e:
    print(f"Tidak dapat melanjutkan karena: {e}")
    root.destroy()
    exit()

previous_status_to_arduino = "UNKNOWN"

send_command_to_arduino("OFF")

root.after(50, update_gui)
root.mainloop()

print("\n--- Menampilkan Grafik Perbandingan Akurasi Model ---")

models = ['KNN', 'Random Forest']
accuracies = [knn_accuracy, rf_accuracy]

plt.figure(figsize=(8, 6))
plt.bar(models, accuracies, color=['skyblue', 'lightcoral'])
plt.ylim(0, 1)
plt.ylabel('Akurasi')
plt.title('Perbandingan Akurasi Model KNN vs Random Forest')
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

print("\n--- Ulasan Perbandingan Metode ---")
if rf_accuracy > knn_accuracy:
    print("Metode **Random Forest** menunjukkan akurasi yang lebih baik.")
    print("Alasan:")
    print("   - Random Forest adalah *ensemble method* yang membangun banyak *decision tree* dan menggabungkan hasil prediksinya.")
    print("   - Ini membuatnya lebih kuat dan lebih tahan terhadap *overfitting* dibandingkan KNN, terutama pada dataset yang mungkin memiliki *noise* atau fitur yang saling terkait.")
    print("   - Kemampuannya untuk menangani data non-linear dan interaksi kompleks antar fitur seringkali menghasilkan performa yang lebih stabil dan akurat.")
elif knn_accuracy > rf_accuracy:
    print("Metode **K-Nearest Neighbors (KNN)** menunjukkan akurasi yang lebih baik.")
    print("Alasan:")
    print("   - Ini mungkin terjadi jika dataset Anda memiliki batasan kelas yang jelas dan relatif linier, serta fitur-fitur yang cukup terpisah.")
    print("   - KNN adalah algoritma non-parametrik yang bergantung pada jarak antar titik data. Jika data Anda memiliki 'kluster' yang jelas untuk setiap kelas (Normal, Warning, Fire), KNN bisa sangat efektif.")
    print("   - Performa KNN sangat sensitif terhadap skala fitur dan pemilihan nilai 'k' (jumlah tetangga). Hasil yang lebih baik bisa menunjukkan bahwa data Anda sudah cukup baik dan nilai 'k' yang dipilih optimal.")
else:
    print("Kedua metode (KNN dan Random Forest) menunjukkan akurasi yang sama.")
    print("Alasan:")
    print("   - Hal ini menunjukkan bahwa dataset Anda mungkin memiliki pola yang cukup sederhana dan mudah dipisahkan, sehingga kedua algoritma mampu mengklasifikasikannya dengan baik.")
    print("   - Dalam kasus ini, Anda bisa memilih salah satu berdasarkan preferensi komputasi atau interpretasi model.")

print("\nPerlu diingat bahwa performa model sangat bergantung pada kualitas dan jumlah data pelatihan. Semakin baik dataset Anda, semakin akurat model yang dihasilkan.")
