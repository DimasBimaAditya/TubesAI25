import serial
import csv
import time
import os

ARDUINO_PORT = 'COM13'
BAUD_RATE = 9600

DATASET_FILENAME = 'fire_detection_dataset_uno_only.csv'

COLLECTION_DURATION_PER_CRITERIA = 60

CRITERIA = {
    "Normal": 0,
    "Warning": 1,
    "Fire": 2
}

try:
    ser = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print(f"Terhubung ke Arduino Uno di {ARDUINO_PORT}")
except serial.SerialException as e:
    print(f"Error: Tidak dapat terhubung ke Arduino Uno. Pastikan port sudah benar dan Arduino terhubung. ({e})")
    exit()

print("\n--- Mulai Pengumpulan Data Dataset Otomatis (Arduino Uno Saja) ---")
print(f"Setiap kriteria akan dikumpulkan selama {COLLECTION_DURATION_PER_CRITERIA} detik (1 menit).")
print("Pastikan sensor terpasang ke Arduino Uno dan dalam kondisi yang sesuai saat instruksi muncul.")
print("------------------------------------")

file_exists = os.path.isfile(DATASET_FILENAME)
mode = 'a' if file_exists else 'w'

with open(DATASET_FILENAME, mode, newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    if not file_exists:
        csv_writer.writerow(['infrared_value', 'mq2_value', 'humidity', 'temperature', 'label'])

    for criterion_name, label_value in CRITERIA.items():
        input(f"\n--- Tekan ENTER untuk mulai mengumpulkan data untuk KONDISI: '{criterion_name}' (Label: {label_value}) ---")
        print(f"** Pastikan sensor berada dalam kondisi '{criterion_name}' yang sesuai sekarang. **")
        print(f"Mengumpulkan data selama {COLLECTION_DURATION_PER_CRITERIA} detik...")

        start_time = time.time()
        collected_count = 0

        while (time.time() - start_time) < COLLECTION_DURATION_PER_CRITERIA:
            try:
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

                                csv_writer.writerow([infrared_val, mq2_val, humidity_val, temperature_val, label_value])
                                collected_count += 1
                        except (ValueError, IndexError) as e:
                            pass
                time.sleep(0.01)
            except KeyboardInterrupt:
                print("\nPengumpulan data dihentikan secara manual untuk kriteria ini.")
                break
            except Exception as e:
                print(f"Terjadi error saat mengumpulkan data untuk {criterion_name}: {e}")
                break

        print(f"Selesai mengumpulkan data untuk '{criterion_name}'. Total {collected_count} sampel terkumpul.")

print("\n--- Pengumpulan Dataset Selesai ---")
print(f"Dataset berhasil disimpan ke {DATASET_FILENAME}")

ser.close()
