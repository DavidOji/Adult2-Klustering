# Benötigte Bibliotheken importieren
import pandas as pd
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.preprocessing import LabelEncoder

# 1. Laden Sie die Datei Adults.csv (mit Spaltennamen)
file_path = "adult 2.csv"
data = pd.read_csv(file_path)

# 2. Geben Sie die ersten Zeilen aus
print("Erste Zeilen des Datensatzes:")
print(data.head())

# 3. Transformieren Sie alle Spalten mit Zeichenketten in numerische Werte
label_encoder = LabelEncoder()
for column in data.select_dtypes(include='object').columns:
    data[column] = label_encoder.fit_transform(data[column])

# 4. Ermitteln Sie die optimale Gruppenanzahl
features = data.drop(['Label'], axis=1)  # Entfernen Sie die Zielvariable 'Label'
model = KMeans()
visualizer = KElbowVisualizer(model, k=(2, 9), timings=False)
visualizer.fit(features)
visualizer.show()

# 5. Gruppieren Sie den Datensatz
optimal_clusters = visualizer.elbow_value_
kmeans = KMeans(n_clusters=optimal_clusters)
pred_labels = kmeans.fit_predict(features)

# Die neuen Spalten "Label" (Gruppennummer) zum Datensatz hinzufügen
data['Label'] = pred_labels

# Speichern Sie die Tabelle mit der neuen Spalte "Label" (Gruppennummer) in einer neuen CSV-Datei
output_file_path = "data_with_labels.csv"
data.to_csv(output_file_path, index=False)

print(f"\nDatensatz mit Gruppennummern wurde erfolgreich in {output_file_path} gespeichert.")
