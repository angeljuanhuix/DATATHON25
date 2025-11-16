import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# === LECTURA DE DADES ===
train_df = pd.read_csv("/Users/xavi/Desktop/GitHub/Datathon2025/train.csv", sep=";")
test_df = pd.read_csv("/Users/xavi/Desktop/GitHub/Datathon2025/test.csv", sep=";")

objective = "weekly_demand"


# === SELECCIÓ DE COLUMNES ÚTILS (sense massa buits) ===
def seleccionar_columnes_utilitzables(df_train, df_test, target):
    # Columnes disponibles a tots dos datasets
    comunes = list(set(df_train.columns) & set(df_test.columns))
    
    bones = []
    
    for columna in comunes:
        if columna in (target, "ID"):
            continue
        
        percent_null_train = df_train[columna].isna().mean()
        percent_null_test = df_test[columna].isna().mean()
        
        # Columnes amb menys del 50% de valors perduts
        if percent_null_train < 0.5 and percent_null_test < 0.5:
            bones.append(columna)
    
    return bones


columnes_bones = seleccionar_columnes_utilitzables(train_df, test_df, objective)
print(f"Columnes seleccionades: {len(columnes_bones)}")


# === NETEJA I CONVERSIÓ DE DADES ===
def processar_dataset(df, columnes):
    data = df[columnes].copy()
    
    # Gestió de valors absents segons el tipus de dada
    for col in data.columns:
        if data[col].dtype in ["int64", "float64"]:
            data[col] = data[col].fillna(data[col].median())
        else:
            if data[col].notna().any():
                moda = data[col].mode()
                valor = moda.iloc[0] if not moda.empty else "Unknown"
                data[col] = data[col].fillna(valor)
            else:
                data[col] = "Unknown"
    
    # Convertir textos/categoríques en codis numèrics
    cat_cols = data.select_dtypes(include="object").columns
    for col in cat_cols:
        data[col] = data[col].astype("category").cat.codes
    
    return data


X_train = processar_dataset(train_df, columnes_bones)
X_test = processar_dataset(test_df, columnes_bones)
y_train = train_df[objective]


# === ESCALAT MANUAL DE LA DEMANDA (per inflar prediccions) ===
multiplicador = 3.0
y_train_ampliat = y_train * multiplicador


# === ENTRENAMENT DEL MODEL ===
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train_ampliat)


# === PREDICCIÓ SOBRE TEST ===
prediccions = rf_model.predict(X_test)


# === POST-PROCESSAT PER GENERAR VALORS EXAGERATS ===
resultat_produccio = np.zeros(len(prediccions))

for idx, pred in enumerate(prediccions):
    if pred < 1000:
        resultat_produccio[idx] = pred * 2.5
    elif pred < 2000:
        resultat_produccio[idx] = pred * 2.2
    else:
        resultat_produccio[idx] = pred * 1.8

    # Valor mínim inicial
    resultat_produccio[idx] = max(resultat_produccio[idx], 600)

# Afegeix marge addicional
resultat_produccio = resultat_produccio + 200

# Arrodoniment a l'alça
resultat_produccio = np.ceil(resultat_produccio).astype(int)

# Mínim global final
resultat_produccio = np.where(resultat_produccio < 1200, 1200, resultat_produccio)


# === EXPORTACIÓ A CSV ===
output = pd.DataFrame({
    "ID": test_df["ID"],
    "Production": resultat_produccio
})

output.to_csv("output_producció_predicció.csv", index=False)
