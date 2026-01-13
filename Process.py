import pandas as pd
import numpy as np
import re
import folium
from folium import Element
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ============================
# 1) CARGA Y FILTRADO ESTRICTO
# ============================
print("📖 Cargando y filtrando datos...")
try:
    df1 = pd.read_excel('./assets/1era-quincena-febrero-catastro-nacional-2025-banecuador.xlsx')
    df2 = pd.read_excel('./assets/mdi_personasdesaparecidas_pm_2025_enero_octubre.xlsx')
except FileNotFoundError as e:
    print(f"❌ Error: No se encontró el archivo {e.filename}")
    exit()

# --- FILTRO DE CLASIFICACIÓN ---
# Convertimos a mayúsculas y quitamos espacios para evitar errores de digitación
df1['Clasificación'] = df1['Clasificación'].astype(str).str.upper().str.strip()
tipos_permitidos = ['HOTEL', 'HOSTAL', 'HOSTERÍA']
df1 = df1[df1['Clasificación'].isin(tipos_permitidos)]

print(f"✅ Filtro aplicado: Solo se procesarán {len(df1)} registros de tipo Hotel, Hostal u Hostería.")

# Estandarizar nombres para la unión
df1 = df1.rename(columns={'Provincia':'PROVINCIA','Cantón':'CANTON','Parroquia':'PARROQUIA'})
df2 = df2.rename(columns={'Provincia':'PROVINCIA','Cantón':'CANTON','Parroquía':'PARROQUIA','latitud':'Latitud','longitud':'Longitud'})

# Limpiar coordenadas
def limpiar_coord(v):
    try: return float(str(v).replace(',', '.'))
    except: return 0.0

df2['Latitud'] = df2['Latitud'].apply(limpiar_coord)
df2['Longitud'] = df2['Longitud'].apply(limpiar_coord)

# Unión de datos (Inner join para asegurar que tengan datos de riesgo)
df = pd.merge(df1.dropna(subset=['PROVINCIA','CANTON','PARROQUIA']), 
              df2, on=['PROVINCIA','CANTON','PARROQUIA'], how='inner')

# ============================
# 2) MACHINE LEARNING (100%)
# ============================
print("🧠 Entrenando modelo con zonas de riesgo...")
df['RIESGO_NUM'] = df.groupby(['PROVINCIA','CANTON','PARROQUIA'])['PARROQUIA'].transform('count')
df['NIVEL_REAL'] = df['RIESGO_NUM'].apply(lambda x: 'ALTO' if x >= 10 else ('MEDIO' if x >= 5 else 'BAJO'))

le_p, le_c = LabelEncoder(), LabelEncoder()
df['P_CODE'] = le_p.fit_transform(df['PROVINCIA'].astype(str))
df['C_CODE'] = le_c.fit_transform(df['CANTON'].astype(str))

X = df[['P_CODE', 'C_CODE', 'Latitud', 'Longitud']]
y = df['NIVEL_REAL']

modelo = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
df['NIVEL_PREDICHO'] = modelo.predict(X)

# ============================
# 3) AJUSTE DE ESTRELLAS (REFINADO)
# ============================
def extraer_y_ajustar(valor_cat, nivel):
    # Buscamos solo los números (ej: "3" de "3 Estrellas" o "3 ESTRELLAS")
    numeros = re.findall(r'\d+', str(valor_cat))
    est = float(numeros[0]) if numeros else 0.0
    
    # Aplicar penalización según nivel de riesgo
    if nivel == 'ALTO': 
        return max(est - 1, 0)
    elif nivel == 'MEDIO': 
        return max(est - 0.5, 0)
    return est

df['ESTRELLAS_AJUSTADAS'] = df.apply(lambda x: extraer_y_ajustar(x['Categoría'], x['NIVEL_PREDICHO']), axis=1)

# ============================
# 4) MAPA FINAL
# ============================
print("🗺️  Generando mapa...")
m = folium.Map(location=[df['Latitud'].iloc[0], df['Longitud'].iloc[0]], zoom_start=7)

# Leyenda y Colores
colores = {'ALTO': 'red', 'MEDIO': 'orange', 'BAJO': 'green'}
leyenda_html = '''
     <div style="position: fixed; bottom: 50px; left: 50px; width: 180px; height: 110px; 
     background-color: white; border:2px solid grey; z-index:9999; font-size:12px; padding: 10px;">
     <b>Filtro: Hoteles/Hostales</b><br>
     <i style="background:red;width:10px;height:10px;display:inline-block"></i> Riesgo ALTO (-1⭐)<br>
     <i style="background:orange;width:10px;height:10px;display:inline-block"></i> Riesgo MEDIO (-0.5⭐)<br>
     <i style="background:green;width:10px;height:10px;display:inline-block"></i> Riesgo BAJO (0)
     </div>
     '''
m.get_root().html.add_child(Element(leyenda_html))

for _, r in df.head(3000).iterrows():
    popup_txt = f"""
    <b>{r['Nombre Comercial']}</b><br>
    Tipo: {r['Clasificación']}<br>
    Riesgo Predicho: {r['NIVEL_PREDICHO']}<br>
    Estrellas Finales: {r['ESTRELLAS_AJUSTADAS']} ⭐
    """
    folium.CircleMarker(
        location=[r['Latitud'], r['Longitud']],
        radius=5,
        color=colores[r['NIVEL_PREDICHO']],
        fill=True,
        fill_opacity=0.7,
        popup=folium.Popup(popup_txt, max_width=300)
    ).add_to(m)

m.save('mapa_final_filtrado.html')
print("✅ ¡Listo! Abre 'mapa_final_filtrado.html' para ver los resultados filtrados.")