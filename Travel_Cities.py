import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import csv

# ── CARGAR DATOS ──────────────────────────────────────────────
df_base = pd.read_csv('Worldwide Travel Cities Dataset.csv')
df_uccn = pd.read_csv('Creative Cities UNESCO 2025.csv')
df_wh   = pd.read_csv('WHC UNESCO Sites 2025.csv')

print(f"Dataset base: {len(df_base)} ciudades")
print(f"UNESCO Ciudades Creativas: {len(df_uccn)} entradas")
print(f"UNESCO World Heritage Sites: {len(df_wh)} sitios")

# ── JOIN 1: UNESCO CIUDADES CREATIVAS ─────────────────────────
df_base['city_norm'] = df_base['city'].str.strip().str.lower()
df_uccn['city_norm'] = df_uccn['Título EN'].str.strip().str.lower()

df_uccn_clean = df_uccn[['city_norm', 'Términos', 'Fecha']].copy()
df_uccn_clean.columns = ['city_norm', 'creative_city_category', 'creative_city_year']

df_merged = df_base.merge(df_uccn_clean, on='city_norm', how='left')

# Desambiguar Granada: España (Literatura, 2014) vs Nicaragua (Diseño, 2023)
spain_mask     = (df_merged['city_norm'] == 'granada') & (df_merged['country'] == 'Spain')     & (df_merged['creative_city_year'] == 2014)
nicaragua_mask = (df_merged['city_norm'] == 'granada') & (df_merged['country'] == 'Nicaragua') & (df_merged['creative_city_year'] == 2023)
other_mask     = df_merged['city_norm'] != 'granada'
df_merged = df_merged[other_mask | spain_mask | nicaragua_mask].copy()

# Renombrar Granadas para distinguirlas en Tableau
df_merged.loc[(df_merged['city'] == 'Granada') & (df_merged['country'] == 'Spain'), 'city'] = 'Granada (Spain)'
df_merged.loc[(df_merged['city'] == 'Granada') & (df_merged['country'] == 'Nicaragua'), 'city'] = 'Granada (Nicaragua)'

print(f"\nJOIN 1 completado: {len(df_merged)} filas | {df_merged['creative_city_category'].notna().sum()} ciudades con categoría creativa UNESCO")

# ── JOIN 2: UNESCO WORLD HERITAGE SITES ───────────────────────
df_wh = df_wh.dropna(subset=['latitude', 'longitude']).copy()

cities_coords = np.radians(df_merged[['latitude', 'longitude']].values)
wh_coords     = np.radians(df_wh[['latitude', 'longitude']].values)

tree = BallTree(cities_coords, metric='haversine')
distances, indices = tree.query(wh_coords, k=1)
distances_km = distances.flatten() * 6371

df_wh['city_idx']    = indices.flatten()
df_wh['distance_km'] = distances_km

UMBRAL_KM = 50
df_wh_cerca = df_wh[df_wh['distance_km'] <= UMBRAL_KM].copy()

agg = df_wh_cerca.groupby('city_idx').agg(
    unesco_wh_count=('name_en', 'count'),
    unesco_wh_names=('name_en', lambda x: ' | '.join(x))
).reset_index()

df_merged = df_merged.reset_index(drop=True)
df_merged['city_idx'] = df_merged.index
df_merged = df_merged.merge(agg, on='city_idx', how='left')

df_merged['unesco_wh_count'] = df_merged['unesco_wh_count'].fillna(0).astype(int)

# Limpiar unesco_wh_names: quitar tags HTML y rellenar nulls con 'None'
df_merged['unesco_wh_names'] = (
    df_merged['unesco_wh_names']
    .str.replace(r'<[^>]+>', '', regex=True)
    .str.strip()
    .fillna('None')
)

print(f"JOIN 2 completado: {(df_merged['unesco_wh_count'] > 0).sum()} ciudades con sitios World Heritage en {UMBRAL_KM}km")

# ── EXPORTAR ──────────────────────────────────────────────────
df_final = df_merged.drop(columns=['city_norm', 'city_idx'])

# Eliminar columnas no utilizadas en Tableau
df_final = df_final.drop(columns=['avg_temp_monthly', 'ideal_durations'], errors='ignore')

# Arreglar año (evitar 2014.0) y rellenar nulls con 0
df_final['creative_city_year'] = df_final['creative_city_year'].fillna(0).astype(int)

# Rellenar nulls de creative_city_category con 'None'
df_final['creative_city_category'] = df_final['creative_city_category'].fillna('None')

# Renombrar seclusion
df_final = df_final.rename(columns={'seclusion': 'remoteness'})

# Latitud y longitud: convertir a enteros x1000000 para evitar problemas de decimales en Tableau
df_final['latitude']  = (df_final['latitude'].astype(float) * 1000000).astype(int)
df_final['longitude'] = (df_final['longitude'].astype(float) * 1000000).astype(int)

output_path = 'Travel_Cities.csv'
df_final.to_csv(output_path, index=False, sep=";", quoting=csv.QUOTE_ALL)

print(f"\n✅ CSV exportado: {output_path}")
print(f"   Filas: {len(df_final)} | Columnas: {len(df_final.columns)}")
print(f"   Columnas: {df_final.columns.tolist()}")