import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree

# ── CARGAR DATOS ──────────────────────────────────────────────
df_base = pd.read_csv('Worldwide Travel Cities Dataset.csv')
df_uccn = pd.read_csv('Creative Cities UNESCO 2025.csv')
df_wh   = pd.read_csv('WHC UNESCO Sites 2025.csv')

print(f"Dataset base: {len(df_base)} ciudades")
print(f"UNESCO Ciudades Creativas: {len(df_uccn)} entradas")
print(f"UNESCO World Heritage Sites: {len(df_wh)} sitios")

# ── JOIN 1: UNESCO CIUDADES CREATIVAS ─────────────────────────
# Normalizar nombres para el match
df_base['city_norm'] = df_base['city'].str.strip().str.lower()
df_uccn['city_norm'] = df_uccn['Título EN'].str.strip().str.lower()

df_uccn_clean = df_uccn[['city_norm', 'Términos', 'Fecha']].copy()
df_uccn_clean.columns = ['city_norm', 'creative_city_category', 'creative_city_year']

# Join por nombre de ciudad
df_merged = df_base.merge(df_uccn_clean, on='city_norm', how='left')

# Desambiguar Granada: España (Literatura, 2014) vs Nicaragua (Diseño, 2023)
spain_mask     = (df_merged['city_norm'] == 'granada') & (df_merged['country'] == 'Spain')     & (df_merged['creative_city_year'] == 2014)
nicaragua_mask = (df_merged['city_norm'] == 'granada') & (df_merged['country'] == 'Nicaragua') & (df_merged['creative_city_year'] == 2023)
other_mask     = df_merged['city_norm'] != 'granada'
df_merged = df_merged[other_mask | spain_mask | nicaragua_mask].copy()

print(f"\nJOIN 1 completado: {len(df_merged)} filas | {df_merged['creative_city_category'].notna().sum()} ciudades con categoría creativa UNESCO")

# ── JOIN 2: UNESCO WORLD HERITAGE SITES ───────────────────────
# Eliminar sitios sin coordenadas
df_wh = df_wh.dropna(subset=['latitude', 'longitude']).copy()

# BallTree con distancia Haversine (coordenadas esféricas)
cities_coords = np.radians(df_merged[['latitude', 'longitude']].values)
wh_coords     = np.radians(df_wh[['latitude', 'longitude']].values)

tree = BallTree(cities_coords, metric='haversine')
distances, indices = tree.query(wh_coords, k=1)
distances_km = distances.flatten() * 6371  # Radio terrestre en km

df_wh['city_idx']    = indices.flatten()
df_wh['distance_km'] = distances_km

# Umbral: sitios a menos de 50km de la ciudad
UMBRAL_KM = 50
df_wh_cerca = df_wh[df_wh['distance_km'] <= UMBRAL_KM].copy()

# Agregar por ciudad: número de sitios + nombres concatenados con " | "
agg = df_wh_cerca.groupby('city_idx').agg(
    unesco_wh_count=('name_en', 'count'),
    unesco_wh_names=('name_en', lambda x: ' | '.join(x))
).reset_index()

# Unir al dataset principal
df_merged = df_merged.reset_index(drop=True)
df_merged['city_idx'] = df_merged.index
df_merged = df_merged.merge(agg, on='city_idx', how='left')

# Rellenar ciudades sin sitios cercanos
df_merged['unesco_wh_count'] = df_merged['unesco_wh_count'].fillna(0).astype(int)
df_merged['unesco_wh_names'] = df_merged['unesco_wh_names'].fillna('')

print(f"JOIN 2 completado: {(df_merged['unesco_wh_count'] > 0).sum()} ciudades con sitios World Heritage en {UMBRAL_KM}km")

# ── EXPORTAR ──────────────────────────────────────────────────
df_final = df_merged.drop(columns=['city_norm', 'city_idx'])

output_path = 'travel_cities_enriched.csv'
df_final.to_csv(output_path, index=False)

print(f"\n✅ CSV exportado: {output_path}")
print(f"   Filas: {len(df_final)} | Columnas: {len(df_final.columns)}")
print(f"   Columnas: {df_final.columns.tolist()}")
