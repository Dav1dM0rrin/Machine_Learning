"""
Prepara datos optimizados para Power BI
Incluye métricas calculadas y categorías
"""

import pandas as pd
import numpy as np

def preparar_datos_powerbi(csv_entrada='transporte_limpio_final.csv', 
                           csv_salida='datos_powerbi.csv'):
    """
    Prepara CSV con todas las métricas para Power BI
    """
    print("\n" + "="*70)
    print("📊 PREPARANDO DATOS PARA POWER BI")
    print("="*70)
    
    # Cargar datos
    try:
        df = pd.read_csv(csv_entrada, encoding='utf-8')
        print(f"✅ Datos cargados: {len(df)} registros")
    except:
        # Intentar con otras codificaciones
        for encoding in ['latin-1', 'iso-8859-1', 'cp1252']:
            try:
                df = pd.read_csv(csv_entrada, encoding=encoding)
                print(f"✅ Datos cargados con {encoding}: {len(df)} registros")
                break
            except:
                continue
    
    # Convertir fecha
    df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d/%m/%Y', errors='coerce')
    
    # ============================================
    # CREAR COLUMNAS ADICIONALES
    # ============================================
    print("\n🔧 Creando columnas calculadas...")
    
    # Temporales
    df['Año'] = df['Fecha'].dt.year
    df['Mes'] = df['Fecha'].dt.month
    df['MesNombre'] = df['Fecha'].dt.month_name()
    df['Trimestre'] = df['Fecha'].dt.quarter
    df['Semana'] = df['Fecha'].dt.isocalendar().week
    df['DiaMes'] = df['Fecha'].dt.day
    
    # Día de la semana
    dia_nombres = {1: 'Lunes', 2: 'Martes', 3: 'Miércoles', 4: 'Jueves', 
                   5: 'Viernes', 6: 'Sábado', 7: 'Domingo'}
    df['DiaNombre'] = df['DiaSemana'].map(dia_nombres)
    
    # Tipo de día
    df['TipoDia'] = df['DiaSemana'].apply(
        lambda x: 'Fin de Semana' if x in [6, 7] else 'Entre Semana'
    )
    
    # Categoría de demanda (basada en percentiles)
    percentil_33 = df['Pasajeros/dia'].quantile(0.33)
    percentil_66 = df['Pasajeros/dia'].quantile(0.66)
    
    def categorizar_demanda(pasajeros):
        if pd.isna(pasajeros):
            return 'Sin Datos'
        elif pasajeros < percentil_33:
            return 'Baja'
        elif pasajeros < percentil_66:
            return 'Media'
        else:
            return 'Alta'
    
    df['CategoriaDemanda'] = df['Pasajeros/dia'].apply(categorizar_demanda)
    
    # Ocupación (porcentaje respecto al día típico)
    df['PorcentajeOcupacion'] = (df['Pasajeros/dia'] / df['Pasajeros dia tipico laboral'] * 100).round(2)
    df['PorcentajeOcupacion'].fillna(0, inplace=True)
    
    # Variación porcentual
    df['VariacionPorcentual'] = (df['Variacion Transmilenio'] * 100).round(2)
    df['VariacionPorcentual'].fillna(0, inplace=True)
    
    # Temporada COVID
    def obtener_temporada(row):
        if pd.isna(row['Fecha']):
            return 'Sin Datos'
        mes = row['Mes']
        año = row['Año']
        
        if año == 2020:
            if mes in [3, 4, 5]:
                return 'Cuarentena Estricta'
            elif mes in [6, 7, 8]:
                return 'Reapertura Gradual'
            else:
                return 'Restricciones Moderadas'
        else:
            return 'Otros'
    
    df['TemporadaCOVID'] = df.apply(obtener_temporada, axis=1)
    
    # Diferencia con día típico
    df['DiferenciaPasajeros'] = df['Pasajeros/dia'] - df['Pasajeros dia tipico laboral']
    df['DiferenciaPasajeros'].fillna(0, inplace=True)
    
    # Clasificación de sistema por tamaño
    def clasificar_sistema(row):
        capacidad = row['Pasajeros dia tipico laboral']
        if pd.isna(capacidad):
            return 'Sin Datos'
        elif capacidad > 1000000:
            return 'Gran Sistema'
        elif capacidad > 200000:
            return 'Sistema Mediano'
        else:
            return 'Sistema Pequeño'
    
    df['TamañoSistema'] = df.apply(clasificar_sistema, axis=1)
    
    # Promedio móvil 7 días (para tendencias)
    df = df.sort_values(['Ciudad', 'Sistema', 'Fecha'])
    df['PromedioMovil7Dias'] = df.groupby(['Ciudad', 'Sistema'])['Pasajeros/dia'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    ).round(0)
    
    # Ranking por ciudad
    df['RankingCiudad'] = df.groupby('Fecha')['Pasajeros/dia'].rank(ascending=False, method='dense')
    
    print("✅ Columnas calculadas creadas")
    
    # ============================================
    # CREAR MÉTRICAS AGREGADAS
    # ============================================
    print("\n📊 Calculando métricas agregadas...")
    
    # Total pasajeros por sistema
    total_pasajeros = df.groupby('Sistema')['Pasajeros/dia'].sum().reset_index()
    total_pasajeros.columns = ['Sistema', 'TotalPasajeros']
    df = df.merge(total_pasajeros, on='Sistema', how='left')
    
    # Promedio por ciudad
    promedio_ciudad = df.groupby('Ciudad')['Pasajeros/dia'].mean().reset_index()
    promedio_ciudad.columns = ['Ciudad', 'PromedioCiudad']
    df = df.merge(promedio_ciudad, on='Ciudad', how='left')
    
    print("✅ Métricas agregadas calculadas")
    
    # ============================================
    # LIMPIAR Y ORDENAR
    # ============================================
    
    # Eliminar columnas innecesarias para Power BI
    columnas_a_eliminar = ['id']
    df = df.drop(columns=[col for col in columnas_a_eliminar if col in df.columns])
    
    # Ordenar por fecha
    df = df.sort_values(['Fecha', 'Ciudad', 'Sistema'])
    
    # Reemplazar NaN con valores apropiados
    df.fillna({
        'Pasajeros/dia': 0,
        'VariacionPorcentual': 0,
        'PorcentajeOcupacion': 0
    }, inplace=True)
    
    # ============================================
    # GUARDAR
    # ============================================
    
    print(f"\n💾 Guardando archivo para Power BI...")
    df.to_csv(csv_salida, index=False, encoding='utf-8-sig')  # UTF-8 con BOM para Excel/Power BI
    
    print(f"✅ Archivo guardado: {csv_salida}")
    print(f"   Registros: {len(df)}")
    print(f"   Columnas: {len(df.columns)}")
    
    # ============================================
    # RESUMEN DE COLUMNAS
    # ============================================
    
    print("\n" + "="*70)
    print("📋 COLUMNAS DISPONIBLES EN POWER BI")
    print("="*70)
    
    print("\n🔹 DIMENSIONES (Para filtros y segmentación):")
    dimensiones = ['Ciudad', 'Sistema', 'DiaNombre', 'TipoDia', 'MesNombre', 
                   'Año', 'Trimestre', 'CategoriaDemanda', 'TemporadaCOVID', 'TamañoSistema']
    for dim in dimensiones:
        if dim in df.columns:
            print(f"   • {dim}: {df[dim].nunique()} valores únicos")
    
    print("\n🔹 MÉTRICAS (Para KPIs y gráficos):")
    metricas = ['Pasajeros/dia', 'PorcentajeOcupacion', 'VariacionPorcentual', 
                'DiferenciaPasajeros', 'PromedioMovil7Dias', 'TotalPasajeros', 'PromedioCiudad']
    for metrica in metricas:
        if metrica in df.columns:
            print(f"   • {metrica}")
    
    print("\n🔹 DATOS TEMPORALES:")
    temporales = ['Fecha', 'Año', 'Mes', 'MesNombre', 'Trimestre', 'Semana', 'DiaMes']
    for temp in temporales:
        if temp in df.columns:
            print(f"   • {temp}")
    
    # ============================================
    # ESTADÍSTICAS GENERALES
    # ============================================
    
    print("\n" + "="*70)
    print("📊 ESTADÍSTICAS PARA DASHBOARD")
    print("="*70)
    
    print(f"\n📈 PASAJEROS:")
    print(f"   Total: {df['Pasajeros/dia'].sum():,.0f}")
    print(f"   Promedio diario: {df['Pasajeros/dia'].mean():,.0f}")
    print(f"   Máximo: {df['Pasajeros/dia'].max():,.0f}")
    print(f"   Mínimo: {df['Pasajeros/dia'].min():,.0f}")
    
    print(f"\n🏙️ CIUDADES:")
    for ciudad in df['Ciudad'].unique():
        total = df[df['Ciudad'] == ciudad]['Pasajeros/dia'].sum()
        print(f"   {ciudad}: {total:,.0f} pasajeros totales")
    
    print(f"\n🚌 SISTEMAS:")
    for sistema in df['Sistema'].unique()[:5]:  # Top 5
        total = df[df['Sistema'] == sistema]['Pasajeros/dia'].sum()
        print(f"   {sistema}: {total:,.0f} pasajeros totales")
    
    print(f"\n📅 RANGO DE FECHAS:")
    print(f"   Desde: {df['Fecha'].min().strftime('%d/%m/%Y')}")
    print(f"   Hasta: {df['Fecha'].max().strftime('%d/%m/%Y')}")
    print(f"   Días: {(df['Fecha'].max() - df['Fecha'].min()).days}")
    
    print("\n✅ Datos listos para importar en Power BI")
    print("="*70)
    
    return df

# ============================================
# CREAR TABLA DE RESUMEN PARA POWER BI
# ============================================
def crear_tabla_resumen(df, csv_salida='resumen_sistemas.csv'):
    """
    Crea tabla resumen por sistema para KPIs
    """
    print("\n📊 Creando tabla resumen...")
    
    resumen = df.groupby(['Ciudad', 'Sistema']).agg({
        'Pasajeros/dia': ['sum', 'mean', 'max', 'min', 'count'],
        'PorcentajeOcupacion': 'mean',
        'VariacionPorcentual': 'mean',
        'Pasajeros dia tipico laboral': 'first'
    }).round(2)
    
    resumen.columns = ['_'.join(col).strip() for col in resumen.columns.values]
    resumen = resumen.reset_index()
    
    # Renombrar columnas
    resumen.columns = ['Ciudad', 'Sistema', 'TotalPasajeros', 'PromedioPasajeros', 
                      'MaximoPasajeros', 'MinimoPasajeros', 'DiasMedidos',
                      'PromedioOcupacion', 'PromedioVariacion', 'CapacidadNormal']
    
    resumen.to_csv(csv_salida, index=False, encoding='utf-8-sig')
    print(f"✅ Tabla resumen guardada: {csv_salida}")
    
    return resumen

# ============================================
# EJECUTAR
# ============================================
if __name__ == "__main__":
    print("\n" + "🚀"*35)
    print("   PREPARACIÓN DE DATOS PARA POWER BI")
    print("   Proyecto: Minería de Datos 2025-2")
    print("🚀"*35)
    
    # Preparar datos principales
    df = preparar_datos_powerbi()
    
    # Crear tabla resumen
    resumen = crear_tabla_resumen(df)
    
    print("\n✅ ¡Proceso completado!")
    print("\n📦 Archivos generados:")
    print("   1. datos_powerbi.csv (datos completos)")
    print("   2. resumen_sistemas.csv (tabla resumen)")
    
    print("\n🎯 Siguiente paso:")
    print("   1. Abrir Power BI Desktop")
    print("   2. Importar 'datos_powerbi.csv'")
    print("   3. Seguir la guía de diseño del dashboard")