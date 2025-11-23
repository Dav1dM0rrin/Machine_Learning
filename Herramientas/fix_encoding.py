"""
Script para detectar y corregir problemas de codificación en CSV
Proyecto: Minería de Datos 2025-2
"""

import pandas as pd
import chardet

def detectar_codificacion(archivo="transporte_limpio.csv"):
    """
    Detecta automáticamente la codificación del archivo
    """
    print("\n" + "="*70)
    print("🔍 DETECTANDO CODIFICACIÓN DEL ARCHIVO")
    print("="*70)
    
    # Leer una muestra del archivo
    with open(archivo, 'rb') as f:
        raw_data = f.read(100000)  # Leer primeros 100KB
    
    # Detectar codificación
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    confidence = result['confidence']
    
    print(f"\n📊 Resultado del análisis:")
    print(f"   Codificación detectada: {encoding}")
    print(f"   Confianza: {confidence * 100:.2f}%")
    
    return encoding

def cargar_csv_con_encoding(archivo, encoding=None):
    """
    Carga el CSV probando diferentes codificaciones
    """
    print("\n" + "="*70)
    print("📂 INTENTANDO CARGAR EL CSV")
    print("="*70)
    
    # Lista de codificaciones a probar
    encodings_to_try = [
        encoding,           # La detectada
        'utf-8',
        'latin-1',
        'iso-8859-1',
        'cp1252',           # Windows
        'utf-16',
        'utf-8-sig'         # UTF-8 con BOM
    ]
    
    # Eliminar None de la lista
    encodings_to_try = [e for e in encodings_to_try if e is not None]
    
    for enc in encodings_to_try:
        try:
            print(f"\n🔄 Intentando con: {enc}...")
            df = pd.read_csv(archivo, encoding=enc)
            print(f"✅ ¡Éxito con {enc}!")
            print(f"   Registros: {len(df)}")
            print(f"   Columnas: {len(df.columns)}")
            return df, enc
        except UnicodeDecodeError as e:
            print(f"❌ Falló: {e}")
        except Exception as e:
            print(f"⚠️  Error: {e}")
    
    print("\n❌ No se pudo leer el archivo con ninguna codificación")
    return None, None

def convertir_a_utf8(archivo_entrada, archivo_salida='transporte_limpio_utf8.csv'):
    """
    Convierte el CSV a UTF-8 limpio
    """
    print("\n" + "="*70)
    print("🔄 CONVIRTIENDO A UTF-8")
    print("="*70)
    
    # Detectar codificación
    encoding_original = detectar_codificacion(archivo_entrada)
    
    # Cargar con la codificación correcta
    df, encoding_usado = cargar_csv_con_encoding(archivo_entrada, encoding_original)
    
    if df is None:
        print("❌ No se pudo procesar el archivo")
        return False
    
    # Mostrar información del dataframe
    print(f"\n📊 Información del DataFrame:")
    print(f"   Shape: {df.shape}")
    print(f"   Columnas: {list(df.columns)}")
    
    # Verificar si hay problemas con los datos
    print(f"\n🔍 Verificando calidad de datos:")
    print(f"   Valores nulos: {df.isnull().sum().sum()}")
    print(f"   Duplicados: {df.duplicated().sum()}")
    
    # Mostrar primeras filas
    print(f"\n📋 Primeras 3 filas:")
    print(df.head(3))
    
    # Guardar en UTF-8
    try:
        df.to_csv(archivo_salida, index=False, encoding='utf-8')
        print(f"\n✅ Archivo convertido exitosamente!")
        print(f"   Guardado como: {archivo_salida}")
        print(f"   Codificación: UTF-8")
        return True
    except Exception as e:
        print(f"❌ Error al guardar: {e}")
        return False

def limpiar_csv(archivo_entrada, archivo_salida='transporte_limpio_final.csv'):
    """
    Carga, limpia y guarda el CSV en UTF-8
    """
    print("\n" + "="*70)
    print("🧹 LIMPIEZA COMPLETA DEL CSV")
    print("="*70)
    
    # Detectar y cargar
    encoding = detectar_codificacion(archivo_entrada)
    df, enc_usado = cargar_csv_con_encoding(archivo_entrada, encoding)
    
    if df is None:
        return False
    
    print(f"\n🔧 Aplicando limpieza...")
    
    # Limpiar nombres de columnas (quitar espacios extra)
    df.columns = df.columns.str.strip()
    
    # Eliminar filas completamente vacías
    df = df.dropna(how='all')
    
    # Eliminar duplicados
    registros_antes = len(df)
    df = df.drop_duplicates()
    registros_despues = len(df)
    duplicados_eliminados = registros_antes - registros_despues
    
    print(f"   ✅ Columnas limpiadas")
    print(f"   ✅ Filas vacías eliminadas")
    print(f"   ✅ Duplicados eliminados: {duplicados_eliminados}")
    
    # Mostrar resumen
    print(f"\n📊 Resumen del DataFrame limpio:")
    print(f"   Registros finales: {len(df)}")
    print(f"   Columnas: {len(df.columns)}")
    print(df.info())
    
    # Guardar
    try:
        df.to_csv(archivo_salida, index=False, encoding='utf-8')
        print(f"\n✅ ¡Archivo limpio guardado!")
        print(f"   Nombre: {archivo_salida}")
        print(f"   Codificación: UTF-8")
        print(f"   Listo para usar en Machine Learning")
        return True
    except Exception as e:
        print(f"❌ Error al guardar: {e}")
        return False

def diagnostico_completo(archivo):
    """
    Diagnóstico completo del archivo CSV
    """
    print("\n" + "🔬"*35)
    print("   DIAGNÓSTICO COMPLETO DEL CSV")
    print("🔬"*35)
    
    # 1. Detectar codificación
    encoding = detectar_codificacion(archivo)
    
    # 2. Intentar cargar
    df, enc_usado = cargar_csv_con_encoding(archivo, encoding)
    
    if df is None:
        print("\n❌ No se pudo realizar el diagnóstico completo")
        return
    
    # 3. Análisis de columnas
    print("\n" + "="*70)
    print("📊 ANÁLISIS DE COLUMNAS")
    print("="*70)
    
    for col in df.columns:
        print(f"\n🔹 {col}")
        print(f"   Tipo: {df[col].dtype}")
        print(f"   Nulos: {df[col].isnull().sum()} ({df[col].isnull().sum()/len(df)*100:.2f}%)")
        print(f"   Únicos: {df[col].nunique()}")
        
        # Mostrar valores únicos si son pocos
        if df[col].nunique() < 10:
            print(f"   Valores: {list(df[col].unique())[:5]}")
    
    # 4. Verificar requisitos del proyecto
    print("\n" + "="*70)
    print("✅ VERIFICACIÓN DE REQUISITOS DEL PROYECTO")
    print("="*70)
    
    print(f"\n📋 Requisito: Mínimo 1000 registros")
    if len(df) >= 1000:
        print(f"   ✅ Cumple: {len(df)} registros")
    else:
        print(f"   ❌ No cumple: Solo {len(df)} registros")
    
    # Contar columnas cualitativas y cuantitativas
    cualitativos = df.select_dtypes(include=['object']).columns.tolist()
    cuantitativos = df.select_dtypes(include=['number']).columns.tolist()
    
    print(f"\n📋 Requisito: Mínimo 3 campos cualitativos")
    print(f"   Encontrados: {len(cualitativos)}")
    print(f"   Columnas: {cualitativos}")
    if len(cualitativos) >= 3:
        print(f"   ✅ Cumple")
    else:
        print(f"   ❌ No cumple")
    
    print(f"\n📋 Requisito: Mínimo 3 campos cuantitativos")
    print(f"   Encontrados: {len(cuantitativos)}")
    print(f"   Columnas: {cuantitativos[:10]}")  # Mostrar primeros 10
    if len(cuantitativos) >= 3:
        print(f"   ✅ Cumple")
    else:
        print(f"   ❌ No cumple")
    
    print("\n" + "="*70)

# ============================================
# MENÚ INTERACTIVO
# ============================================
def menu_principal():
    """
    Menú interactivo para elegir acción
    """
    print("\n" + "🚀"*35)
    print("   CORRECTOR DE CODIFICACIÓN CSV")
    print("   Proyecto: Minería de Datos 2025-2")
    print("🚀"*35)
    
    archivo = input("\n📂 Ingresa el nombre del archivo CSV: ").strip()
    
    print("\n🔧 Opciones disponibles:")
    print("   1. Solo detectar codificación")
    print("   2. Convertir a UTF-8")
    print("   3. Limpieza completa + UTF-8")
    print("   4. Diagnóstico completo")
    
    opcion = input("\n▶️  Selecciona una opción (1-4): ").strip()
    
    if opcion == "1":
        detectar_codificacion(archivo)
    
    elif opcion == "2":
        convertir_a_utf8(archivo)
    
    elif opcion == "3":
        limpiar_csv(archivo)
    
    elif opcion == "4":
        diagnostico_completo(archivo)
    
    else:
        print("❌ Opción inválida")

# ============================================
# EJECUCIÓN
# ============================================
if __name__ == "__main__":
    # Intenta instalar chardet si no está disponible
    try:
        import chardet
    except ImportError:
        print("📦 Instalando chardet...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'chardet'])
        import chardet
    
    menu_principal()