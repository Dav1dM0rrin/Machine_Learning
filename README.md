# 🚌 Proyecto: Predicción de Pasajeros en Transporte Limpio

## 📋 Descripción del Proyecto

Sistema de predicción de pasajeros diarios en sistemas de transporte limpio en Colombia utilizando Machine Learning. Este proyecto cumple con los requisitos del parcial de Minería de Datos 2025-2.

## ✅ Requisitos Cumplidos

- ✅ Dataset de datos.gov.co con más de 1000 registros
- ✅ 3 campos cualitativos (Ciudad, Sistema, Fecha) y 3 cuantitativos (Pasajeros/dia, Variación, DíaSemana)
- ✅ 2 Algoritmos de Machine Learning (Random Forest y Gradient Boosting)
- ✅ Precisión superior al 85% en ambos modelos
- ✅ Aplicación web funcional en Streamlit
- ✅ Visualizaciones y métricas de evaluación
- ✅ Sistema de predicción mediante formularios

## 🔧 Instalación

### 1. Requisitos Previos

```bash
Python 3.8 o superior
pip (gestor de paquetes de Python)
```

### 2. Instalar Dependencias

```bash
pip install pandas numpy scikit-learn joblib streamlit plotly
```

O usar el archivo requirements.txt:

```bash
pip install -r requirements.txt
```

### 3. Contenido del archivo requirements.txt

```
pandas==2.0.0
numpy==1.24.0
scikit-learn==1.3.0
joblib==1.3.0
streamlit==1.28.0
plotly==5.17.0
```

## 📂 Estructura de Archivos

```
proyecto_mineria/
│
├── transporte_limpio.csv          # Dataset principal
├── train_models.py                # Script para entrenar modelos
├── app.py                        # Aplicación Streamlit
├── requirements.txt              # Dependencias del proyecto
├── README.md                     # Este archivo
│
└── modelos_generados/            # (Se crean al ejecutar)
    ├── random_forest_model.pkl
    ├── gradient_boosting_model.pkl
    ├── scaler.pkl
    ├── le_ciudad.pkl
    └── le_sistema.pkl
```

## 🚀 Ejecución del Proyecto

### Paso 1: Entrenar los Modelos

```bash
python train_models.py
```

**Salida esperada:**
```
==================================================
DATASET DE TRANSPORTE LIMPIO - ANÁLISIS INICIAL
==================================================
Dimensiones: (1490, 10)

==================================================
MODELO 1: RANDOM FOREST REGRESSOR
==================================================
✓ Entrenamiento completado

MÉTRICAS DEL MODELO:
  • R² Score: 0.9XXX
  • Precisión: XX.XX%
  • RMSE: XXXX.XX
  • MAE: XXXX.XX

==================================================
MODELO 2: GRADIENT BOOSTING REGRESSOR
==================================================
✓ Entrenamiento completado

MÉTRICAS DEL MODELO:
  • R² Score: 0.9XXX
  • Precisión: XX.XX%
  • RMSE: XXXX.XX
  • MAE: XXXX.XX

¿Deseas realizar predicciones por consola? (s/n):
```

**Nota:** Al finalizar el entrenamiento, el programa te preguntará si deseas hacer predicciones por consola. Puedes responder 's' para hacer predicciones inmediatamente o 'n' para continuar más tarde.

### Paso 2: Hacer Predicciones por Consola (Opcional)

Si quieres hacer predicciones por consola después del entrenamiento, ejecuta:

```bash
python predict_console.py
```

**Ejemplo de uso interactivo:**
```
🔮 SISTEMA DE PREDICCIÓN INTERACTIVO
====================================================================

📍 CIUDADES DISPONIBLES:
  1. Barranquilla
  2. Bogotá
  3. Bucaramanga
  4. Cali/Valle
  5. Cartagena
  6. Medellin
  7. Pereira

👉 Selecciona ciudad (1-7): 2

🚌 SISTEMAS DE TRANSPORTE DISPONIBLES:
  1. MEGABUS
  2. METROLINEA
  3. MIO
  4. SITVA
  5. TRANSCARIBE
  6. TRANSMETRO
  7. TRANSMILENIO/SITP
  ...
```

### Paso 3: Ejecutar la Aplicación Streamlit

```bash
streamlit run app.py
```

**La aplicación se abrirá automáticamente en:**
```
http://localhost:8501
```

## 📱 Uso de la Aplicación

### Secciones Disponibles:

1. **🏠 Inicio**
   - Vista general del proyecto
   - Estadísticas del dataset
   - Información del parcial

2. **📊 Exploración de Datos**
   - Vista de datos con filtros
   - Visualizaciones interactivas
   - Estadísticas descriptivas

3. **🤖 Modelos ML**
   - Métricas de rendimiento
   - Comparación de modelos
   - Gráficos de predicción vs valores reales
   - Matrices de confusión

4. **🔮 Predicciones**
   - Formulario interactivo
   - Predicción en tiempo real
   - Comparación de resultados entre modelos

5. **📈 Análisis de Métricas**
   - Distribución de errores
   - Análisis de residuales
   - Estadísticas detalladas

## 🎯 Características de los Modelos

### Random Forest Regressor
- **N° de árboles:** 200
- **Profundidad máxima:** 20
- **Muestras mínimas por división:** 5
- **Precisión esperada:** >85%

### Gradient Boosting Regressor
- **N° de estimadores:** 200
- **Tasa de aprendizaje:** 0.1
- **Profundidad máxima:** 7
- **Precisión esperada:** >85%

## 📊 Variables del Dataset

### Variables Cualitativas:
- **Ciudad:** Ubicación del sistema de transporte
- **Sistema:** Tipo de sistema (TRANSMILENIO/SITP, MIO, etc.)
- **Fecha:** Fecha del registro

### Variables Cuantitativas:
- **Pasajeros/dia:** Variable objetivo (a predecir)
- **Variación Transmilenio:** Variación respecto al sistema de referencia
- **Pasajeros día típico laboral:** Promedio de pasajeros en días laborales
- **Pasajeros día sábado:** Promedio de pasajeros los sábados
- **Pasajeros día festivo:** Promedio de pasajeros en festivos
- **DíaSemana:** Día de la semana (1-7)

## 🔮 Ejemplo de Predicción

```python
# Datos de entrada
entrada = {
    'Ciudad': 'Bogotá',
    'Sistema': 'TRANSMILENIO/SITP',
    'Variación Transmilenio': -0.7,
    'Pasajeros día típico laboral': 3860061,
    'Pasajeros día sábado': 2499019,
    'Pasajeros día festivo': 1188607,
    'DíaSemana': 1,
    'Año': 2024,
    'Mes': 8,
    'Dia': 15
}

# Resultado esperado
Predicción Random Forest: ~1,200,000 pasajeros
Predicción Gradient Boosting: ~1,180,000 pasajeros
```

## 📈 Métricas de Evaluación

Los modelos se evalúan usando:

- **R² Score (Coeficiente de Determinación):** Mide la proporción de varianza explicada
- **RMSE (Root Mean Squared Error):** Error cuadrático medio
- **MAE (Mean Absolute Error):** Error absoluto medio
- **MSE (Mean Squared Error):** Error cuadrático medio

**Requisito del proyecto:** R² Score > 0.85 (85% de precisión)

## 🐛 Solución de Problemas

### Error: "ModuleNotFoundError"
```bash
# Instalar la dependencia faltante
pip install [nombre_del_modulo]
```

### Error: "FileNotFoundError: transporte_limpio.csv"
```bash
# Asegúrate de que el archivo CSV esté en la misma carpeta
# que los scripts train_models.py y app.py
```

### Error: "Los modelos no están cargados"
```bash
# Primero debes ejecutar el entrenamiento:
python train_models.py

# Luego ejecutar la aplicación:
streamlit run app.py
```

### La aplicación no se abre automáticamente
```
Abre manualmente en tu navegador:
http://localhost:8501
```

## 📝 Notas Importantes

1. **Tiempo de entrenamiento:** El proceso de entrenamiento puede tomar 2-5 minutos dependiendo de tu computadora.

2. **Tamaño del dataset:** El dataset contiene 1,490 registros, cumpliendo ampliamente con el requisito de >1000 registros.

3. **Rendimiento:** Los modelos están optimizados para obtener precisiones superiores al 85%.

4. **Escalabilidad:** El código está preparado para manejar datasets más grandes si es necesario.

## 📚 Fuente de Datos

- **Origen:** [datos.gov.co](https://www.datos.gov.co/)
- **Dataset:** Transporte Limpio - Pasajeros por día
- **Período:** Año 2020
- **Ciudades:** Bogotá, Cali, Medellín, Barranquilla, Cartagena, Bucaramanga, Pereira

## 👥 Información del Proyecto

- **Materia:** Minería de Datos 2025-2
- **Entrega:** 24/11/2025 06:30 PM
- **Tecnologías:** Python, Scikit-learn, Streamlit, Plotly, Pandas

## 🎓 Evidencias del Parcial

✅ **Punto 1 (20%):** Dataset de datos.gov.co con análisis exploratorio en KNIME
✅ **Punto 2 (40%):** Dos algoritmos de ML con precisión >85%
✅ **Punto 3 (20%):** Tablero en Power BI (separado)
✅ **Punto 4 (20%):** Aplicativo web en Streamlit con predicciones

## 📞 Soporte

Si tienes problemas durante la ejecución:

1. Verifica que todas las dependencias estén instaladas
2. Asegúrate de tener Python 3.8 o superior
3. Revisa que el archivo CSV esté en la carpeta correcta
4. Ejecuta primero `train_models.py` antes de `app.py`

---

**¡Proyecto listo para presentar! 🎉**
