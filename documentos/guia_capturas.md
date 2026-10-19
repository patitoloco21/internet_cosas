# Guía Oficial de Capturas de Pantalla para la Entrega Académica

Este documento detalla **con precisión milimétrica** qué capturas de pantalla debe tomar el estudiante de cada uno de los cuadernos Jupyter (`.ipynb`) abiertos y ejecutados en su entorno de trabajo (VS Code, JupyterLab o Jupyter Notebook).

> **Aviso Importante del Docente:**  
> Las figuras guardadas en las carpetas `evidencias/A/`, `evidencias/B/` y `evidencias/C/` son archivos gráficos de respaldo exportados por el código. **No reemplazan** las capturas de pantalla de la interfaz del notebook en ejecución donde se aprecie el número de celda (`[1]`, `[2]`, etc.), el código y las salidas correspondientes.

---

## 📸 Inciso A: `A_adaline_sin_frameworks.ipynb` (Adaline con NumPy Puro)

El estudiante debe tomar **4 capturas de pantalla**:

### Captura A-1: Carga, Validación del CSV y 10 Muestras Iniciales
- **Ubicación en el Notebook:** Celdas de código 1 y 2 (Pasos 1 y 2).
- **Elementos visibles obligatorios:**
  1. Celda con la carga de `NumerosManuscritos.csv`, la transposición `X = matriz_cruda.T` y la impresión de dimensiones: `(400, 5000)` y `(5000, 400)`.
  2. La validación numérica en consola: `Rango real de píxeles: [-0.1320, 1.1277]`, `Valores NaN: 0`, `Valores Inf: 0`.
  3. La figura con las 10 imágenes reconstruidas en orden Fortran (`order='F'`) con sus títulos de etiquetas (`Etiqueta: 0`, `Etiqueta: 1`, etc.).
- **Nombre sugerido de archivo:** `captura_A1_carga_muestras.png`

### Captura A-2: Bucle de Entrenamiento y Progreso de Épocas
- **Ubicación en el Notebook:** Celda de código 4 (Paso 4).
- **Elementos visibles obligatorios:**
  1. El código del bucle `for epoca in range(EPOCAS):`, el cálculo del gradiente de la regla delta `grad = Xb.T @ error / len(lote)` y la actualización `W += LR * grad`.
  2. La salida en texto mostrando el progreso de las épocas: desde `[Época  1/50] ECM: 0.1180 | Exactitud Train: 80.42%` hasta `[Época 50/50] ECM: 0.0741 | Exactitud Train: 88.22%`.
- **Nombre sugerido de archivo:** `captura_A2_entrenamiento_epocas.png`

### Captura A-3: Curvas de Aprendizaje (ECM y Exactitud)
- **Ubicación en el Notebook:** Celda de código 5 (Curvas de Aprendizaje).
- **Elementos visibles obligatorios:**
  1. La celda con el código de graficación de Matplotlib.
  2. Las dos gráficas generadas lado a lado: curva azul de descenso suave del Error Cuadrático Medio y curva verde de incremento de Exactitud hasta 88.22%.
- **Nombre sugerido de archivo:** `captura_A3_curvas_aprendizaje.png`

### Captura A-4: Evaluación en Prueba, 10 Predicciones y Matriz de Confusión
- **Ubicación en el Notebook:** Celdas de código 6 y 7 (Paso 5 y Análisis Avanzado).
- **Elementos visibles obligatorios:**
  1. El resumen en consola: `Muestras de prueba: 1000`, `Predicciones correctas: 851`, `Exactitud final (Test): 85.10%`.
  2. La rejilla de 10 imágenes con títulos en verde/rojo (`Pred: X | Real: Y`).
  3. La matriz de confusión con mapa de calor azul y números de conteo por casilla.
- **Nombre sugerido de archivo:** `captura_A4_evaluacion_test_matriz.png`

---

## 📸 Inciso B: `B_digitos_keras_csv.ipynb` (Dígitos con Keras y TensorFlow)

El estudiante debe tomar **4 capturas de pantalla**:

### Captura B-1: Carga, Normalización y Resumen de la Red Densa (`model.summary()`)
- **Ubicación en el Notebook:** Celdas de código 1 y 2.
- **Elementos visibles obligatorios:**
  1. Salida de normalización del rango real `[-0.1320, 1.1277] \to [0.0, 1.0]`, confirmación de la carga de `particion_train_test.npz` (3,200 train / 800 val / 1,000 test).
  2. La tabla de `model.summary()` de la Red Densa mostrando las capas `densa_128_relu`, `densa_64_relu` y `salida_10_softmax`, con el total de **60,234 parámetros**.
- **Nombre sugerido de archivo:** `captura_B1_arquitectura_resumen.png`

### Captura B-2: Ejecución del Entrenamiento con Keras
- **Ubicación en el Notebook:** Celda de código 3.
- **Elementos visibles obligatorios:**
  1. El llamado a `model_dense.fit(...)` con `batch_size=32`, `epochs=20`.
  2. Las líneas de progreso mostrando la evolución de `loss`, `accuracy`, `val_loss`, `val_accuracy` y el tiempo medido en segundos (~11.66 s).
- **Nombre sugerido de archivo:** `captura_B2_ejecucion_entrenamiento.png`

### Captura B-3: Curvas de Aprendizaje (Train vs Validación)
- **Ubicación en el Notebook:** Celda de código 4.
- **Elementos visibles obligatorios:**
  1. Gráfica izquierda: Pérdida train (azul) vs Pérdida val (naranja discontinua).
  2. Gráfica derecha: Exactitud train (verde) vs Exactitud val (roja discontinua), superando el 93.6% en validación.
- **Nombre sugerido de archivo:** `captura_B3_curvas_aprendizaje_dense.png`

### Captura B-4: Evaluación en Test, Matriz de Confusión y Galería de Predicciones
- **Ubicación en el Notebook:** Celdas de código 5 y 6.
- **Elementos visibles obligatorios:**
  1. Resumen en consola de Test: `Aciertos: 916`, `Errores: 84`, `Exactitud en Test: 91.60%`.
  2. La matriz de confusión en mapa azul con 84 errores totales.
  3. La galería de 10 imágenes: fila superior con aciertos en verde y fila inferior con errores en rojo y sus probabilidades de softmax.
- **Nombre sugerido de archivo:** `captura_B4_evaluacion_matriz_errores.png`

---

## 📸 Inciso C: `C_fashion_mnist_dense_vs_cnn.ipynb` (Fashion-MNIST: Dense vs CNN)

El estudiante debe tomar **5 capturas de pantalla**:

### Captura C-1: Exploración del Dataset y Muestras Representativas
- **Ubicación en el Notebook:** Celda de código 1 (Paso 2).
- **Elementos visibles obligatorios:**
  1. Consola con dimensiones: `(60000, 28, 28)` train y `(10000, 28, 28)` test, normalización $[0, 1]$ y detección del dispositivo (`CPU Intel/AMD con oneDNN`).
  2. Rejilla de 10 imágenes con los nombres canónicos en español: Camiseta/top, Pantalón, Pulóver, Vestido, Abrigo, Sandalia, Camisa, Tenis/zapatilla, Bolso, Botín.
- **Nombre sugerido de archivo:** `captura_C1_muestras_clases.png`

### Captura C-2: Resumen de Arquitecturas (`model.summary()` Dense vs CNN)
- **Ubicación en el Notebook:** Celdas de código 2 y 3.
- **Elementos visibles obligatorios:**
  1. `model.summary()` del Modelo Denso: **109,386 parámetros**.
  2. `model.summary()` del Modelo CNN: capas `Conv2D(32)`, `MaxPooling2D`, `Conv2D(64)`, `MaxPooling2D`, `Dense(64)` y `Dense(10)` con **121,930 parámetros**.
- **Nombre sugerido de archivo:** `captura_C2_model_summary_comparativo.png`

### Captura C-3: Curvas Comparativas de Aprendizaje Superpuestas
- **Ubicación en el Notebook:** Celda de código 4 (Paso 8).
- **Elementos visibles obligatorios:**
  1. Gráfica izquierda: Curvas de pérdida superpuestas de Dense (azul) y CNN (verde) para train y validación.
  2. Gráfica derecha: Curvas de exactitud superpuestas para train y validación durante las 10 épocas.
- **Nombre sugerido de archivo:** `captura_C3_curvas_comparativas.png`

### Captura C-4: Matrices de Confusión y Análisis de Categorías Críticas
- **Ubicación en el Notebook:** Celda de código 5 (Paso 9).
- **Elementos visibles obligatorios:**
  1. Salida en consola: `Red Densa -> Exactitud: 87.28% | Fallos: 1272`, `Red CNN -> Exactitud: 88.55% | Fallos: 1145`.
  2. La matriz de confusión con etiquetas en español rotadas y mapas de calor (azul y verde).
- **Nombre sugerido de archivo:** `captura_C4_matrices_confusion_comparativas.png`

### Captura C-5: Muestras Mal Clasificadas y Tabla Comparativa Final
- **Ubicación en el Notebook:** Celdas de código 6 y 7 (Pasos 10 y 11).
- **Elementos visibles obligatorios:**
  1. Galería de prendas mal clasificadas con etiquetas reales vs predichas y probabilidades.
  2. La tabla comparativa formal renderizada con todas las métricas medidas (parámetros, tiempos, exactitud train/val/test y brechas de generalización).
- **Nombre sugerido de archivo:** `captura_C5_errores_y_tabla_final.png`

---

## 💡 Consejos para la Máxima Calificación
1. Asegúrate de que los números de ejecución de celda (`[1]`, `[2]`, `[3]...`) sean visibles y consecutivos.
2. Captura la pantalla completa de la ventana del notebook o utiliza la herramienta de recortes de Windows (`Win + Shift + S`) seleccionando el área de la celda y su salida completa.
3. Guarda las capturas en una carpeta organizada (por ejemplo, `capturas/A/`, `capturas/B/`, `capturas/C/`) para adjuntarlas en tu reporte o plataforma académica.
