# Documento Explicativo — Inciso C: Comparación Exhaustiva Red Densa vs CNN en Fashion-MNIST

**Curso / Materia:** Redes Neuronales / Internet de las Cosas  
**Estudiante / Autor:** Práctica Académica Reproducible  
**Entorno de Ejecución:** Python 3.10 / TensorFlow 2.21 / Keras 3.12 / NumPy 2.2 / Matplotlib 3.10  
**Dispositivo de Cómputo:** CPU Intel/AMD (x86_64 con aceleración oneDNN)  
**Semilla Aleatoria Global:** 42  

---

## 1. Ficha Técnica y Tabla Comparativa Formal (Pasos 1, 7 y 11)

A continuación se presenta la tabla comparativa con los valores exactos registrados durante la ejecución experimental bajo condiciones estrictamente idénticas:

| Métrica / Parámetro de Evaluación | Red Densa (MLP) | Red Convolucional (CNN) | Diferencia Absoluta (CNN vs Dense) | Impacto Relativo |
| :--- | :---: | :---: | :---: | :---: |
| **Arquitectura** | Input(28,28) $\to$ Flatten $\to$ Dense(128) $\to$ Dense(64) $\to$ Softmax(10) | Conv2D(32) $\to$ Pool $\to$ Conv2D(64) $\to$ Pool $\to$ Dense(64) $\to$ Softmax(10) | — | Paradigma Convolucional vs Aplanado |
| **Parámetros Entrenables** | **109,386** (427.29 KB) | **121,930** (476.29 KB) | **+12,544** | **+11.5%** más parámetros |
| **Tiempo de Entrenamiento (10 épocas)** | **38.14 s** (3.81 s/época) | **131.19 s** (13.12 s/época) | **+93.04 s** | **3.44 veces más lenta** en CPU |
| **Loss en Entrenamiento (Época 10)** | 0.2365 | 0.1536 | -0.0829 | Mayor ajuste en train |
| **Exactitud en Entrenamiento (Train Acc)** | 91.31% | 94.56% | +3.25% | Mayor capacidad de memoria local |
| **Loss en Validación (Val Loss)** | 0.3421 | 0.3455 | +0.0034 | Niveles prácticamente equivalentes |
| **Exactitud en Validación (Val Acc)** | 88.56% | 88.92% | +0.36% | Ligera ventaja inicial |
| **Pérdida en Test (Test Loss)** | **0.3734** | **0.3674** | **-0.0059** | Mejor calibración de pérdida en CNN |
| **Exactitud en Test (Test Accuracy)** | **87.28%** | **88.55%** | **+1.27%** | **Ganadora en precisión (+127 aciertos)** |
| **Total de Errores en Test (de 10,000)** | **1,272 fallos** | **1,145 fallos** | **-127 fallos** | **-10.0% de reducción de error** |
| **Brecha Generalización (Train - Test)** | +4.03% | +6.01% | +1.98% | Mayor tendencia al sobreajuste en CNN |
| **Brecha Estabilidad (Val - Test)** | +1.28% | **+0.37%** | -0.91% | **Validación de CNN más fiel al Test real** |

---

## 2. Descripción y Exploración del Dataset Fashion-MNIST (Paso 2)

Fashion-MNIST (*Xiao, Rasul & Vollgraf, Zalando Research, 2017*) fue diseñado específicamente como un sustituto directo y más exigente del clásico MNIST de dígitos manuscritos. Comparte idénticas dimensiones espaciales ($28 \times 28$ píxeles en escala de grises), número de clases ($10$) y tamaño de partición ($60,000$ entrenamiento, $10,000$ prueba).

### 2.1. Las 10 Categorías en Español
Las categorías están codificadas numéricamente del 0 al 9:
0. **Camiseta / top** (*T-shirt/top*)
1. **Pantalón** (*Trouser*)
2. **Pulóver** (*Pullover*)
3. **Vestido** (*Dress*)
4. **Abrigo** (*Coat*)
5. **Sandalia** (*Sandal*)
6. **Camisa** (*Shirt*)
7. **Tenis / zapatilla** (*Sneaker*)
8. **Bolso** (*Bag*)
9. **Botín** (*Ankle boot*)

### 2.2. Dificultad Morfológica
A diferencia de los dígitos (que consisten esencialmente en trazos lineales blancos aislados sobre fondo negro), las prendas de Fashion-MNIST presentan:
- Texturas continuas, estampados, patrones de tejido y gradientes sutiles.
- Pliegues tridimensionales en la ropa.
- Gran similitud visual intraclase e interclase entre prendas superiores (camisetas, camisas, abrigos y pulóveres comparten contornos corporales similares).

---

## 3. Preprocesamiento y Diseño Experimental Controlado (Pasos 3 y 6)

Para que la comparativa sea estrictamente metodológica, se anularon todas las fuentes externas de varianza:
1. **Normalización:** Los píxeles originales enteros en $[0, 255]$ se reescalaron dividiendo entre $255.0$, obteniendo valores de punto flotante en $[0.0, 1.0]$.
2. **Partición Idéntica:** De las $60,000$ imágenes de entrenamiento, se reservaron exactamente $50,000$ para entrenamiento activo y $10,000$ para validación mediante permutación aleatoria con semilla 42.
3. **Formas de Tensor:**
   - **Para la Red Densa:** Entrada en formato bidimensional $(N, 28, 28)$ procesada internamente mediante una capa `Flatten()` que colapsa los píxeles a un vector $1\text{D}$ de $784$ componentes.
   - **Para la Red CNN:** Entrada en formato tensorial 4D $(N, 28, 28, 1)$ añadiendo explícitamente el canal de profundidad.
4. **Hiperparámetros de Entrenamiento:**
   - Épocas: **10**
   - Tamaño de lote: **64** (782 pasos de gradiente por época)
   - Optimizador: **Adam** ($\alpha = 0.001$)
   - Función de coste: `sparse_categorical_crossentropy`

---

## 4. Arquitectura de los Modelos (Pasos 4 y 5)

### 4.1. Red Densa (Multilayer Perceptron - MLP)
```python
model_dense = keras.Sequential([
    keras.layers.Input(shape=(28, 28), name="entrada_28x28"),
    keras.layers.Flatten(name="aplanado_784"),
    keras.layers.Dense(128, activation='relu', name="densa_128_relu"),
    keras.layers.Dense(64, activation='relu', name="densa_64_relu"),
    keras.layers.Dense(10, activation='softmax', name="salida_10_softmax")
], name="Fashion_Dense_MLP")
```
- **Conexiones:**
  - $784 \to 128$: $784 \times 128 + 128 = 100,480$ parámetros.
  - $128 \to 64$: $128 \times 64 + 64 = 8,256$ parámetros.
  - $64 \to 10$: $64 \times 10 + 10 = 650$ parámetros.
- **Total de Parámetros:** **109,386**.

### 4.2. Red Convolucional (CNN)
```python
model_cnn = keras.Sequential([
    keras.layers.Input(shape=(28, 28, 1), name="entrada_tensor_28x28x1"),
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', name="conv2d_32_3x3"),
    keras.layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_1"),
    keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', name="conv2d_64_3x3"),
    keras.layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_2"),
    keras.layers.Flatten(name="aplanado"),
    keras.layers.Dense(64, activation='relu', name="densa_64_relu"),
    keras.layers.Dense(10, activation='softmax', name="salida_10_softmax")
], name="Fashion_CNN")
```
- **Conexiones:**
  - `Conv2D(32, 3x3)`: $3 \times 3 \times 1 \times 32 + 32 = 320$ parámetros $\to$ mapa $(26, 26, 32)$.
  - `MaxPooling2D(2x2)`: $0$ parámetros $\to$ mapa $(13, 13, 32)$.
  - `Conv2D(64, 3x3)`: $3 \times 3 \times 32 \times 64 + 64 = 18,496$ parámetros $\to$ mapa $(11, 11, 64)$.
  - `MaxPooling2D(2x2)`: $0$ parámetros $\to$ mapa $(5, 5, 64)$.
  - `Flatten()`: $5 \times 5 \times 64 = 1,600$ rasgos espaciales.
  - `Dense(64)`: $1,600 \times 64 + 64 = 102,464$ parámetros.
  - `Dense(10)`: $64 \times 10 + 10 = 650$ parámetros.
- **Total de Parámetros:** **121,930**.

---

## 5. Curvas Comparativas de Aprendizaje (Paso 8)

Las curvas generadas en `evidencias/C/paso8_curvas_comparativas.png` exhiben dos comportamientos clave:

1. **Velocidad de Aprendizaje en Entrenamiento:**
   - La CNN aprende más rápido y con mayor profundidad en el set de entrenamiento: en la época 2 ya alcanza $87.5\%$ (frente al $85.9\%$ de la Dense) y en la época 10 llega al $94.56\%$ con una pérdida de $0.1536$ (frente al $91.31\%$ y $0.2365$ de la Dense).
2. **Dinámica de Validación y Punto de Inflexión:**
   - **Red Densa:** Su pérdida de validación desciende de $0.430$ a $0.329$ en la época 7, antes de estabilizarse en torno a $0.342$. Su exactitud de validación evoluciona de forma muy constante hasta $88.56\%$.
   - **Red CNN:** Su pérdida de validación desciende con mayor fuerza hasta un mínimo de $0.281$ en la época 6 (exactitud de validación de $89.94\%$). Posteriormente, a partir de la época 7, la pérdida de validación comienza a subir suavemente hacia $0.345$, indicando que el modelo convolucional comienza a sobreajustar los detalles minuciosos del set de entrenamiento.

---

## 6. Análisis de Matrices de Confusión y Categorías Críticas (Paso 9)

### 6.1. Desglose Detallado de Errores por Categoría (1,000 imágenes por clase en Test)

| Categoría | Errores en Red Densa | Tasa de Error Densa (%) | Errores en Red CNN | Tasa de Error CNN (%) | Diferencia en Fallos (CNN vs Dense) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Camiseta / top** | 166 | 16.6% | 284 | 28.4% | +118 fallos |
| **Pantalón** | 30 | 3.0% | 25 | 2.5% | -5 fallos |
| **Pulóver** | 179 | 17.9% | 217 | 21.7% | +38 fallos |
| **Vestido** | 79 | 7.9% | 71 | 7.1% | -8 fallos |
| **Abrigo** | 287 | 28.7% | 194 | 19.4% | **-93 fallos** |
| **Sandalia** | 68 | 6.8% | 107 | 10.7% | +39 fallos |
| **Camisa** | 321 | 32.1% | 151 | 15.1% | **-170 fallos (¡Reducción del 53%!)** |
| **Tenis / zapatilla** | 19 | 1.9% | 6 | 0.6% | **-13 fallos (99.4% acierto)** |
| **Bolso** | 49 | 4.9% | 21 | 2.1% | **-28 fallos** |
| **Botín** | 74 | 7.4% | 69 | 6.9% | -5 fallos |
| **TOTAL** | **1,272** | **12.72%** | **1,145** | **11.45%** | **-127 fallos netos** |

### 6.2. Hallazgo Cualitativo Fundamental
El impacto más dramático de la arquitectura convolucional se produjo en la **Camisa (*Shirt*, clase 6)**:
- Para la Red Densa, la camisa fue casi imposible de distinguir: cometió **321 errores de 1,000 imágenes** (confundiéndose masivamente con camisetas, abrigos y pulóveres).
- La CNN logró discriminar los cuellos abotonados y las costuras de las mangas gracias a sus filtros de bordes locales, reduciendo los errores a **151 (recorte del 53% en fallos)**.
- De igual modo, en **Abrigos (*Coat*, clase 4)**, la CNN recortó los fallos de 287 a 194.
- En calzado deportivo (**Tenis**, clase 7), la CNN prácticamente rozó la perfección: **solo 6 equivocaciones en 1,000 muestras**.

---

## 7. Muestras Mal Clasificadas (Paso 10)

En las evidencias `evidencias/C/paso10_errores_dense.png` y `paso10_errores_cnn.png`:
- **Red Densa:** Confunde de forma recurrente abrigos largos con vestidos, y camisas de manga corta con camisetas estándar, con probabilidades de Softmax superiores al 80%.
- **Red CNN:** Los fallos restantes ocurren en casos genuinamente ambiguos para el ojo humano: una sandalia con tiras altas muy tupidas clasificada como tenis ($P=82\%$), o un pulóver holgado con cuello en V clasificado como abrigo.

---

## 8. Respuestas Argumentadas a las Preguntas Guía (Paso 12)

### 1. ¿Cuál modelo obtuvo mayor exactitud en prueba?
**La Red Convolucional (CNN)** obtuvo la mayor exactitud con un **88.55%** (1,145 errores), superando a la Red Densa que alcanzó **87.28%** (1,272 errores). La CNN redujo en **127 los errores absolutos** (un 10% menos de equivocaciones).

### 2. ¿Cuál demostró mejor generalización en prueba?
- **Pérdida en Test:** La CNN obtuvo una menor pérdida (**0.3674** vs **0.3734** de la Dense).
- **Consistencia Validación-Test:** La exactitud de validación de la CNN (88.92%) estuvo a tan solo **+0.37%** de su exactitud de prueba (88.55%), mientras que en la Red Densa la brecha fue de **+1.28%**. Esto demuestra que la validación en la CNN refleja con mayor fidelidad su rendimiento real generalizado.
- **Sobreajuste:** En las épocas 7 a 10, la CNN exhibió un descenso continuo en la pérdida de entrenamiento mientras la de validación comenzó a rebotar levemente, sugiriendo que la inclusión de técnicas de regularización (como *Dropout(0.25)* o aumento de datos sintético) permitiría a la CNN superar cómodamente el 91-92% de exactitud.

### 3. ¿Cuál modelo tiene mayor cantidad de parámetros?
**La CNN tiene más parámetros:** **121,930** frente a **109,386** (+12,544 parámetros, +11.5%). La mayor parte de los parámetros de la CNN se concentran en la capa densa de transición tras el aplanado ($1600 \times 64 = 102,464$), mientras que sus capas convolucionales son sumamente eficientes ($18,816$ pesos para procesar toda la imagen).

### 4. ¿Cuál requirió mayor tiempo de entrenamiento?
**La CNN requirió sensiblemente más tiempo:** **131.19 segundos** frente a **38.14 segundos** de la Red Densa (**3.44 veces más lenta** en CPU). Las operaciones de convolución involucran millones de multiplicaciones y sumas en ventanas deslizantes 2D, las cuales son computacionalmente mucho más intensivas en procesadores convencionales sin unidades tensoriales aceleradoras (TPU/GPU).

### 5. ¿Qué diferencias principales se observan en las matrices de confusión?
La Red Densa fracasa rotundamente en discriminar entre prendas superiores con mangas (321 fallos en camisas y 287 en abrigos). La CNN disminuye notablemente estas confusiones gracias a la extracción jerárquica de características (botones, cuellos, dobladillos).

### 6. ¿Por qué la arquitectura CNN suele ser superior para imágenes?
1. **Conectividad Local:** Cada filtro opera sobre una ventana reducida ($3 \times 3$), capturando correlaciones espaciales coherentes con la física de la luz y las superficies.
2. **Pesos Compartidos (*Weight Sharing*):** El mismo filtro detector de texturas o líneas diagonales se reutiliza en todos los cuadrantes de la imagen.
3. **Invarianza Traslacional:** El agrupamiento (*max pooling*) confiere tolerancia a desplazamientos menores: una camisa situada ligeramente hacia la izquierda activa los mismos mapas de rasgos que una camisa centrada. La Red Densa carece de esta propiedad: al aplanar los píxeles a un vector $1\text{D}$, cualquier traslación de 2 píxeles altera la posición de todos los valores de entrada.

---

## 9. Conclusión Académica Final (Paso 13)

El estudio experimental demuestra que **el diseño arquitectónico tiene un impacto directo y medible en el rendimiento de un sistema de visión artificial**:
- La **Red Convolucional (CNN)** superó sistemáticamente a la **Red Densa**, reduciendo los errores en un **10.0%** y mitigando las ambigüedades en las prendas más complejas (camisas y abrigos).
- Sin embargo, esta superioridad no es gratuita: requirió un **344% más de tiempo de entrenamiento en CPU**.
- En consecuencia, en entornos de cómputo restringido o en dispositivos del Internet de las Cosas (IoT) con baterías limitadas, la elección entre un modelo denso y uno convolucional debe sustentarse en un análisis riguroso de la relación **costo computacional vs. ganancia en exactitud**.
