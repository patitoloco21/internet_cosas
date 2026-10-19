# Práctica de Redes Neuronales: De Modelos Lineales a Redes Profundas y Convolucionales

**Repositorio del Módulo:** `internet_cosas`  
**Autor:** Práctica Académica Reproducible  
**Sistema Operativo Objetivo:** Microsoft Windows 10 / 11  
**Versión de Python:** Python 3.10.8 (64-bit)  
**Semilla Global de Reproducibilidad:** 42  

---

## 📁 Estructura Completa del Repositorio

El proyecto se encuentra estrictamente confinado dentro de la carpeta `internet_cosas`, organizado de la siguiente manera:

```text
internet_cosas/
├── datos/
│   ├── NumerosManuscritos.csv            <- Dataset crudo de dígitos manuscritos (400 píxeles x 5000 muestras)
│   └── particion_train_test.npz          <- Partición determinista 80/20 compartida de forma idéntica entre A y B
├── notebooks/
│   ├── A_adaline_sin_frameworks.ipynb    <- [Inciso A] Adaline con NumPy puro, regla delta y salida lineal
│   ├── B_digitos_keras_csv.ipynb         <- [Inciso B] Red Densa Keras y Experimento Adicional CNN
│   └── C_fashion_mnist_dense_vs_cnn.ipynb<- [Inciso C] Comparativa sistemática Dense vs CNN en Fashion-MNIST
├── evidencias/
│   ├── A/                                <- Figuras exportadas del Inciso A (muestras, curvas, matriz, campos receptivos)
│   ├── B/                                <- Figuras exportadas del Inciso B (curvas dense, matriz, predicciones, CNN)
│   └── C/                                <- Figuras exportadas del Inciso C (muestras, curvas, matrices, errores, tabla)
├── documentos/
│   ├── A_explicacion.md                  <- Documento académico detallado del Inciso A
│   ├── B_explicacion.md                  <- Documento académico detallado del Inciso B
│   ├── C_comparacion_y_analisis.md       <- Documento académico y comparativa del Inciso C
│   └── guia_capturas.md                  <- Especificación exacta de capturas solicitadas al estudiante
├── adaline_digitos.py                    <- Script base original intacto
├── requirements.txt                      <- Dependencias del proyecto (NumPy, Matplotlib, TensorFlow, Keras, etc.)
└── README.md                             <- Guía de instalación, ejecución y síntesis de resultados
```

---

## 📊 Síntesis de Resultados Empíricos Reales

| Ejercicio / Modelo | Dataset / Entrada | Parámetros | Tiempo | Exactitud Train | Exactitud Val | Exactitud Test | Errores en Test |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **A: Adaline (NumPy puro)** | `NumerosManuscritos` (401 entradas con sesgo) | 4,010 | ~3.8 s | 88.22% | — | **85.10%** | 149 / 1,000 |
| **B: Red Densa (Keras MLP)** | `NumerosManuscritos` (400 características) | 60,234 | 11.66 s | 98.97% | 93.62% | **91.60%** | 84 / 1,000 |
| **B: CNN Adicional (Keras)** | `NumerosManuscritos` (Tensores 20x20x1 'F') | 55,274 | 22.68 s | 99.34% | 97.25% | **96.80%** | **32 / 1,000** |
| **C: Red Densa (Fashion)** | Fashion-MNIST (Aplanado 784 píxeles) | 109,386 | 38.14 s | 91.31% | 88.56% | **87.28%** | 1,272 / 10,000 |
| **C: Red CNN (Fashion)** | Fashion-MNIST (Tensores 28x28x1) | 121,930 | 131.19 s | 94.56% | 88.92% | **88.55%** | **1,145 / 10,000** |

---

## ⚙️ Instrucciones de Instalación y Ejecución en Windows

### 1. Ubicación y Terminal
Abre PowerShell o Símbolo del Sistema (CMD) y sitúate en la carpeta del proyecto:
```powershell
cd c:\red_neuronal\internet_cosas
```

### 2. Entorno Virtual e Instalación de Dependencias
```powershell
# Crear y activar entorno virtual
python -m venv venv
.\venv\Scripts\Activate.ps1

# Instalar dependencias exactas
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Selección de Kernel y Apertura de Notebooks
- **En Visual Studio Code:**
  1. Abre la carpeta `internet_cosas` (`File > Open Folder...`).
  2. Abre cualquiera de los tres cuadernos ubicados en `notebooks/`.
  3. En la esquina superior derecha, pulsa sobre **"Select Kernel"** $\to$ **"Python Environments..."** y elige el entorno virtual o el Python instalado en tu máquina.
  4. Los cuadernos ya cuentan con todas las salidas, tablas y gráficas renderizadas de su ejecución real, y pueden volverse a ejecutar secuencialmente celda por celda (`Run All` o `Shift + Enter`).
- **En JupyterLab / Jupyter Notebook clásico:**
  ```powershell
  jupyter notebook
  ```

---

## 📸 Guía para la Toma de Capturas de Pantalla
Consulta el archivo [documentos/guia_capturas.md](file:///c:/red_neuronal/internet_cosas/documentos/guia_capturas.md) para conocer con exactitud qué celdas y salidas deben capturarse de cada cuaderno para conformar el reporte final de entrega.
