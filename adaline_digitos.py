"""
Adaline (ADAptive LInear NEuron) con NumPy puro para reconocer dígitos manuscritos.

Pasos:
  1. Cargar el CSV (400 filas = píxeles, 5000 columnas = imágenes)
  2. Mostrar 10 imágenes aleatorias del dataset
  3. Dividir 80% entrenamiento / 20% prueba
  4. Entrenar Adaline por lotes (mini-batch gradient descent)
  5. Evaluar y mostrar 10 imágenes aleatorias de PRUEBA con su clase predicha
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)          # semilla para reproducibilidad

# ----------------------------------------------------------------------------
# PASO 1: Cargar datos
# ----------------------------------------------------------------------------
# El CSV viene como (400 píxeles x 5000 imágenes). Lo transponemos para que
# cada FILA sea una imagen: X tiene forma (5000, 400).
RUTA = r"C:\Users\lolqu\Downloads\NumerosManuscritos.csv"
X = np.loadtxt(RUTA, delimiter=",").T
n_muestras, n_pixeles = X.shape

# El CSV no trae etiquetas. Las imágenes están ordenadas en bloques de 500
# por dígito (0,0,...,1,1,...,9,9), así que la clase es índice // 500.
y = np.arange(n_muestras) // 500

print(f"[Paso 1] X: {X.shape} | y: {y.shape} | clases: {np.unique(y)}")


def a_imagen(vector):
    # Las imágenes se guardaron por columnas (estilo MATLAB), de ahí order="F"
    return vector.reshape(20, 20, order="F")


# ----------------------------------------------------------------------------
# PASO 2: Mostrar 10 imágenes aleatorias del dataset
# ----------------------------------------------------------------------------
idx = rng.choice(n_muestras, 10, replace=False)
fig, ejes = plt.subplots(2, 5, figsize=(10, 4.5))
for ax, i in zip(ejes.ravel(), idx):
    ax.imshow(a_imagen(X[i]), cmap="gray")
    ax.set_title(f"Etiqueta: {y[i]}")
    ax.axis("off")
fig.suptitle("Paso 2: 10 imágenes aleatorias del dataset")
plt.tight_layout()
plt.savefig("paso2_muestras.png", dpi=120)
plt.show()

# ----------------------------------------------------------------------------
# PASO 3: Separar entrenamiento (80%) y prueba (20%)
# ----------------------------------------------------------------------------
perm = rng.permutation(n_muestras)       # barajamos: el CSV viene ordenado por clase
n_train = int(0.8 * n_muestras)
X_train, y_train = X[perm[:n_train]], y[perm[:n_train]]
X_test,  y_test  = X[perm[n_train:]], y[perm[n_train:]]
print(f"[Paso 3] Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# Agregamos una columna de 1s: así el sesgo (bias) vive dentro de la matriz de pesos.
X_train_b = np.hstack([np.ones((len(X_train), 1)), X_train])
X_test_b  = np.hstack([np.ones((len(X_test), 1)),  X_test])

# Objetivos: Adaline es un clasificador binario, así que usamos "uno contra todos":
# 10 neuronas Adaline, una por dígito. Objetivo = +1 si es su dígito, -1 si no.
N_CLASES = 10
T_train = -np.ones((len(y_train), N_CLASES))
T_train[np.arange(len(y_train)), y_train] = 1.0

# ----------------------------------------------------------------------------
# PASO 4: Entrenamiento de Adaline por lotes
# ----------------------------------------------------------------------------
TAM_LOTE = 32      # lote sugerido (ver explicación)
EPOCAS = 50
LR = 0.01          # tasa de aprendizaje (pequeña para que no diverja)

# W: (401 x 10). Una columna de pesos por cada neurona/dígito.
W = rng.normal(0, 0.01, size=(n_pixeles + 1, N_CLASES))

historial_ecm, historial_acc = [], []

for epoca in range(EPOCAS):
    orden = rng.permutation(n_train)     # barajar en cada época
    for ini in range(0, n_train, TAM_LOTE):
        lote = orden[ini:ini + TAM_LOTE]
        Xb, Tb = X_train_b[lote], T_train[lote]

        salida = Xb @ W                  # salida LINEAL (sin escalón): net = w·x
        error = Tb - salida              # error = deseado - obtenido
        grad = Xb.T @ error / len(lote)  # gradiente promedio del lote (regla delta)
        W += LR * grad                   # actualización de pesos

    # Métricas al final de cada época (sobre todo el set de entrenamiento)
    salida_tr = X_train_b @ W
    ecm = np.mean((T_train - salida_tr) ** 2) / 2
    acc = np.mean(np.argmax(salida_tr, axis=1) == y_train)
    historial_ecm.append(ecm)
    historial_acc.append(acc)
    if epoca % 5 == 0 or epoca == EPOCAS - 1:
        print(f"[Paso 4] Época {epoca + 1:3d} | ECM: {ecm:.4f} | Acc train: {acc:.2%}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.5))
a1.plot(historial_ecm); a1.set_title("Error cuadrático medio"); a1.set_xlabel("Época")
a2.plot(historial_acc); a2.set_title("Exactitud (train)");      a2.set_xlabel("Época")
plt.tight_layout()
plt.savefig("paso4_entrenamiento.png", dpi=120)
plt.show()

# ----------------------------------------------------------------------------
# PASO 5: Evaluar en prueba y mostrar 10 predicciones aleatorias
# ----------------------------------------------------------------------------
pred = np.argmax(X_test_b @ W, axis=1)   # gana la neurona con mayor salida
acc_test = np.mean(pred == y_test)
print(f"[Paso 5] Exactitud en PRUEBA: {acc_test:.2%}")

idx = rng.choice(len(X_test), 10, replace=False)
fig, ejes = plt.subplots(2, 5, figsize=(10, 4.8))
for ax, i in zip(ejes.ravel(), idx):
    ax.imshow(a_imagen(X_test[i]), cmap="gray")
    ok = pred[i] == y_test[i]
    ax.set_title(f"Pred: {pred[i]} (real: {y_test[i]})", color="green" if ok else "red")
    ax.axis("off")
fig.suptitle(f"Paso 5: predicciones sobre el conjunto de prueba (acc total: {acc_test:.1%})")
plt.tight_layout()
plt.savefig("paso5_predicciones.png", dpi=120)
plt.show()
