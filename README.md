# Lab01_ProcesamientoImagenes
Análisis del movimiento de un vehículo en video mediante visión por computadora con Python y OpenCV.

## Objetivo

Analizar el movimiento de un vehículo a partir de un video, aplicando segmentación, morfología, detección de contornos y cálculo de centroides para estimar posición, velocidad y aceleración, y comparar con un modelo cinemático teórico.

## Estructura del proyecto

- `video/Carro2.mp4`: video de entrada.
- `codigo/deteccion.py`: Punto 2 (detección y extracción de centroides).
- `codigo/analisis_cinematico.py`: Punto 3 (análisis cinemático).
- `datos_cinematicos.json`: salida del Punto 2 (insumo del Punto 3).
- `graficas_cinematica.png`: salida del Punto 3.
- `resultados_cinematicos.csv`: salida del Punto 3.

## Crear entorno virtual e instalar dependencias (Windows)

1. Si no existe `.venv` (o se borró), créalo:

```cmd
python -m venv .venv
```

Si `python` no funciona en tu sistema, usa:

```cmd
py -m venv .venv
```

2. Activar el entorno virtual:

```cmd
.venv\Scripts\activate
```
Si usas `PowerShell`, puede que necesites:

```PowerShell
.venv\Scripts\Activate.ps1
```
3. Instalar dependencias:

```cmd
python -m pip install -r requirements.txt
```

## Flujo de ejecución recomendado

Ejecuta solo 2 comandos en este orden:

1. `python codigo/deteccion.py`  → cubre Punto 2 y Punto 4 en la misma corrida.
2. `python codigo/analisis_cinematico.py` → cubre Punto 3 usando el JSON generado.

No es necesario ejecutar `deteccion.py` dos veces (a menos que quieras recalibrar, ajustar segmentación o volver a exportar video).

## Ejecución detallada (por puntos)

### 1) Detección + visualización (Punto 2 y Punto 4 en una sola ejecución)

Ejecuta:

```cmd
python codigo/deteccion.py
```

Controles durante ejecución:

- Clic izquierdo: muestra coordenada del píxel (útil para calibración A y B).
- Clic derecho: pausar/reanudar video.
- Tecla `Esc`: salir.

Salidas principales generadas:

- `datos_cinematicos.json` con:
	- FPS y resolución del video.
	- lista de centroides por frame detectado.
	- vector de tiempos asociado.
- `video_procesado_punto4.mp4` (si respondes `s` al exportar video).

### 2) Análisis cinemático (Punto 3)

Ejecuta:

```cmd
python codigo/analisis_cinematico.py
```

El script solicitará:

- Coordenada X del punto A (en píxeles).
- Coordenada X del punto B (en píxeles).
- Distancia real entre A y B (en metros).

Salidas generadas:

- `graficas_cinematica.png` (posición, velocidad y aceleración vs tiempo).
- `resultados_cinematicos.csv` (tiempo, centroides, posición, velocidad, aceleración).

### 3) Análisis cinemático (Punto 3)

Se ejecuta después de terminar la detección/visualización, usando el archivo `datos_cinematicos.json` generado en el paso anterior.

## Cobertura de la rúbrica (hasta Punto 3)

### Punto 1 (Grabación)

Se utiliza un video lateral con cámara fija y marcadores de referencia A/B para calibrar escala píxeles-metros.

### Punto 2 (Detección y segmentación)

Implementado en `codigo/deteccion.py`:

- Sustracción de fondo con MOG2.
- Soporte de segmentación HSV (opcional y combinable).
- Limpieza por operaciones morfológicas (apertura, cierre y dilatación).
- Detección de contornos con `cv2.findContours()`.
- Cálculo de centroides por momentos con `cv2.moments()`.

### Punto 3 (Análisis cinemático)

Implementado en `codigo/analisis_cinematico.py`:

- Conversión de posición de píxeles a metros (calibración A-B).
- Velocidad instantánea por diferencias finitas.
- Aceleración instantánea como derivada de la velocidad.
- Suavizado de señal (Savitzky-Golay) antes de derivar.
- Clasificación del tipo de movimiento y comparación con modelo teórico (MRU/MRUA).
- Gráficas y exportación de resultados en CSV.

## Punto 4 (Visualización y resultados)

Implementado en `codigo/deteccion.py`:

- Superposición de trayectoria histórica del centroide sobre el video.
- Visualización en tiempo real de velocidad instantánea (`px/s` y `m/s`).
- Marcadores visuales A/B con líneas verticales ajustables por trackbar.
- Escala métrica en pantalla (`m`, `px`, `px/m`) usando distancia A-B configurable.
- Exportación opcional de video final procesado: `video_procesado_punto4.mp4`.

Uso sugerido para el Punto 4:

1. Ejecuta `python codigo/deteccion.py`.
2. Responde `s` si deseas guardar video final procesado.
3. Ajusta en trackbars: `A X`, `B X`, `Dist AB cm`.
4. Verifica en pantalla trayectoria, velocidad y escala durante la detección.

