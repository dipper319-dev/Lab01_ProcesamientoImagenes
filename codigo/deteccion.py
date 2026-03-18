import cv2
import numpy as np
import json

# ------------------------
# Para Los que usen esto por primera vez ejecuten esto en consola:
# "pip install -r requirements.txt"
# ------------------------

pausar_video = False
mostrar_pixel = False
x, y = 0, 0

# Listas para guardar datos cinematicos (usados en punto 3)
centroides = []   # Lista de (cx, cy) por frame
tiempos = []      # Tiempo en segundos de cada frame

# Estado del último frame procesado (se reutiliza al pausar)
# CORRECCIÓN BUG: separar "calcular" de "dibujar" evita que MOG2
# siga acumulando el frame congelado y expanda la máscara infinitamente
mascara_guardada     = None
contorno_guardado    = None
cx_guardado          = None
cy_guardado          = None
frame_base           = None   # Copia limpia del frame (sin dibujos) para redibujar

# -----------------------------
# Función del mouse
# -----------------------------

def mouse_callback(event, _x, _y, flags, param):
    global pausar_video, mostrar_pixel, x, y

    if event == cv2.EVENT_LBUTTONDOWN:
        x, y = _x, _y
        mostrar_pixel = True
        # Imprimir en consola para copiar fácilmente al punto 3
        print(f"📍 Clic en X={_x}, Y={_y}  ← usa este valor en el Punto 3")

    if event == cv2.EVENT_RBUTTONDOWN:
        pausar_video = not pausar_video


# -----------------------------
# Abrir video
# -----------------------------

cap = cv2.VideoCapture("video/Test3.mkv")

if not cap.isOpened():
    print("Error: no se pudo abrir el video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30.0
    print(f"Advertencia: no se pudo leer FPS, usando {fps} por defecto")
else:
    print(f"Video abierto correctamente | FPS: {fps:.2f}")

frame_num = 0

# -----------------------------
# Crear sustractor de fondo MOG2
# -----------------------------

subtractor = cv2.createBackgroundSubtractorMOG2(
    history=300,
    varThreshold=50,
    detectShadows=True
)

# -----------------------------
# Crear ventanas
# -----------------------------

WINDOW_WIDTH  = 800
WINDOW_HEIGHT = 600

cv2.namedWindow("Video",    cv2.WINDOW_NORMAL)
cv2.resizeWindow("Video",   WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.setMouseCallback("Video", mouse_callback)

cv2.namedWindow("Mascara",  cv2.WINDOW_NORMAL)
cv2.resizeWindow("Mascara", WINDOW_WIDTH, WINDOW_HEIGHT)

cv2.namedWindow("MOG2 Raw", cv2.WINDOW_NORMAL)
cv2.resizeWindow("MOG2 Raw",WINDOW_WIDTH, WINDOW_HEIGHT)

cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Trackbars", 400, 400)

def nothing(x):
    pass

# -----------------------------
# Trackbars
# -----------------------------

cv2.createTrackbar("Area Min",    "Trackbars", 500,  10000, nothing)
cv2.createTrackbar("Kernel",      "Trackbars", 5,    15,    nothing)
cv2.createTrackbar("Sensibilidad","Trackbars", 50,   200,   nothing)
cv2.createTrackbar("Mezcla HSV",  "Trackbars", 0,    100,   nothing)
cv2.createTrackbar("H Min",       "Trackbars", 0,    179,   nothing)
cv2.createTrackbar("H Max",       "Trackbars", 179,  179,   nothing)
cv2.createTrackbar("S Min",       "Trackbars", 0,    255,   nothing)
cv2.createTrackbar("S Max",       "Trackbars", 255,  255,   nothing)
cv2.createTrackbar("V Min",       "Trackbars", 0,    255,   nothing)
cv2.createTrackbar("V Max",       "Trackbars", 255,  255,   nothing)

# -----------------------------
# Loop principal
# -----------------------------

while True:

    # ===========================================================
    # BLOQUE A: Solo se ejecuta cuando el video NO está pausado
    # Aquí van TODOS los cálculos pesados: MOG2, morfología, etc.
    # CORRECCIÓN: al pausar, este bloque no se ejecuta → MOG2
    # deja de acumular el frame congelado → la máscara no crece.
    # ===========================================================

    if not pausar_video:
        ret, frame = cap.read()

        if not ret:
            print("Fin del video.")
            break

        # Guardar copia limpia del frame para redibujar sin acumular trazos
        frame_base = np.copy(frame)

        # --- Leer trackbars ---
        area_min     = cv2.getTrackbarPos("Area Min",     "Trackbars")
        kernel_size  = cv2.getTrackbarPos("Kernel",       "Trackbars")
        sensibilidad = cv2.getTrackbarPos("Sensibilidad", "Trackbars")
        mezcla_hsv   = cv2.getTrackbarPos("Mezcla HSV",   "Trackbars")

        h_min = cv2.getTrackbarPos("H Min", "Trackbars")
        h_max = cv2.getTrackbarPos("H Max", "Trackbars")
        s_min = cv2.getTrackbarPos("S Min", "Trackbars")
        s_max = cv2.getTrackbarPos("S Max", "Trackbars")
        v_min = cv2.getTrackbarPos("V Min", "Trackbars")
        v_max = cv2.getTrackbarPos("V Max", "Trackbars")

        subtractor.setVarThreshold(max(sensibilidad, 1))

        k = max(1, kernel_size)
        if k % 2 == 0:
            k += 1
        kernel = np.ones((k, k), np.uint8)

        # --- Máscara MOG2 ---
        mascara_mog2_raw = subtractor.apply(frame_base)
        mascara_mog2     = cv2.threshold(mascara_mog2_raw, 200, 255, cv2.THRESH_BINARY)[1]

        # --- Máscara HSV (opcional) ---
        mascara_hsv = np.zeros_like(mascara_mog2)
        if mezcla_hsv > 0:
            hsv   = cv2.cvtColor(frame_base, cv2.COLOR_BGR2HSV)
            lower = np.array([h_min, s_min, v_min])
            upper = np.array([h_max, s_max, v_max])
            mascara_hsv = cv2.inRange(hsv, lower, upper)

        # --- Combinar máscaras ---
        if mezcla_hsv == 0:
            mascara = mascara_mog2
        elif mezcla_hsv == 100:
            mascara = mascara_hsv
        else:
            mascara = cv2.bitwise_and(mascara_mog2, mascara_hsv)

        # --- Morfología ---
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN,  kernel)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
        mascara = cv2.dilate(mascara, kernel, iterations=1)

        # --- Contornos y centroide ---
        contornos, _ = cv2.findContours(
            mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        contorno_guardado = None
        cx_guardado       = None
        cy_guardado       = None

        if len(contornos) > 0:
            contorno_candidato = max(contornos, key=cv2.contourArea)

            if cv2.contourArea(contorno_candidato) > area_min:
                M = cv2.moments(contorno_candidato)

                if M["m00"] != 0:
                    cx_guardado = int(M["m10"] / M["m00"])
                    cy_guardado = int(M["m01"] / M["m00"])
                    contorno_guardado = contorno_candidato

                    # Guardar para el punto 3
                    centroides.append((cx_guardado, cy_guardado))
                    tiempos.append(frame_num / fps)

        # Guardar máscaras para mostrarlas aunque esté pausado
        mascara_guardada     = mascara.copy()
        mascara_mog2_guardada = mascara_mog2.copy()

        # Leer modo para el texto (también dentro del bloque)
        mezcla_hsv_txt = mezcla_hsv
        frame_num += 1

    # ===========================================================
    # BLOQUE B: Se ejecuta SIEMPRE (pausado o no)
    # Solo dibuja sobre el frame usando los datos ya calculados.
    # No llama a subtractor.apply() ni a operaciones morfológicas.
    # ===========================================================

    if frame_base is None:
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            break
        continue

    # Partir siempre desde la copia limpia del frame
    frame2 = np.copy(frame_base)

    # Mostrar valor del pixel clicado
    if mostrar_pixel:
        pixel_color = frame2[y, x]
        pixel_hsv   = cv2.cvtColor(np.uint8([[pixel_color]]), cv2.COLOR_BGR2HSV)[0][0]
        # Coordenadas del clic — esto es lo que necesitas para el Punto 3
        cv2.putText(frame2, f"Clic: X={x}  Y={y}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        # Color del pixel — útil para ajustar trackbars HSV
        cv2.putText(frame2, f"BGR: {pixel_color} | HSV: {pixel_hsv}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        # Cruz visual sobre el punto clicado
        cv2.drawMarker(frame2, (x, y), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

    # Dibujar detecciones si existen
    centroide_detectado = False

    if contorno_guardado is not None and cx_guardado is not None:
        centroide_detectado = True

        x_bb, y_bb, w_bb, h_bb = cv2.boundingRect(contorno_guardado)
        cv2.rectangle(frame2, (x_bb, y_bb), (x_bb + w_bb, y_bb + h_bb),
                      (255, 165, 0), 2)

        cv2.drawContours(frame2, [contorno_guardado], -1, (0, 255, 0), 2)
        cv2.circle(frame2, (cx_guardado, cy_guardado), 6, (0, 0, 255), -1)

        cv2.putText(frame2, f"Centroide: ({cx_guardado}, {cy_guardado})",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        area_txt = f"Area: {int(cv2.contourArea(contorno_guardado))} px"
        cv2.putText(frame2, area_txt,
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Textos de estado
    estado      = "Vehiculo detectado" if centroide_detectado else "Sin deteccion"
    color_estado = (0, 255, 0) if centroide_detectado else (0, 0, 255)
    cv2.putText(frame2, estado,
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_estado, 2)

    tiempo_actual = (frame_num - 1) / fps if frame_num > 0 else 0
    cv2.putText(frame2, f"Frame: {frame_num} | t: {tiempo_actual:.2f}s",
                (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Indicador visual de pausa
    if pausar_video:
        cv2.putText(frame2, "|| PAUSADO", (10, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    modo_txt = "MOG2" if mezcla_hsv_txt == 0 else ("HSV" if mezcla_hsv_txt == 100 else "MOG2+HSV")
    cv2.putText(frame2, f"Modo: {modo_txt}",
                (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Mostrar ventanas
    cv2.imshow("Video",    frame2)

    if mascara_guardada is not None:
        cv2.imshow("Mascara",  mascara_guardada)
        cv2.imshow("MOG2 Raw", mascara_mog2_guardada)

    key = cv2.waitKey(30) & 0xFF

    if key == 27:  # ESC
        break

    if key == ord('s'):
        print(f"\n--- Datos guardados hasta frame {frame_num} ---")
        print(f"Total centroides registrados: {len(centroides)}")
        if centroides:
            print(f"Ultimo centroide: {centroides[-1]}")
            print(f"Tiempo total: {tiempos[-1]:.2f}s")

# -----------------------------
# Guardar resultados y cerrar
# -----------------------------

print(f"\n=== Procesamiento finalizado ===")
print(f"FPS del video: {fps:.2f}")
print(f"Frames procesados: {frame_num}")
print(f"Centroides detectados: {len(centroides)}")

if len(centroides) > 0:
    print(f"Primer centroide: {centroides[0]} en t={tiempos[0]:.3f}s")
    print(f"Ultimo centroide: {centroides[-1]} en t={tiempos[-1]:.3f}s")
    print(f"\nEstos datos (centroides y tiempos) estan listos para el Punto 3.")

    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    datos_cinematicos = {
        "fps": float(fps),
        "resolucion": {
            "ancho": frame_width,
            "alto":  frame_height
        },
        "total_frames":     frame_num,
        "total_centroides": len(centroides),
        "centroides": [
            {"x": int(cx), "y": int(cy)}
            for cx, cy in centroides
        ],
        "tiempos": [float(t) for t in tiempos]
    }

    try:
        with open("datos_cinematicos.json", "w") as f:
            json.dump(datos_cinematicos, f, indent=4)
        print(f"\n✓ Datos guardados en: datos_cinematicos.json")
        print(f"  - FPS: {fps:.2f}")
        print(f"  - Resolución: {frame_width}x{frame_height}")
        print(f"  - Centroides: {len(centroides)}")
    except Exception as e:
        print(f"\n✗ Error al guardar JSON: {e}")
else:
    print("\n✗ No hay centroides para guardar")

cap.release()
cv2.destroyAllWindows()