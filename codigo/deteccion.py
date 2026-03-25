import cv2
import numpy as np
import json

pausar_video = False
mostrar_pixel = False
x, y = 0, 0

centroides = []
tiempos = []

mascara_guardada = None
mascara_mog2_guardada = None
contorno_guardado = None
cx_guardado = None
cy_guardado = None
frame_base = None

ultimo_centroide = None
ultimo_tiempo = None
velocidad_inst_px_s = 0.0
velocidad_inst_m_s = 0.0

mezcla_hsv_txt = 0
ax = 100
ay = 100
bx = 300
by = 100
distancia_ab_cm = 500
distancia_ab_m = 5.0
px_por_m = 0.0
mostrar_marcadores = True
modo_seleccion_marcador = None

WARMUP_FRAMES = 20
MAX_DIST_SEGUIMIENTO_PX = 150
MAX_GAP_FACTOR = 3.0
AREA_MAX_RATIO = 0.35
BORDER_MARGIN_PX = 20
ASPECT_MIN = 0.8
ASPECT_MAX = 5.5
TRACK_LOST_FRAMES = 10

frames_sin_deteccion = 0


def mouse_callback(event, _x, _y, flags, param):
    global pausar_video, mostrar_pixel, x, y
    global ax, ay, bx, by, modo_seleccion_marcador

    if event == cv2.EVENT_LBUTTONDOWN:
        if modo_seleccion_marcador == "A":
            ax, ay = _x, _y
            modo_seleccion_marcador = None
            print(f"✅ Marcador A fijado: ({ax}, {ay})")
        elif modo_seleccion_marcador == "B":
            bx, by = _x, _y
            modo_seleccion_marcador = None
            print(f"✅ Marcador B fijado: ({bx}, {by})")
        else:
            x, y = _x, _y
            mostrar_pixel = True
            print(f"📍 Clic en X={_x}, Y={_y}  ← usa este valor para calibración")

    if event == cv2.EVENT_RBUTTONDOWN:
        pausar_video = not pausar_video


cap = cv2.VideoCapture("video/Carro2.mp4")

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
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Posiciones iniciales aproximadas; convención: A=cono derecho, B=cono izquierdo
ax = int(0.95 * frame_width)
ay = int(0.62 * frame_height)
bx = int(0.06 * frame_width)
by = int(0.62 * frame_height)

print("\n=== Configuración Punto 4 ===")
print("Controles: tecla 'a' + clic para fijar A en cono derecho; tecla 'b' + clic para B en cono izquierdo; tecla 'm' para ocultar/mostrar A-B.")
resp = input("¿Deseas guardar video final procesado? (s/n): ").strip().lower()
exportar_video = resp == 's'

video_writer = None
if exportar_video:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        "video_procesado_punto4.mp4", fourcc, fps, (frame_width, frame_height)
    )
    if not video_writer.isOpened():
        print("⚠ No se pudo crear video de salida. Se desactiva exportación.")
        video_writer = None
        exportar_video = False

subtractor = cv2.createBackgroundSubtractorMOG2(
    history=300,
    varThreshold=50,
    detectShadows=True
)

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Video", WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.setMouseCallback("Video", mouse_callback)

cv2.namedWindow("Mascara", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Mascara", WINDOW_WIDTH, WINDOW_HEIGHT)

cv2.namedWindow("MOG2 Raw", cv2.WINDOW_NORMAL)
cv2.resizeWindow("MOG2 Raw", WINDOW_WIDTH, WINDOW_HEIGHT)

cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Trackbars", 450, 450)


def nothing(value):
    pass


def seleccionar_mejor_contorno(contornos, area_minima, ultimo_centroide_ref, ancho_frame, alto_frame):
    area_maxima = AREA_MAX_RATIO * ancho_frame * alto_frame
    candidatos_validos = []

    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area < area_minima or area > area_maxima:
            continue

        x_box, y_box, w_box, h_box = cv2.boundingRect(contorno)
        if w_box <= 0 or h_box <= 0:
            continue

        relacion_aspecto = w_box / float(h_box)
        if relacion_aspecto < ASPECT_MIN or relacion_aspecto > ASPECT_MAX:
            continue

        momentos = cv2.moments(contorno)
        if momentos["m00"] == 0:
            continue

        cx = int(momentos["m10"] / momentos["m00"])
        cy = int(momentos["m01"] / momentos["m00"])

        if cx < BORDER_MARGIN_PX or cx > (ancho_frame - BORDER_MARGIN_PX):
            continue
        if cy < BORDER_MARGIN_PX or cy > (alto_frame - BORDER_MARGIN_PX):
            continue

        candidatos_validos.append((contorno, cx, cy, area))

    if not candidatos_validos:
        return None

    if ultimo_centroide_ref is None:
        return max(candidatos_validos, key=lambda item: item[3])

    candidatos_cercanos = []
    for candidato in candidatos_validos:
        _, cx, cy, area = candidato
        dist = np.hypot(cx - ultimo_centroide_ref[0], cy - ultimo_centroide_ref[1])
        if dist <= MAX_DIST_SEGUIMIENTO_PX:
            candidatos_cercanos.append((dist, -area, candidato))

    if candidatos_cercanos:
        candidatos_cercanos.sort(key=lambda item: (item[0], item[1]))
        return candidatos_cercanos[0][2]

    return max(candidatos_validos, key=lambda item: item[3])


cv2.createTrackbar("Area Min", "Trackbars", 500, 10000, nothing)
cv2.createTrackbar("Kernel", "Trackbars", 5, 15, nothing)
cv2.createTrackbar("Sensibilidad", "Trackbars", 50, 200, nothing)
cv2.createTrackbar("Mezcla HSV", "Trackbars", 0, 100, nothing)
cv2.createTrackbar("H Min", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("H Max", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("S Min", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("S Max", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("V Min", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("V Max", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("Dist AB cm", "Trackbars", 500, 5000, nothing)

while True:
    if not pausar_video:
        ret, frame = cap.read()

        if not ret:
            print("Fin del video.")
            break

        frame_base = np.copy(frame)

        area_min = cv2.getTrackbarPos("Area Min", "Trackbars")
        kernel_size = cv2.getTrackbarPos("Kernel", "Trackbars")
        sensibilidad = cv2.getTrackbarPos("Sensibilidad", "Trackbars")
        mezcla_hsv = cv2.getTrackbarPos("Mezcla HSV", "Trackbars")

        h_min = cv2.getTrackbarPos("H Min", "Trackbars")
        h_max = cv2.getTrackbarPos("H Max", "Trackbars")
        s_min = cv2.getTrackbarPos("S Min", "Trackbars")
        s_max = cv2.getTrackbarPos("S Max", "Trackbars")
        v_min = cv2.getTrackbarPos("V Min", "Trackbars")
        v_max = cv2.getTrackbarPos("V Max", "Trackbars")
        distancia_ab_cm = cv2.getTrackbarPos("Dist AB cm", "Trackbars")

        subtractor.setVarThreshold(max(sensibilidad, 1))

        k = max(1, kernel_size)
        if k % 2 == 0:
            k += 1
        kernel = np.ones((k, k), np.uint8)

        mascara_mog2_raw = subtractor.apply(frame_base)
        mascara_mog2 = cv2.threshold(mascara_mog2_raw, 200, 255, cv2.THRESH_BINARY)[1]

        mascara_hsv = np.zeros_like(mascara_mog2)
        if mezcla_hsv > 0:
            hsv = cv2.cvtColor(frame_base, cv2.COLOR_BGR2HSV)
            lower = np.array([h_min, s_min, v_min])
            upper = np.array([h_max, s_max, v_max])
            mascara_hsv = cv2.inRange(hsv, lower, upper)

        if mezcla_hsv == 0:
            mascara = mascara_mog2
        elif mezcla_hsv == 100:
            mascara = mascara_hsv
        else:
            mascara = cv2.bitwise_and(mascara_mog2, mascara_hsv)

        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
        mascara = cv2.dilate(mascara, kernel, iterations=1)

        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contorno_guardado = None
        cx_guardado = None
        cy_guardado = None

        if frame_num >= WARMUP_FRAMES:
            candidato = seleccionar_mejor_contorno(
                contornos,
                area_min,
                ultimo_centroide,
                frame_width,
                frame_height
            )

            if candidato is not None:
                contorno_guardado, cx_guardado, cy_guardado, _ = candidato

                tiempo_actual = frame_num / fps
                centroide_actual = (cx_guardado, cy_guardado)

                if ultimo_centroide is not None and ultimo_tiempo is not None:
                    dt = tiempo_actual - ultimo_tiempo
                    if dt > 1e-9:
                        dx = centroide_actual[0] - ultimo_centroide[0]
                        dy = centroide_actual[1] - ultimo_centroide[1]
                        velocidad_inst_px_s = float(np.hypot(dx, dy)) / dt
                else:
                    velocidad_inst_px_s = 0.0

                centroides.append(centroide_actual)
                tiempos.append(tiempo_actual)
                ultimo_centroide = centroide_actual
                ultimo_tiempo = tiempo_actual
                frames_sin_deteccion = 0
            else:
                frames_sin_deteccion += 1
                if frames_sin_deteccion > TRACK_LOST_FRAMES:
                    ultimo_centroide = None
                    ultimo_tiempo = None
                    velocidad_inst_px_s = 0.0

        distancia_ab_px = float(np.hypot(bx - ax, by - ay))
        if distancia_ab_px > 0 and distancia_ab_cm > 0:
            distancia_ab_m = distancia_ab_cm / 100.0
            px_por_m = distancia_ab_px / distancia_ab_m
            velocidad_inst_m_s = velocidad_inst_px_s / px_por_m
        else:
            distancia_ab_m = 0.0
            px_por_m = 0.0
            velocidad_inst_m_s = 0.0

        mascara_guardada = mascara.copy()
        mascara_mog2_guardada = mascara_mog2.copy()
        mezcla_hsv_txt = mezcla_hsv
        frame_num += 1

    if frame_base is None:
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            break
        continue

    frame2 = np.copy(frame_base)

    if mostrar_pixel:
        pixel_color = frame2[y, x]
        pixel_hsv = cv2.cvtColor(np.uint8([[pixel_color]]), cv2.COLOR_BGR2HSV)[0][0]
        cv2.putText(frame2, f"Clic: X={x}  Y={y}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(frame2, f"BGR: {pixel_color} | HSV: {pixel_hsv}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.drawMarker(frame2, (x, y), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

    centroide_detectado = False

    if contorno_guardado is not None and cx_guardado is not None and cy_guardado is not None:
        centroide_detectado = True

        x_bb, y_bb, w_bb, h_bb = cv2.boundingRect(contorno_guardado)
        cv2.rectangle(frame2, (x_bb, y_bb), (x_bb + w_bb, y_bb + h_bb), (255, 165, 0), 2)
        cv2.drawContours(frame2, [contorno_guardado], -1, (0, 255, 0), 2)
        cv2.circle(frame2, (cx_guardado, cy_guardado), 6, (0, 0, 255), -1)

        cv2.putText(frame2, f"Centroide: ({cx_guardado}, {cy_guardado})", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        area_txt = f"Area: {int(cv2.contourArea(contorno_guardado))} px"
        cv2.putText(frame2, area_txt, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    if len(centroides) >= 2:
        max_gap_t = MAX_GAP_FACTOR / fps
        for i in range(1, len(centroides)):
            p0 = centroides[i - 1]
            p1 = centroides[i]
            dt = tiempos[i] - tiempos[i - 1]
            salto = np.hypot(p1[0] - p0[0], p1[1] - p0[1])
            if dt <= max_gap_t and salto <= (MAX_DIST_SEGUIMIENTO_PX * 1.6):
                cv2.line(frame2, p0, p1, (255, 0, 255), 2)
    cv2.putText(frame2, f"Trayectoria: {len(centroides)} pts", (10, 225),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    if mostrar_marcadores:
        cv2.circle(frame2, (ax, ay), 8, (255, 255, 0), -1)
        cv2.circle(frame2, (bx, by), 8, (255, 255, 0), -1)
        cv2.putText(frame2, "A", (ax + 10, ay - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame2, "B", (bx + 10, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.line(frame2, (ax, ay), (bx, by), (0, 255, 255), 2)

    if float(np.hypot(bx - ax, by - ay)) > 0 and distancia_ab_cm > 0:
        texto_escala = f"Escala: {distancia_ab_m:.2f} m = {int(np.hypot(bx - ax, by - ay))} px ({px_por_m:.2f} px/m)"
        cv2.putText(frame2, texto_escala, (10, frame_height - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    estado = "Vehiculo detectado" if centroide_detectado else "Sin deteccion"
    color_estado = (0, 255, 0) if centroide_detectado else (0, 0, 255)
    cv2.putText(frame2, estado, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_estado, 2)

    tiempo_actual = (frame_num - 1) / fps if frame_num > 0 else 0
    cv2.putText(frame2, f"Frame: {frame_num} | t: {tiempo_actual:.2f}s", (10, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    modo_txt = "MOG2" if mezcla_hsv_txt == 0 else ("HSV" if mezcla_hsv_txt == 100 else "MOG2+HSV")
    cv2.putText(frame2, f"Modo: {modo_txt}", (10, 195),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.putText(frame2, f"Velocidad: {velocidad_inst_px_s:.2f} px/s | {velocidad_inst_m_s:.3f} m/s",
                (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)
    cv2.putText(frame2, "Teclas: a=marcar A, b=marcar B, m=ocultar/mostrar A-B",
                (10, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 255), 2)

    if modo_seleccion_marcador == "A":
        cv2.putText(frame2, "Seleccion A: haz clic sobre el cono derecho", (10, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    elif modo_seleccion_marcador == "B":
        cv2.putText(frame2, "Seleccion B: haz clic sobre el cono izquierdo", (10, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    if frame_num < WARMUP_FRAMES:
        cv2.putText(frame2, f"Inicializando fondo: {frame_num}/{WARMUP_FRAMES}",
                    (10, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    if pausar_video:
        cv2.putText(frame2, "|| PAUSADO", (10, 350),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    cv2.imshow("Video", frame2)

    if mascara_guardada is not None and mascara_mog2_guardada is not None:
        cv2.imshow("Mascara", mascara_guardada)
        cv2.imshow("MOG2 Raw", mascara_mog2_guardada)

    if exportar_video and video_writer is not None:
        video_writer.write(frame2)

    key = cv2.waitKey(30) & 0xFF

    if key == 27:
        break

    if key == ord('a'):
        modo_seleccion_marcador = "A"
        print("🟨 Selección A activada: haz clic en el cono derecho")

    if key == ord('b'):
        modo_seleccion_marcador = "B"
        print("🟨 Selección B activada: haz clic en el cono izquierdo")

    if key == ord('m'):
        mostrar_marcadores = not mostrar_marcadores
        print("👁️ Marcadores A/B visibles" if mostrar_marcadores else "🙈 Marcadores A/B ocultos")

    if key == ord('s'):
        print(f"\n--- Estado actual en frame {frame_num} ---")
        print(f"Centroides registrados: {len(centroides)}")
        if centroides:
            print(f"Último centroide: {centroides[-1]}")
            print(f"Velocidad instantánea: {velocidad_inst_m_s:.3f} m/s")

print("\n=== Procesamiento finalizado ===")
print(f"FPS del video: {fps:.2f}")
print(f"Frames procesados: {frame_num}")
print(f"Centroides detectados: {len(centroides)}")

if len(centroides) > 0:
    print(f"Primer centroide: {centroides[0]} en t={tiempos[0]:.3f}s")
    print(f"Último centroide: {centroides[-1]} en t={tiempos[-1]:.3f}s")

    datos_cinematicos = {
        "fps": float(fps),
        "resolucion": {
            "ancho": frame_width,
            "alto": frame_height
        },
        "total_frames": frame_num,
        "total_centroides": len(centroides),
        "calibracion_punto4": {
            "A_x_px": int(ax),
            "A_y_px": int(ay),
            "B_x_px": int(bx),
            "B_y_px": int(by),
            "distancia_AB_cm": int(distancia_ab_cm),
            "distancia_AB_m": float(distancia_ab_m),
            "distancia_AB_px": float(np.hypot(bx - ax, by - ay)),
            "px_por_m": float(px_por_m)
        },
        "centroides": [
            {"x": int(cx), "y": int(cy)}
            for cx, cy in centroides
        ],
        "tiempos": [float(t) for t in tiempos]
    }

    try:
        with open("datos_cinematicos.json", "w") as f:
            json.dump(datos_cinematicos, f, indent=4)
        print("\n✓ Datos guardados en: datos_cinematicos.json")
    except Exception as e:
        print(f"\n✗ Error al guardar JSON: {e}")
else:
    print("\n✗ No hay centroides para guardar")

if exportar_video and video_writer is not None:
    video_writer.release()
    print("✓ Video procesado guardado en: video_procesado_punto4.mp4")

cap.release()
cv2.destroyAllWindows()