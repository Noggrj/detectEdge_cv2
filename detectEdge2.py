import cv2
import numpy as np

# Configura a câmera
cap = cv2.VideoCapture(0)

while True:
    # Captura a imagem da câmera
    ret, frame = cap.read()

    # Converte a imagem para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplica um filtro gaussiano para suavizar a imagem e reduzir o ruído
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Aplica o filtro de Canny para detectar as bordas
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)

    # Dilata as bordas para conectar regiões próximas
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated = cv2.dilate(edges, kernel)

    # Encontra os contornos na imagem
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Loop sobre todos os contornos encontrados
    for cnt in contours:
        # Aproxima o contorno para um polígono
        approx = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)

        # Verifica se o polígono tem 4 lados (um retângulo)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            # Verifica se a área do retângulo é suficientemente grande
            area = cv2.contourArea(approx)
            if area > 1000:
                # Encontra os ângulos dos lados do retângulo
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)

                # Desenha um retângulo em volta do contorno identificado
                cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

    # Exibe a imagem resultante na tela
    cv2.imshow('frame', frame)

    # Verifica se o usuário pressionou a tecla 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos da câmera e fecha a janela
cap.release()
cv2.destroyAllWindows()