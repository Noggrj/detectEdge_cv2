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
    edges = cv2.Canny(blur, 100, 200)
    
    # Encontra os contornos na imagem
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Loop sobre todos os contornos encontrados
    for cnt in contours:
        # Aproxima o contorno para um polígono
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        
        # Verifica se o polígono tem 4 lados (um retângulo)
        if len(approx) == 4:
            # Desenha um retângulo em volta do contorno identificado
            cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)
    
    # Exibe a imagem resultante na tela
    cv2.imshow('frame', frame)
    
    # Verifica se o usuário pressionou a tecla 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# Libera os recursos da câmera e fecha a janela
cap.release()
cv2.destroyAllWindows()
