import cv2
import numpy as np
import pytesseract

def encontrarPlaca(source):
    imagem = cv2.imread(source)

    cv2.imshow('teste', imagem)

    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)


    filtro = cv2.bilateralFilter(cinza, 11, 17, 17)
    desenho = cv2.Canny(filtro, 30, 200)

    cv2.imshow('deseho', filtro)

    contornos, _ = cv2.findContours(desenho.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)


    areaFoco= None
    for contorno in contornos:
        perimetro = cv2.arcLength(contorno, True)
        aprox = cv2.approxPolyDP(contorno, 0.015 * perimetro, True)
        if len(aprox) == 4:
            areaFoco = aprox
            break

    areaFoco = np.array([areaFoco], np.int32)
    pontos = areaFoco.reshape(4,2)

    x, y = np.split(pontos, [-1], axis=1)

    (x1, x2) = (np.min(x), np.max(x))
    (y1, y2) = (np.min(y), np.max(y))

    placa = imagem[y1+5:y2, x1+5:x2]

    cv2.imwrite('placas/placa.jpg', placa)

def processarPlaca():

    placa = cv2.imread('placas/placa.jpg')
    
    placaResize = cv2.resize(placa, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    placaCinza = cv2.cvtColor(placaResize, cv2.COLOR_BGR2GRAY)

    placaFiltro = cv2.medianBlur(placaCinza, 5)

    _, bin = cv2.threshold(placaFiltro, 90, 255, cv2.THRESH_BINARY)

    return(bin)

def lerPlaca():
    
    bin = processarPlaca()

    pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

    config = r'-c tessedit_char_whitelist=ABCDEFGHIJGKLMNOPQRSTUVWXYZ1234567890 --psm 6'

    texto = pytesseract.image_to_string(bin, lang='eng', config= config)
    print(texto)
    cv2.imshow('binchiling', bin)

if __name__ == '__main__':

    source = 'imgs/placa_carro2.jpg'

    # encontrarPlaca(source)

    # processarPlaca()

    lerPlaca()
    
    if cv2.waitKey(0) & 0xFF == 27:
        cv2.destroyAllWindows

