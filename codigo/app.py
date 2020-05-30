from flask import Flask,render_template,jsonify,request
import cv2
import os
import sys
import numpy
import matplotlib.pyplot as plt
from enhance import image_enhance
from skimage.morphology import skeletonize, thin

rootDir = os.getcwd() + '\database'
data = {}
data['resultados'] = []

app = Flask(__name__)
app.secret_key="MY_SECRET_KEY"
#configura ruta de descargas

app.config['UPLOAD_FOLDER']="muestra/"

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/api/v1/subir',methods=['GET','POST'])
def subir():
	if request.method=="POST":
		f=request.files["myfile"]
		filename=f.filename
		f.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
		#procesar(filename)
		data={"nombre":filename}
	return jsonify(data)
		
#Remover puntos 
def removedot(invertThin):
	#Convertimos en array invertThin
    temp0 = numpy.array(invertThin[:])
    temp0 = numpy.array(temp0)
    temp1 = temp0/255
    temp2 = numpy.array(temp1)
    temp3 = numpy.array(temp2)

 	#Convertimos en array temp0
    enhanced_img = numpy.array(temp0)
	#Creamos una array de 10 filas y columnas
    filter0 = numpy.zeros((10,10))
	#Obtenemos las filas y columnas de temp0
    W,H = temp0.shape[:2]
    filtersize = 6

	#Limpiamos los puntos del array que salgan como 0
	#Restamos las filas - 6
    for i in range(W - filtersize):
		#Restamos las columnas - 6
        for j in range(H - filtersize):
            filter0 = temp1[i:i + filtersize,j:j + filtersize]

            flag = 0
            if sum(filter0[:,0]) == 0:
                flag +=1
            if sum(filter0[:,filtersize - 1]) == 0:
                flag +=1
            if sum(filter0[0,:]) == 0:
                flag +=1
            if sum(filter0[filtersize - 1,:]) == 0:
                flag +=1
            if flag > 3:
                temp2[i:i + filtersize, j:j + filtersize] = numpy.zeros((filtersize, filtersize))
	#Retornamos la nueva imagen limpiada de 0 os
    return temp2

#PROCESAR IMAGEN
def get_descriptors(img):
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	img = clahe.apply(img)
	#	LIMPIAMOS LA IMAGEN DE RUIDO MEDIANTE UNA LIBRERIA IMAGE_ENHANCE
	img = image_enhance.image_enhance(img)
	img = numpy.array(img, dtype=numpy.uint8)
	# ESCOGEMOS EL MEJOR UMBRAL PARA LA IMAGEN
	ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
	img[img == 255] = 1

	# ESQUELETIZAR LA HUELLA PARA TENER MEJOR RESOLUCION
	skeleton = skeletonize(img)
	skeleton = numpy.array(skeleton, dtype=numpy.uint8)
	skeleton = removedot(skeleton)
	# DETECTAMOS LOS PUNTOS CRITICOS CON HARRIS
	harris_corners = cv2.cornerHarris(img, 3, 3, 0.04)
	harris_normalized = cv2.normalize(harris_corners, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
	threshold_harris = 125
	# EXTRAEMOS LOS PUNTOS CLAVES
	keypoints = []
	for x in range(0, harris_normalized.shape[0]):
		for y in range(0, harris_normalized.shape[1]):
			if harris_normalized[x][y] > threshold_harris:
				keypoints.append(cv2.KeyPoint(y, x, 1))
	# DEFINIMOS UN DESCRIPTOR
	orb = cv2.ORB_create()
	# CALCULAR DESCRIPTOR
	_, des = orb.compute(img, keypoints)
	return (keypoints, des)

@app.route('/api/v1/procesar',methods=['GET','POST'])
def procesar():
	if request.method=="POST":
		imagen= request.get_json()
		img=imagen['name']

		#RECORREMOS TODAS LAS FOTOS 
		#EN ESTE CASO RECORREMOS 7 ARCHIVOS
		for dirName, subdirList, fileList in os.walk(rootDir):
			for index in range(7):
				print('\t%s' % fileList[index])
				image_name = img
				image_namedos=fileList[index]
				#COMPARANDO LA IMAGEN SELECCIONADA CON OTRA CON OTRA
				img1 = cv2.imread("muestra/" + image_name, cv2.IMREAD_GRAYSCALE)
				kp1, des1 = get_descriptors(img1)
				img2 = cv2.imread("database/" + image_namedos, cv2.IMREAD_GRAYSCALE)
				kp2, des2 = get_descriptors(img2)
				# IGUALDADES ENTRE DESCRIPTORES
				# APLICANDO HAMMING OBTENER LA DIFERENCIA ENTRE DOS PUNTOS
				bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
				matches = sorted(bf.match(des1, des2), key= lambda match:match.distance)
				# CALCULAR 
				score = 0
				for match in matches:
					score += match.distance
				#UMBRAL DE ACIERTOS TIENE QUE SER MENOR A 0 PARA QUE SEA VALIDO
				score_threshold = 1
				#CREAMOS UN OBJETO JSON
				if score/len(matches) < score_threshold:
					print("Fingerprint matches.")
					data['resultados'].append({'Nombre':image_namedos,"usuario":image_name,'resultado':'Las huellas coinciden',"estado":1})
					return jsonify(data)
				else:
					print("Fingerprint does not match.")
					data['resultados'].append({'Nombre':image_namedos,"usuario":image_name,'resultado':'Las huellas no coinciden',"estado":0})
	return jsonify(data)


@app.route('/api/v1/resultado',methods=['GET','POST'])
def main():
	return "algo"


if __name__=='__main__':
	app.run(port=3000,debug=True)