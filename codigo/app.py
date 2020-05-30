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
		

def removedot(invertThin):
    temp0 = numpy.array(invertThin[:])
    temp0 = numpy.array(temp0)
    temp1 = temp0/255
    temp2 = numpy.array(temp1)
    temp3 = numpy.array(temp2)

    enhanced_img = numpy.array(temp0)
    filter0 = numpy.zeros((10,10))
    W,H = temp0.shape[:2]
    filtersize = 6

    for i in range(W - filtersize):
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

    return temp2


def get_descriptors(img):
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	img = clahe.apply(img)
	img = image_enhance.image_enhance(img)
	img = numpy.array(img, dtype=numpy.uint8)
	# Threshold
	ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
	# Normalize to 0 and 1 range
	img[img == 255] = 1

	#Thinning
	skeleton = skeletonize(img)
	skeleton = numpy.array(skeleton, dtype=numpy.uint8)
	skeleton = removedot(skeleton)
	# Harris corners
	harris_corners = cv2.cornerHarris(img, 3, 3, 0.04)
	harris_normalized = cv2.normalize(harris_corners, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
	threshold_harris = 125
	# Extract keypoints
	keypoints = []
	for x in range(0, harris_normalized.shape[0]):
		for y in range(0, harris_normalized.shape[1]):
			if harris_normalized[x][y] > threshold_harris:
				keypoints.append(cv2.KeyPoint(y, x, 1))
	# Define descriptor
	orb = cv2.ORB_create()
	# Compute descriptors
	_, des = orb.compute(img, keypoints)
	return (keypoints, des);

@app.route('/api/v1/procesar',methods=['GET','POST'])
def procesar():
	if request.method=="POST":
		imagen= request.get_json()
		img=imagen['name']

		for dirName, subdirList, fileList in os.walk(rootDir):
			for index in range(2):
				print('\t%s' % fileList[index])
				image_name = img
				image_namedos=fileList[index]
				img1 = cv2.imread("muestra/" + image_name, cv2.IMREAD_GRAYSCALE)
				kp1, des1 = get_descriptors(img1)
				#print(image_namedos)
				img2 = cv2.imread("database/" + image_namedos, cv2.IMREAD_GRAYSCALE)
				kp2, des2 = get_descriptors(img2)
				# Matching between descriptors
				bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
				matches = sorted(bf.match(des1, des2), key= lambda match:match.distance)
				# Calculate score
				score = 0;
				for match in matches:
					score += match.distance
				score_threshold = 1
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