from flask import Flask, render_template, request

import os
import base64
import yolov_main


app = Flask(__name__)

IMG_PATH = os.path.join(app.root_path, 'static/img')


@app.route('/')
def index():  # put application's code here
	return render_template('index.html')


@app.route('/detector', methods=['POST'])
def upload():
	f = request.files['file']
	upload_path = os.path.join(IMG_PATH, f.filename)
	f.save(upload_path)
	output_img = yolov_main.main(upload_path)

	img_stream = ''
	try:
		with open(output_img, 'rb') as img:
			img_stream = base64.b64encode(img.read())
	except:
		pass
	return img_stream


if __name__ == '__main__':
	app.run()
