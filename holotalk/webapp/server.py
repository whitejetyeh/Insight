from flask import Flask, flash, request, redirect, render_template
import os
import urllib.request
from werkzeug.utils import secure_filename
import holotalk.load_for_reconstruct as lforr
import holotalk.webcam_capture as webcam

#import sys
## insert at 1, 0 is the script path (or '' in REPL)
#sys.path.insert(1, '/holotalk')
#import load_for_reconstruct as lforr

UPLOAD_FOLDER = './static/img/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Create the application object
app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.route("/")
def home_page():
	return render_template('index.html')

@app.route('/upload_check', methods=['POST'])
def upload_file():
	if request.method == 'POST':
        # check if the post request has the file part
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		if file.filename == '':
			flash('No portrait selected for uploading')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'photo.jpg'))
			flash('File successfully uploaded')
			return redirect('/')
		else:
			flash('Allowed file types are png, jpg, and jpeg')
			return redirect(request.url)
@app.route("/generation", methods=['POST'])
def output1():
	(roll, pitch, yaw) = lforr.sideview_generation()
	return render_template('demo_page_1.html',roll=roll,pitch=pitch,yaw=yaw)
@app.route("/3dgeneration", methods=['POST'])
def output2():
	return render_template('demo_page_2.html')
@app.route("/take_a_pic", methods=['POST'])
def shoot():
	webcam.take_pic()
	return redirect('/')


# start the server with the 'run()' method
if __name__ == "__main__":
	#webcam_capture.capture_face(cropscale=1.3)
	app.run(debug=True) #will run locally http://127.0.0.1:5000/
