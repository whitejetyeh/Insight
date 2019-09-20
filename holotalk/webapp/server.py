from flask import Flask, render_template, request
from holotalk import webcam_capture

# Create the application object
app = Flask(__name__)

@app.route("/")
def home_page():
	return render_template('index.html')

# start the server with the 'run()' method
if __name__ == "__main__":
	#webcam_capture.capture_face(cropscale=1.3)
	app.run(debug=True) #will run locally http://127.0.0.1:5000/
