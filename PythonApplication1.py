import os
import tensorflow as tf
from flask import (Flask, redirect, render_template, request,
                   send_from_directory, url_for)

UPLOAD_FOLDER = './static/datastorage'
ALLOWED_EXTENSIONS = set(['pdf', 'png', 'jpg', 'jpeg'])
name = "PythonApplication1.py"

app = Flask(name)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def start_():
   return render_template('home.htm')

@app.route('/home.htm')
def return_home():
   print('enter the home page received')
   return render_template('home.htm')

@app.route('/second.htm',methods=['POST','GET'])
def initial():  
    if request.method == "POST":
        file = request.files["my-local-file"]
        if file and allowed_file(file.filename):
            filename = file.filename    #secure_filename()

            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("savefile at:")
            print(filename)
            return render_template("second.htm",filename = filename)
        
    else:
       print('enter the second page received')
       return render_template('second.htm')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS  

@app.route('/second.htm/<filename>',methods=['POST','GET'])
def get_body_img(filename):
    print("get file, present on page")
    return send_from_directory(app.config['UPLOAD_FOLDER'],filename)

@app.route('/about_us.htm', methods=['POST','GET'])
def about_us():
   print('enter the about us page received')
   return render_template('about_us.htm')

@app.route('/language.htm', methods=['POST','GET'])
def switch_language():
   print('enter the language page received')
   return render_template('language.htm')

@app.route('/upload_1.htm', methods=['POST','GET'])
def choose_upload():
    print('enter the upload_1 page received')
    return render_template('upload_1.htm')



if __name__ == '__main__':
   app.run()

