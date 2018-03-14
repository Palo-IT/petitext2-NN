# coding: utf-8
from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')
def student():
   return render_template('question.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      resultat = request.form
      return render_template("result.html",result = resultat)

if __name__ == '__main__':
    
    #print ("debut")
    app.run(debug = True)
    
