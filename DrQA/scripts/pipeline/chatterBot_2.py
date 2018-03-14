# coding: utf-8
import copy
from flask import Flask, render_template, request
from interactive_gdpr import Process

app = Flask(__name__)

my_process = Process()


def process2 (question , 
              nombreReponse = 3) :
    dico = {'answer': "answer 1",
            'docid' : 215,
            'docscore' : 1.234,
            'answerscore': 12.345,
            'doc' : "Ceci est mon doc ",
            }
    r = []
    for i in range(0, nombreReponse) :
        d = copy.copy (dico)
        d["answer"] = "Answer "+str(i+1)
        d['docid'] += i+1
        r.append(d)
        continue
    return r
        

@app.route('/')
def student():
   return render_template('question.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      question = request.form ["Question"]
      
      reponse = my_process.process (question)
      resultat = [("Question" , question),]
      if len (reponse) > 0 :
            
          reponseId = 1
          for dico in reponse:
                resultat.append (   ('Answer '+str(reponseId)+"    " , "#"*120  ) )
                resultat.append (   ('Answer ' , str(dico ["answer"] ) ) )
                resultat.append (   ('Doc ID ' ,str (dico ["docid"])  ) )
                resultat.append (   ('Doc score ' ,str (dico ["docscore"])  ) )
                resultat.append (   ('Answer score ' ,str (dico ["answerscore"])  ) )
                resultat.append (   ('Document  ' ,str (dico ["doc"])  ) )
                reponseId += 1
                continue
      else:
        resultat.append (   ('Houps ' , "no answer !!!!!!!!!!!!!!!"  ) )
        
      
      return render_template("result.html",result = resultat)

if __name__ == '__main__':
    
    #print ("debut")
    app.run(host='54.37.20.249',port=5000,debug=True)
    
