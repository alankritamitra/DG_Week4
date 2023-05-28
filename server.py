import model 
from flask import Flask, request, render_template 

# Initialize the flask class and specify the templates directory
app = Flask(__name__,template_folder="templates")

# Default route set as 'home'
@app.route('/home')
def home():
    return render_template('homepage.html') 


@app.route('/classify',methods=['POST','GET'])
def classify_type():
    try:
        sepal_len = request.args.get('slen') 
        sepal_wid = request.args.get('swid') 
        petal_len = request.args.get('plen') 
        petal_wid = request.args.get('pwid') 

        # Get the output from the classification model
        variety = model.classify(sepal_len, sepal_wid, petal_len, petal_wid)

        # Render the output in new HTML page
        return render_template('output.html', variety=variety)
    except:
        return 'Error'

# Run the Flask server
if(__name__=='__main__'):
    app.run(debug=True)        