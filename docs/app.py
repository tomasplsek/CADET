from flask import Flask, request, render_template
import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    fits_file = request.files['fits_file']
    hdulist = fits.open(fits_file)
    data = hdulist[0].data
    plt.imshow(np.log10(data), cmap='gray')
    plt.savefig('static/fits_plot.png')
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
