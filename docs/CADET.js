// import { Fits } from 'astrojs';

// Load the FITS file
fetch('NGC4649_512.fits')
  .then(response => response.arrayBuffer())
  .then(buffer => new Fits(buffer))
  .then(fits => {
    // Access the FITS header and data
    const header = fits.getHeader();
    const data = fits.getData();
    console.log(header, data);
  })
  .catch(error => console.error(error));

// // Import the required libraries
// import * as astrojs from 'astrojs';
// import { createCanvas } from 'canvas';

// // Load the FITS file
// const fits = new astrojs.FITS('docs/NGC4649.fits');
// const image = fits.getHDU(0).getImage();

// // Create a canvas to draw the image on
// const canvas = createCanvas(image.width, image.height);
// const context = canvas.getContext('2d');

// // Draw the image on the canvas
// const imageData = context.createImageData(image.width, image.height);
// imageData.data.set(image.pixelData);
// context.putImageData(imageData, 0, 0);

// // Append the canvas to the DOM
// document.getElementById('myChart').appendChild(canvas);
// // document.body.appendChild(canvas);

// // // Load the required libraries
// // const fits = require('fitsjs');
// // const tf = require('@tensorflow/tfjs-node-gpu');

// // // Load the FITS file
// // const fitsData = fits.read('NGC4649_512.fits');

// // // Reshape the data to match the input shape of the Keras model
// // const inputData = tf.tensor(fitsData.data).reshape([1, 128, 128, 1]);

// // // Load the Keras model
// // const model = await tf.loadLayersModel('CADET.json');

// // // Apply the model to the input data
// // const outputData = model.predict(inputData);

// // // Convert the output data to a 2D array
// // const outputArray = await outputData.array();

// // // Plot the resulting 128x128 image
// // const plotData = {
// //   z: outputArray[0].map(row => row.map(val => val * 255)),
// //   type: 'heatmap',
// //   colorscale: 'Viridis'
// // };

// // Plotly.newPlot('plotDiv', [plotData]);

