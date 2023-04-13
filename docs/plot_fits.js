Plotly.d3.csv('static/fits_plot.csv', function(data){ 
    var x = [], y = [];
    for(var i=0; i<data.length; i++) {
        x.push(data[i]['x']);
        y.push(data[i]['y']);
    }
    var trace1 = {
        x: x, 
        y: y, 
        mode: 'markers', 
        type: 'scatter'
    };
    var data = [trace1];
    var layout = {
        title:'FITS File Plot', 
        xaxis: {
            title: 'X Axis'
        },
        yaxis: {
            title: 'Y Axis'
        }
    };
    Plotly.newPlot('fits_plot', data, layout);
});
