require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');
const LinearRegression = require('./linear-regression');
const plot = require('node-remote-plot');

let { features, labels, testFeatures, testLabels } = loadCSV(
  './cars.csv',
  {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower', 'weight', 'displacement'],
    labelColumns: ['mpg']
  }
);

const regression = new LinearRegression(features, labels, {
  learningRate: .01,
  iterations: 50
});

regression.train();
const r2 = regression.test(testFeatures, testLabels);

plot({
    x: regression.mseHistory.reverse(),
    xLabel: 'Iteration #',
    yLabel: 'Mean Squared Error'
  });

console.log('R2 is', r2);