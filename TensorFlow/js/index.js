const features = tf.tensor([
    [-121, 47],
    [-121.2, 46.5],
    [-122, 46.5],
    [-120.9, 46.7]
]);

const labels = tf.tensor([
    [200],
    [250],
    [215],
    [240]
]);

const predictonPoint = tf.tensor([-121, 47]);

const k = 2

features
.sub(predictonPoint)
.pow(2)
.sum(1)
.pow(.5)
.expandDims(1)
.concat(labels, 1)
.unstack()
.sort((a, b) => a.get(0) > b.get(0) ? 1 : -1)
.slice(0,k)
.reduce((acc, pair) => acc + pair.get(1), 0 ) / k