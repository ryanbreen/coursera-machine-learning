
/*
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% 1/2m * sum((h-of-theta minus y^2))
*/

// Read in cost data

var linearAlgebra = require('linear-algebra')(),     // initialise it 
    Vector = linearAlgebra.Vector,
    Matrix = linearAlgebra.Matrix;

var LineByLineReader = require('line-by-line'),
    lr = new LineByLineReader('ex1data1.txt');

lr.on('error', function (err) {
  // 'err' contains error object
});

X = [];
Y = [];

lr.on('line', function (line) {
  // 'line' contains the current line without the trailing newline character.
  X.push(parseFloat(line.substring(0, line.indexOf(','))));
  Y.push(parseFloat(line.substring(line.indexOf(',')+1)));
});

lr.on('end', function () {
  // All lines are read, file is closed now.
  //console.log(X);
  //console.log(Y);

  var theta = [0, 0];

  var max = X.length;

  var squared_distance = 0;
  for (var i=0; i<max; ++i) {
    var delta = (theta[0] + theta[1] * X[i]) - Y[i];
    var delta1 = theta[0] + X[i];
    var delta2 = theta[0] + theta[1] * X[i];
    console.log("%s %s %s %s %s %s", X[i], Y[i], delta1, delta2, delta, Math.pow(delta, 2));
    //console.log("%s %s", delta, Math.pow(delta, 2));
    squared_distance += Math.pow(delta, 2);
  }

  console.log("sum: %s", squared_distance);
  console.log("cost: %s", (1/(2*max)) * squared_distance);


  // Vector add theta0

  console.log(Vector);

  var m = new Vector(X);
  m = m.addEach(theta[0]);

  // Vector multiply theta1
  m = m.mulEach(theta[1]);

  // Vector subtract y
  m = m.sub(y);

  // Square vector
  m = m.mul(m);

  console.log(m.getSum());
});