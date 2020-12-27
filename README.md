# ml from scratch

using this tut for refernce, been doing most on my own
https://www.youtube.com/watch?v=lGLto9Xd7bU&ab_channel=sentdex

ml from scratch in nearly pure python

TODO:

- use nupmy operations to get inputs (dot product)
- make typesafe with errors
- implement loss on data (try the mnist set?)
- implement backpropogation
 - need to do more research in this area
- implement a network method to save the weight and bias state
 - array of arrays of weights with last element bias? first elem bias?
- refactor to get rid of the node class, this can be done in much simpler fashion
 - use the layer to hold array of weights and biases

how to update the weights and biases?
- loss needs to be calculated to see what kinds of changes need to happen
- network needs access to all weights and biases in every layer
 - how to represent this?


