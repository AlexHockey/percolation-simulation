# Summary

Program that simulates a forest fire on a 2D rectangular grid. 

Each cell in the grid starts as either empty or has a tree, based on a "density" parameter in the range [0, 1.0]. 
The fire is started by setting fire to each tree on the first row. 
On each iteration of the simulation, each burning tree sets fire to its neighbours and then burns out, becomming empty. 
If the fire reaches the final row, the fire has "percolated". 
The probability the fire percolates for a given density is the percolation probability. 

This simulation is a form of phase transition. There is a density threshold, where:
* Below the threshold, the percolation probability is close to 0.
* Above the threshold, it's close to 1.0.
When close the percolation threshold, the percolation probability changes rapidly from 0 to 1. 

We want to find the percolation threshold.

# Build and run 

Build the code with 
```
cargo build --release
```

See the command line options with 
```
cargo run --release -- -h
```

Run a simulation with 
```
cargo run --release -- --rows=100 --columns=100
```