## Examples

These jupyter notebooks implement examples on how the library works.

* `1d_example` is a simple one-dimensional example on how to use the library.
* `2d_example` extends this to two-dimensional parameters, which is not very different except for the plotting.
* `1d_multiple_constraints_example` shows how multiple constraints can be incorporated into the library.
* `context_example` shows how contexts, variables that impact the resulting performance, but are fixed by the environment (e.g., temperature, wind) rather than the user, can be included with the library.

## Usage

Clone the Repository and move to the examples folder. Launch a `jupyter-notebook` instance within this folder. In the browser, select the notebook that you want to run and evaluate all cells. 

## Structure

These notebooks all follow the same structure. In the first cell, the library is imported, along with other relevant libraries.
The second cell defines the Gaussian process model over functions and specifies the optimization problem (by sampling a function from the Gaussian process distribution). The last step is to plot the GP and to run the safeopt optimization routine to obtain a new data point.

