# Old unfinished toy project started during undergraduate;

# The Development of A Prerdictive Model
# Part 1: ASG Transform

This repository contains the infrastructure used to develop the results covered in the corresponding research paper titled "Algorithmic Fixed Income Trading on Macro-Financial Indicators Part 1: Training Sets & ASG Transforms",
which can be found on ResearchGate.

## TODO
* Remove windows dependent packages such as win32com etc
* Test on linux
* Clean code

## Getting Started

### Prerequisites

Jupytr Notebook
Python 3+
R
Windows operating system. ( win32Com is used to control Excel)
Specific python packages to install
-win32com, Sknn, theanom


# Structure
The project uses a structure wherein all jupytr notebooks, produce an identical python script to be imported into another notebook or pythons script. Furthermore, code is divided into two sections API and Presentation.
Functions and procedures are in the API. The Presentation section provides a presentable format to execute the code. Often many of the charts and graphs can be viewed directly within the notebooks. 
As such users can simply use the Presentation notebooks in order to review results, adjusting essential parameters as needed. Users interested in the underlying methodoly should view the functions defined in the API section.

# Deployment

### Answering hypothesis questions
Method 1
-Open the jupytr notebook file titled "research.ipynb"
-Run the first two first two cells, ignoring any output
-Run any subsequent cell, relevant to the hypothesis question
Method 2
-import bond_price_pred.py or run the bond_price_pred.ipynb file.
-run the functions in the appropriate order from command line or a suitable IDE

### Investigating Data retreival
for functions and procedures
- In the api folder open "api_data_retreival.ipynb" or "api_data_retreival.py"
- In python Scripts folder "bond_price_data_prep.ipynb" imports the above folders and initiates the data downloading.


### Investigating Meta Learning algorithm development
for functions and procedures
-In the api folder open "api_algorithm.ipynb"
for an overview 
-In the Scripts folder open one of the following 
               - bond_price_pred to build an algorithm with all features
               - bond_price_pred_excl_MP4 to build an algorithm with all feaatures except MP4
               - bond_price_pred_only_MP4 to build an algorithm that only uses MP4



## Authors

* **Rilwan Adewoyin** - *The Development of a Predictive Model* - [Part 1] (https://github.com/Akanni96)


## Acknowledgments

* 
* 
When developing the Meta Learning Algorithm, you must be sure to use an adaptation of the mlxtend package, specifically one py file must be edited.
The replacement for the file “ensemble_vote.py” is located in this github repository. Copy this into the mlxtend package to replace the old code. Alternatively, 
the edit required is in line 194 of ensemble_vote.py located at mlxtend/classifier/. The code np.bincount(x,weights=self.weights) should be changed to np.bincount(x.astype(int), weights=self.weights). 
This explicit type casting ensures that this method works with classifiers which are not part of scikit learn. For example the neural network used in my script is from the sknn library, and causes a ValueError, without this edit. 
Secondly, line 131 must have “if self.weights and len(self.weights) changed to “if self.weights.any() and len(self.weights)”
