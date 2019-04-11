--------------
About Anaconda
--------------

Anaconda is a free and open-source distribution of the Python and R programming languages for scientific computing used in various fields of data science, machine learning, large-scale data processing, predictive analytics, etc. Package versions are managed by the package management system named as "conda". The Anaconda distribution is used by over 6 million users and includes more than 1400 popular data-science packages suitable for Windows, Linux, and MacOS.

Link to install Anaconda: https://www.anaconda.com

------------------------------
Installing Module Dependencies
------------------------------

numpy

Note: For Windows system, you can use "pip install <package_name>". Example: "pip install numpy"

----------------------------------------------------------------------------------
Required files/folders to run the source codes "NB.py"/"NBSW.py"/"LR.py"/"LRSW.py"
----------------------------------------------------------------------------------

1. Copy the folders "hw2_train" and "hw2_test" and these files "NB.py"/"NBSW.py"/"LR.py"/"LRSW.py" to the location where you want to run the "NB.py" as first source code.
2. From the current folder, enter the below commands to run the program.

----------------------
Command Line Arguments
----------------------

Usage: Python 3.7.2

Example:  python3 NB.py <train folder path> <test folder path>

Options for the above argument values:

<train folder path> - "./hw2_train/train"
<test folder path> - "./hw2_test/test"
<lambda-value> - "0.001"
NB - indicates not using stopwordslist
NBSW - indicates using stopwordslist
LR - indicates not using stopwordslist
LRSW - indicates using stopwordslist

1. python3 NB.py <train folder path> <test folder path>

Example: python3 NB.py ./hw2_train/train ./hw2_test/test

2. python3 NBSW.py <train folder path> <test folder path>

Example: python3 NBSW.py ./hw2_train/train ./hw2_test/test

3. python3 LR.py <train folder path> <test folder path> <λ-value>

Example: python3 LR.py ./hw2_train/train ./hw2_test/test 0.001

4. python3 LRSW.py <train folder path> <test folder path> <λ-value>

Example: python3 LRSW.py ./hw2_train/train ./hw2_test/test 0.001

------------------------------------------------------------------------------------
List of import packages used to include certain functions used in the programs
------------------------------------------------------------------------------------

import os
import re
import sys
import glob
import time
import math
import string
from numpy import *

-----------------------------------------------------------
List of functions implemented in "NB.py" program
-----------------------------------------------------------

1. main
2. loadTrainingData
3. countUniqueWords
4. calcCondProb
5. classifyLabel

-----------------------------------------------------------
List of functions/classes implemented in "NBSW.py" program
-----------------------------------------------------------

1. main
2. getAllStopWords
3. loadTrainingData
4. countUniqueWords
5. calcCondProb
6. classifyLabel

-----------------------------------------------------------
List of functions/classes implemented in "LR.py" program
-----------------------------------------------------------

1. main
2. readAllFiles
3. setLabels
4. attributeValue
5. sigmoid
6. updateWeights
7. classifyLabel

-----------------------------------------------------------
List of functions/classes implemented in "LRSW.py" program
-----------------------------------------------------------

1. main
2. getAllStopWords
3. readAllFileStopWords
4. setLabels
5. attributeValue
6. sigmoid
7. updateWeights
8. classifyLabel
