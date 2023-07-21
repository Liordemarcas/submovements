# submovements
Welcome!!!
This is a hackathon project for Python for Neuroscience course at TAU, working on prof. Jason Friedman project.\
This code is for decomposition movement velocity data into submovements.\
Right now only 2d movements are implemented.

# To install
git clone this repository to your wanted location.\
To run this module, you will need the following packages:\
numpy, scipy, matplotlib, os & re.\
\
In order to run the example script, you will also need jupiter notebook in you environment.\
\
In the folder "conda_env" there is a file that demonstrate an environment this module was developed & run on.\
You can import this environment by entering this folder, and running:\
```
    conda env create -n submovements --file environment.yml
```

# To run
import the module "movement_decompose_2d".\
In it:\
    load the data using "load_data"\
    plot movements using "plot_position" or "plot_velocity"\
    decompose to submovments using "decompose_2D"\
    plot the submovments using "plot_submovements_2D".\
\
You can see a full example run in "run_example.ipynb".
# Contributors
Omer Ophir:             https://github.com/omerophir\
Omri FK:                https://github.com/OmriFK\
Shay Eylon:             https://github.com/ShayEylon\
Lior de Marcas (LdM):   https://github.com/Liordemarcas\
\
Code is a python import of the work done by prof. Jason Friedman (https://github.com/JasonFriedman), from Tel Aviv University.\
You can find the origin at https://github.com/JasonFriedman/submovements/tree/master.\
\
This project is incomplete! Feel free to contribute.