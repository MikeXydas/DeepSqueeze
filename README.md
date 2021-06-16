# DeepSqueeze  
  
In this repo I attempt to reproduce the compression utility described in *[DeepSqueeze: Deep Semantic Compression for  
Tabular Data](https://cs.brown.edu/people/acrotty/pubs/3318464.3389734.pdf)* by *Amir Ilkhechi, Andrew Crotty,  
 Alex Galakatos, Yicong Mao, Grace Fan, Xiran Shi, Ugur Cetintemel* from Brown University.  
   
 You can read the `report.pdf` for info about the original paper, my implementation, my additions and  
 results.  
   
 There are 3 branches in this repo:  
 * `master`, used in my demo presentation  
 * `experiment`, mainly used to run experiments and producing results  
 * `mixture_of_experts`, since I was not able to achieve better results using   
 the Mixture of Experts architecture, I decided to keep it into a separate branch   
 reducing code complexity of the `master` and `demo` branches  
   
 ## Running  
 1. Create a python environment. The DeepSqueeze package was developed in `python3.8`  
 2. Install the requirements in `requirements.txt`  
 3. Download the processed tables (no header, only numerical values).  
    * Corel dataset  
    * Intel Berkeley Research Lab Sensor Data  
    * Monitor dataset, due to to size I have not uploaded its preprocessed version.  
    You can download it here and preprocess it using `notebooks/preprocessing.ipynb`  
 4. Compress the table with the command:   
   
 `python compress.py -i path/to/input/ -o path/to/output/ -e <error_threshold_percentage>`  
  Note that the `-e` parameter takes a value between `[0, 100]` with suggested values being:  
 `0.5, 1, 5, 10`.  
   
 5.Decompress the table with the command:  
   
 `python decompress.py -i path/to/compressed_tables.zip`  
  This is the full pipeline of DeepSqueeze with some simplification   
 presented below.