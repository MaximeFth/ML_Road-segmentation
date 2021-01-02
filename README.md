# Project Road Segmentation
The goal of the project is to perform road segmentation on satellite images by assigning each pixel either label 1 if it is a road or 0 otherwise.  For this project, we use a U-net model implemented in tensorflow.
 We achieved 87% of F1-Score.



 # Python and external libraries

This project has been developped using `Python 3.6`.
We use the following libraries with the mentionned version.

`tensorflow '2.3.1'
 Sklearn : '0.23.2', 
 Numpy : '1.19.4', 
 Keras : '2.4.3'
 Skimage: '0.17.2'`

 `'pip3 install -r requirements.txt'`



 
 # Project structure
<pre>
.
├── README.md
│                           
├── data            
│   └── images                          Satellite images
│   └── labels                          Satellite groundtruths
│   └── test                            Test images
│  
├── report                              Final report
│   └── final_report.pdf
│ 
├── src                                 Source folder
│   └── unet.py
│   │ 
│   └── helpers.py
│   │ 
│   ├── Notebooks
│   │   ├── Main_notebook.ipynb  
│   │   └── Fastai-test.ipynb  
│   │
│   └── smooth_tiled_predictions.py
│   │ 
│   └── <strong>run.py</strong>
│   
├── models                              Models folder                               
│ 
│         
├── requirements.txt    				                              
│ 
└── submissions                         Submission folders
</pre>



 # run.py usage & Best AIcrowd submission
<pre>
In order to obtain the best submission please launch:   <strong># cd src </strong>
							<strong># python3 run.py </strong>

python run.py -h
usage: run.py [-h] [-model MODEL_FILENAME] [-train TRAIN] [-epochs EPOCH_NUMBER] 
              [-steps STEPS_PER_EPOCH] [-loss LOSS] [-batch BATCH_SIZE]
              [-sub SUBNAME]

optional arguments:
  -h, --help                    Show this help message and exit
  -model MODEL_FILENAME         Name of the model
  -train                        Train model from scratch
  -norm                         Normalize the data
  -epochs EPOCH_NUMBER          Number of epoch
  -steps STEPS_PER_EPOCH        Number of steps per epoch
  -loss LOSS                    Loss used in training
  -batch BATCH_SIZE             Batch size
  -images IMAGES_GEN_NUMBER     Number of new images generated per source image (40 will result in 4000 train images)
  -imFolder NEW_IMAGES_FOLDER   Name of new images folder
  -sub SUBNAME                  Submission file name

example which gave our model with best submission: 
<strong>python3 run.py -train -model bestModel -epochs 200 -loss binary_ce
 -batch 1 -images 90 -imFolder best_Dataset -sub best_submission</strong>
</pre>


 
  # Contributors

- Yura Tak [@YuraTak](https://github.com/yuratak)
- Martin Cibils [@MartinEpfl](https://github.com/MartinEpfl)
- Maxime Fellrath [@MaximeFth](https://github.com/MaximeFth)
