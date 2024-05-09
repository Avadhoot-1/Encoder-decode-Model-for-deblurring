# Code Running Instructions ##
## python3 image.py --batch_size 32 --model_name shufflenet-v2-x1-5 --augmented

The parameter --batch_size gives the batch size to be used for training 

The parameter --model_name gives name of the model to be trained which can take name from [efficientnet-b0, shufflenet-v2-x1-5, densenet121, shufflenet+efficient ]

The parameter --augmented is optional. when augmented parameter is given the model runs on augmented data. Else it uses default data
