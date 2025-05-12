# Advanced NLP Exercise 1: Fine Tuning

This is the code base for ANLP HUJI course exercise 1, fine tuning pretrained models to perform sentiment analysis on the SST2 dataset.

# Install
``` pip install -r requirements.txt ```

# Fine-Tune and Predict on Test Set
Run:

``` python ex1.py --max_train_samples <number of train samples> --max_eval_samples <number of validation samples> --max_predict_samples <number of prediction samples> --lr <learning rate> --num_train_epochs <number of training epochs> --batch_size <batch size> --do_train/--do_predict/--do_predict_worst --model_path <path to prediction model> --use_mps```

If you use --do_predict, a prediction.txt file will be generated, containing prediction results for all test samples.
If you use use_mps , this will be running on the gpu of your mac 
if you use --do_predict_worst ,its will create a prediction_worst.txt for the worst configuration you train on 
