To run write python main.py
To open outputs double click on csv files on current directory.

Tokenized and labels are default to create feature set and vocabulary.

Main.py finished in ~45-50 minutes because of the forward selection and frequency selection. Naive bayes works fine for the Q2. For that question my implementation gives outputs of the forward selection and frequency selection. However, due to complexity issues run time is approximately ~20-25 minutes for each of them. I figured out using numpy arrays can get rid of the double for loops in naïve_bayes which increase efficiency. Unfortunately, I figured out that issue too late and could not spend time to fix that problem. Although it is working slow, it gives true outputs.

