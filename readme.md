to solve the identification problem
1. eig method: the eigenvector corresponding to the largest eigenvalue
2. energy detection: using MRC/MVDR to filter the signal in spatial domain, then using the difference/MAP detection to derive the identification information
3. DL method: according to different input and output type
    (1) input:
        1) using the raw data
        2) a.using the eigenvector matrix  b.using the eigenvector matrix + single/multiple DoAs
        3) using the filtered identification matrix
    (2) the corresponding output:
        1) a. (acitivity vector, theta vector) b.(deactivity vector, theta grid)
        2) a. (acitivity vector, theta vector) b.(deactivity vector, theta grid); a. the classification of the id info corresponding to the single input theta (value or one-hot) b.the id mat theta-onehot position indicator vector. 
    (3) 