# Collaborative filtering algorithms are prone to mainstream-taste bias 
Source code for the RecSys 2023 paper [Collaborative filtering algorithms are prone to mainstream-taste bias](https://philipphager.github.io/assets/papers/2023-recsys-mainstream-taste.pdf) by Pantelis P. Analytis and Philipp Hager.

## Installation
1. Install dependencies using conda: `conda env create -f environment.yaml`
2. Activate environment: `conda activate meainstream-taste-bias`
3. Run experiments as described below.

## Run experiments
### 1. Evaluate all models and store user-level evaluation results:
```Bash
python main.py -m \
    data=faces,jester,movielens \
    model=ease,funk,knn-item-item,knn-user-user,nmf
```

### 2. Plot results:
```Bash
python plot-figure-1.py
```

### 3. Evaluate how predictive user features are of user-level model performance:
```Bash
python r2_analysis.py
```

## Reference
```
@inproceedings{Analytis2023MainstreamTasteBias,
  author = {Pantelis P. Analytis and Philipp Hager},
  title = {Collaborative filtering algorithms are prone to mainstream-taste bias},
  booktitle = {Proceedings of the 17th ACM Conference on Recommender Systems (RecSys`23)},
  organization = {ACM},
  year = {2023},
}
```

## License
This project uses the [MIT license](https://github.com/philipphager/recsys-mainstream-taste-bias/blob/main/LICENSE).
