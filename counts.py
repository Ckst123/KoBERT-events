import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

label_list = ['확진자수','완치자수','사망여부','집단감염','백신관련','방역지침','경제지원','마스크','국제기구','병원관련']

if __name__ == '__main__':
    data = pd.read_csv('train.csv')
    data = data[label_list].dropna()
    print('total counts:', len(data))
    for lab in label_list:
        print(lab)
        lab, counts = np.unique(data[lab], return_counts=True)
        for l, c in zip(lab, counts):
            print(l, ':', c, end=' ')
        print()
    

