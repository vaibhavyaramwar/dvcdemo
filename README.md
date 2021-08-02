create env

'''
    bash
    conda create -n wineq  python 3.7 -y
'''

activate env

'''
    conda activate wineq
'''

create requirements.txt file and install requirements

'''bash
    
pip install -r requirements.txt

'''

Download Wine Quality data from kaggle and copy in data_given folder:

'''
    https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009
'''

initiate git

'''
git init
'''


initiate dvc

'''
dvc init
'''



add dataset

'''
dvc add data_given/winequality.csv
'''


do git add . && git commit -m "first commit"

