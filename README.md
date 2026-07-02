# phenotype-prediction
Built a binary classification model for disease prediction using genomic datasets. Applied neural network techniques and data preprocessing methods to support medical decision-making scenarios.

Use Guide:
### 1) Open Comand Prompt as Administrator;

### 2) GO to the 'Deep_Model.py' location;

### 3) Run the train function from 'Deep_model.py':


python Deep_Model.py --type_training="LGO" --alpha=1e-3 --epochs=50 --is_training=True --display_step=1 --save=True --processing="train" --bn=True (netural networw with weight penalties)

python Deep_Model.py --alpha=0 --epochs=50 --is_training=True --display_step=1 --save=True --processing="train" --bn=True (normal neural networks)

### 4) Run the evaluate function from 'Deep_model.py':


python Deep_Model.py --type_training="LGO" --alpha=1e-3 --epochs=50 --is_training=False --restore=True --processing="evaluate" --bn=True


### 5) Run the predict function from 'Deep_model.py':


python Deep_Model.py --type_training="LGO" --alpha=1e-3 --epochs=50 --is_training=False --restore=True --processing="predict" --bn=True
