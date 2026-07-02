Ghid pentru folosirea aplicației:
### 1) Se deschide Command Prompt ca administrator;

### 2) Se introduce locația aplicației ( D:\Licenta\Python );

### 3) Se rulează funcția train din cadrul programului Deep_model în felul următor:


python Deep_Model.py --type_training="LGO" --alpha=1e-3 --epochs=50 --is_training=True --display_step=1 --save=True --processing="train" --bn=True


### 4) Se rulează funcția evaluate din cadrul programului Deep_model în felul următor:


python Deep_Model.py --type_training="LGO" --alpha=1e-3 --epochs=50 --is_training=False --restore=True --processing="evaluate" --bn=True


### 5) Se rulează funcția prdeict din cadrul programului Deep_model în felul următor:


python Deep_Model.py --type_training="LGO" --alpha=1e-3 --epochs=50 --is_training=False --restore=True --processing="predict" --bn=True


### Informație adițională: comanda pentru a rula o rețea neuronală clasică (fără auto-explicabilitate) este următoarea:


python Deep_Model.py --alpha=0 --epochs=50 --is_training=True --display_step=1 --save=True --processing="train" --bn=True
