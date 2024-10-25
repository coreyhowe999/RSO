# RSO binder design
This is a barebones modal app that designs binding sequences given a target pdb. The [RSO method](https://www.science.org/doi/10.1126/science.adq1741) for binder sequence design utilizes the [colabdesign repository](https://github.com/sokrypton/ColabDesign/tree/main) and was developed by Chris Frank and team.


## install and set up modal
```
pip install modal
python3 -m modal setup
```
## clone repository
```
git clone https://github.com/coreyhowe999/RSO.git
```
## run binder design 
```
cd RSO
mv path/to/target.pdb RSO/target.pdb
modal run modal_rso.py --pdb target.pdb --numdesigns 10 --trajiters 100 --binderlen 100
```
