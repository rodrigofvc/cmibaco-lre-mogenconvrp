# Lightly robust solutions for MOGenConVRP under uncertainty

The official implementation of the <em>Cooperative Multi-Indicator Based Ant Colony Optimization Algorithm</em> ($\text{cMIBACO}$) for the lightly robust optimization on MOGenConVRP under uncertainty. This project contains the implementation of individual-indicator-based ant colony algorithms in state-of-the-art such as $\text{IBACO}_ {HV}$, $\text{IBACO} _{R2}$, $\text{IBACO} _{\epsilon^+}$ and weighted sum $\text{IBACO} _{ws}$. Also, we implemented the cMIBACO variants.


## Dependencies

* [autorank](https://github.com/sherbold/autorank?tab=readme-ov-file) v1.1.3
* [matplotlib](https://matplotlib.org/) v3.7.2
* [numpy](https://numpy.org/) v1.25.2
* [pandas](https://pandas.pydata.org/docs/index.html) v2.2.3
* [pymoo](https://pymoo.org/) v0.6.0.1
* [scipy](https://scipy.org/) v1.14.1
* [seaborn](https://seaborn.pydata.org/) v0.13.2
* [Shapely](https://shapely.readthedocs.io/en/stable/) v2.0.6

## Run

You can obtain the set of approximated solutions $P_{Q,\epsilon}$ and the set of lightly robust solutions with approximated solutions as nominal scenario. Important: Only use instances ending in ```_0.5.txt``` for nominal scenario, for example ```Christofides_1_5_0.5.txt```.

### Approximated solutions

For approximated solutions use:

```bash
python3 batch.py <algorithm> <params> <instance>
```

where ```algorithm``` might be:

* ```cmibaco-lns``` for $\text{cMIBACO}$ with local search.
* ```ibaco-hv-lns``` $\text{IBACO} _{HV}$ with local search.
* ```ibaco-r2-lns``` $\text{IBACO} _{R2}$ with local search.
* ```ibaco-epsilon-lns``` $\text{IBACO} _{\epsilon^+}$ with local search.
* ```ibaco-ws-lns``` $\text{IBACO} _{ws}$ with local search.

the ```params``` must be a string of the path with the params in JSON format (see examples in ```params``` dir),

the ```instance``` must be the name of the instance to execute the algorithm (see examples in ```dataset``` dir).

For example:

```bash
python3 batch.py cmibaco-lns params/cmibaco-lns/params-cmibaco.json Christofides_1_5_0.9.txt
```

The result will be stored in ```results\<algorithm>\<instance>\<dir-date>\``` dir, where ```<dir-date>``` is the date of finished in format yyyy-mm-dd-hh-mm-ss. 


### Lightly robust solutions

Once the approximated solutions obtained in the previous step, use the obtained dir ```dir-date``` and the used ```algorithm``` for the ```instance```.

```bash
python3 lre-cmibaco.py <algorithm> <dir-date> <instance>
```

For example:

```bash
python3 lre-cmibaco.py cmibaco-lns 2024-07-29-21-42-00 Christofides_1_5_0.5.txt
```

The result will be stored in ```uncertainty\<instance>\<algorithm>\<dir-date>\```.
  
