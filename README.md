# W-Transformers : A Wavelet-based Transformer Framework for Univariate Time Series Forecasting

W-Transformer is a new framework to forecast univariate time series. To know more about the achievement of W-Transformer or contact us for more information, you can find the full paper [here](https://arxiv.org/abs/2209.03945). 

## Installation

```bash
git clone https://github.com/CapWidow/W-Transformer.git
cd w-transformer
pip install -r requirement.txt
```

## Usage

```bash
python WTransformer.py
```
### Forcasting horizon
In the code, you can change the variable :
```python
nforecastList = [288,576]
```
this list are the forecasting horizon that you want. (Here for the [NetworkAnalytics.csv](Data/NetworkAnalytics.csv))


Our recommandation are:

For Daily data 
```python
nforecastList = [30,90]
```

For Weekly data
```python
nforecastList = [26,52]
```

For Monthly data
```python
nforecastList = [12,24]
```
### Data
The data you want to use should be in the folder refer in the variable 
```python
 path = 'C:/Users/Admin/ResearchW/MODWT/Data/Data_tmp'
```
Change the path variable in the code by your folder where your data are located. 
```python
 path = 'PATH/TO/YOUR/DATA/FOLDER/'
```
You can put multiple csv in this folder and they will all be execute. 
Please note that the column name of the value should be named 'Cases' as the execution is automated with this name.



## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Cite

If this research help you in your work please cite [W-Transformer](https://arxiv.org/abs/2209.03945) :

```
@article{sasal2022w,
  title={W-Transformers: A Wavelet-based Transformer Framework for Univariate Time Series Forecasting},
  author={Sasal, Lena and Chakraborty, Tanujit and Hadid, Abdenour},
  journal={arXiv preprint arXiv:2209.03945},
  year={2022}
}
```
