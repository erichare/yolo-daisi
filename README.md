# Serverless Principal Components Analysis with Daisies

## How to Call

First, we simply load the PyDaisi package:

```python
import pydaisi as pyd
```

Next, we connect to the Daisi:

```python
principal_components_analysis = pyd.Daisi("erichare/Principal Components Analysis")
```

Now, let's get the PCA data:

```python
pca_data = principal_components_analysis.fit_pca(df=None, vars=None, n_components=2).value
```

And finally, let's plot it!

```python
principal_components_analysis.plot_pca(pca_data, x_component=1, y_component=2, split_by="Variety").value
```

## Running the Streamlit App

Or, we can automate everything by just [Running the Streamlit App](https://dev3.daisi.io/daisies/3504c0f1-10d7-47b3-a2f0-e79f81178ed9/streamlit)
