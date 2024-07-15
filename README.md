# Labelled-Data-Array

Is a package for handling N-dimensional numerical data arrays with labels. This is simply a wrapper numpy wrapper with labels management and some data selection and casting support.

It is important to note that all the labels for a single axis need to be unique!

### Example
Given **N** ground motion recording stations, **M** different intensity measures, **K** different realisations, a **LabelledDataArray** allows for easily handling of this data while also keeping track of associated lables.

```python
import numpy as np
import labelled_data_array as lda

stations = ["ACB", "DSF", "GHI", "JKL"]
ims = ["PGA", "PGV"]
rels = ["REL01", "REL02", "REL03", "REL04"]

# Generate some random data
data = np.random.rand(len(stations), len(ims), len(rels))

# Creation of a LabelledDataArray
im_data = lda.LabelledDataArray(data, axis_labels=[stations, ims, rels])

# Numpy indexing is supported
print(im_data[0, :, :])

### Data selection by axis value(s) ###

# E.g. select all IM values for station "ACB"
# 2D output is returned as a DataFrame
print(im_data.sel["ACB", :, :])

# Select all values for station "ACB" and IM "PGA"
# 1D output is returned as a Series
print(im_data.sel["ACB", "PGA", :])

# Get the scalar value for station "ACB", IM "PGA" and REL "REL01"
print(im_data.sel["ACB", "PGA", "REL01"])
```


