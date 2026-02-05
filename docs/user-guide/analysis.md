# Analyze Data

Data are analyzed using different ```classes```. Each one is dedicated to a specific task. 

The class needs to be organized as follows: 

```python

class Analysis:

    def __init__(self, flight):

        """
        Use the flight container to input data

        Args:
            flight (pils.Flight): Container with the data info
        """

    def to_hdf5(self):

        self.data.to_hdf5()
```

The ```to_hdf5``` method needs to create a HDF5 that can store the result of the data analysis. 