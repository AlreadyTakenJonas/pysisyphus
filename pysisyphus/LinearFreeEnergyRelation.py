# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 20:03:39 2021

@author: jonas
"""

import numpy as np
from sklearn.linear_model import LinearRegression 

class Regression(LinearRegression):
    
    def __init__(self, trainingset, validationset, targetset, xlabel, ylabel, namelabel):
        # Convert input data to numpy arrays and format the arrays as sklearn expects them.
        self.trainingset = {
            "x": np.array(trainingset[xlabel]).reshape((-1,1)),
            "y": np.array(trainingset[ylabel]),
            "name": trainingset[namelabel]
            }
        self.validationset = {
            "x": np.array(validationset[xlabel]).reshape((-1,1)),
            "y": np.array(validationset[ylabel]),
            "name": validationset[namelabel]
            }
        
        self.targetset = {
            "x": np.array(targetset[xlabel]).reshape((-1,1)),
            "name": targetset[namelabel]
            }
        
        # Call init function of LinearRegression class
        super().__init__()
        
        # Fit the trainingset
        self._fit()
        
    def _fit(self):
        self.model = super().fit(self.trainingset["x"], self.trainingset["y"] )
        
    def validate(self):
        """
        Compute the error of the linear regression. 
        Compare experimental values of validationset to predictions and compute
        square difference.

        Returns
        -------
        mean_square_error : TYPE
            DESCRIPTION.
        square_errors : TYPE
            DESCRIPTION.

        """
        self.validationset["y_pred"] = self.model.predict(self.validationset["x"])
        self.validationset["square_errors"] = [ 
            float( (y - y_pred)**2 ) for y, y_pred 
            in zip( self.validationset["y"], 
                    self.validationset["y_pred"] ) ]
        self.mean_square_error = np.mean(self.validationset["square_errors"])
        square_errors = { 
            name:err for name, err 
            in zip( self.validationset["name"], self.validationset["square_errors"] ) 
            }
        
        # Create additional information to return from this function
        prediction = { 
            name:{
                "y": float(self.validationset["y"][id_]), 
                "y_pred": float(self.validationset["y_pred"][id_])
                } 
            for id_, name in enumerate(self.validationset["name"]) 
            }
        
        # Return no numpy type, because yaml.dump can't handle that.
        return float(self.mean_square_error), square_errors, prediction
    
    def lfer_correction(self):
        """
        Uses the linear regression from the trainingsset to correct the computations of
        the target set.

        Returns
        -------
        result : TYPE
            DESCRIPTION.

        """
        self.targetset["y_pred"] = self.model.predict(self.targetset["x"])
        result = { 
            name:y_pred for name, y_pred 
            in zip( self.targetset["name"], self.targetset["y_pred"].tolist() ) 
            }
        
        # Return no numpy type, because yaml.dump can't handle that.
        return result
    
    def __call__(self):
        return self.model