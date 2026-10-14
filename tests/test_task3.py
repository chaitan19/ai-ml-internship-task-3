from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from src.utils import ROOT, load_dataset, load_task2_split, load_preprocessor_blueprint
from src.train import train_and_evaluate
from src.evaluate import evaluate_classifier
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

def test_dataset_and_split_integrity():
    data=load_dataset(); train,test,manifest=load_task2_split(data)
    assert data.shape == (1470,37)
    assert len(train)==1176 and len(test)==294
    assert set(train.index).isdisjoint(test.index)
    assert "Attrition" not in train.drop(columns="Attrition").columns

def test_task2_preprocessor_is_cloneable():
    pre=load_preprocessor_blueprint()
    data=load_dataset(); train,_,_=load_task2_split(data)
    fitted=pre.fit(train.drop(columns="Attrition"),train["Attrition"])
    transformed=fitted.transform(train.drop(columns="Attrition").iloc[:8])
    assert transformed.shape[0]==8 and np.isfinite(transformed).all()

def test_metrics_positive_class():
    from sklearn.dummy import DummyClassifier
    X=pd.DataFrame({"x":[0,1,2,3]}); y=pd.Series(["No","No","Yes","Yes"])
    model=DummyClassifier(strategy="prior").fit(X,y)
    metrics=evaluate_classifier("dummy",model,X,y)
    assert sum(metrics["confusion_matrix"][0]) == 2
    assert len(metrics["confusion_matrix"]) == 2
    assert 0 <= metrics["roc_auc"] <= 1

def test_full_training_persistence_and_unseen_rows():
    result=train_and_evaluate(ROOT,cv_folds=3)
    assert result["model_path"].is_file()
    model=joblib.load(result["model_path"])
    data=load_dataset(); train,test,_=load_task2_split(data)
    unseen=test.drop(columns="Attrition").iloc[:2].copy(); unseen.loc[unseen.index[0],"JobRole"]="Unseen role"
    assert len(model.predict(unseen))==2
    assert result["reload_verified"] and result["unseen_rows_verified"]==5
