import argparse,json
from pathlib import Path
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,brier_score_loss

def dataset(n=2600,seed=27):
 r=np.random.default_rng(seed);x=r.normal(size=(n,15));logit=.7*x[:,0]-.6*x[:,5]+.8*x[:,10]+.45*x[:,12]-1;y=r.binomial(1,1/(1+np.exp(-logit)));miss=r.random((n,3))<np.array([.18,.28,.12]);xm=x.copy();
 for g in range(3):xm[miss[:,g],g*5:(g+1)*5]=np.nan
 return xm,y,miss
def evaluate(seed=27):
 x,y,miss=dataset(seed=seed);cut=1750;out={}
 for name,indicator in {"mean_imputation":False,"missingness_aware":True}.items():
  imp=SimpleImputer(add_indicator=indicator).fit(x[:cut]);tr=imp.transform(x[:cut]);te=imp.transform(x[cut:]);m=LogisticRegression(max_iter=500).fit(tr,y[:cut]);p=m.predict_proba(te)[:,1];out[name]={"auroc":float(roc_auc_score(y[cut:],p)),"brier_score":float(brier_score_loss(y[cut:],p))}
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=Path("outputs/demo"));p.add_argument("--seed",type=int,default=27);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);r=evaluate(a.seed);(a.output/"metrics.json").write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if __name__=="__main__":main()
