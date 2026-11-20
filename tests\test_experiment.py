from missing_survival.experiment import dataset,evaluate
def test_shape():
 x,y,m=dataset(20,2);assert x.shape==(20,15) and m.shape==(20,3) and len(y)==20
def test_metrics():assert all(0<=v<=1 for m in evaluate(3).values() for v in m.values())

