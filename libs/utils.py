

def accuracy(y, t):
    pred = y.data.max(1, keepdim=True)[1]
    acc = pred.eq(t.data.view_as(pred)).cpu().sum()
    return float(acc)
def class_num(real):
    assert len(set(real.tolist())) > 1
    if len(set(real.tolist())) == 2:
        return True
    else:
        return False
