def f1_binary(preds, gts):
    # preds/gts: list of bools (True if disaster)
    tp = sum(p and g for p,g in zip(preds,gts))
    fp = sum(p and not g for p,g in zip(preds,gts))
    fn = sum((not p) and g for p,g in zip(preds,gts))
    prec = tp/(tp+fp+1e-9); rec = tp/(tp+fn+1e-9)
    f1 = 2*prec*rec/(prec+rec+1e-9)
    return {"precision":prec, "recall":rec, "f1":f1}

def macro_f1_types(pred_types, gt_types, labels):
    # compute macro-F1 over provided labels (ignore None in ground truth)
    per={}
    for lab in labels:
        y_true=[g==lab for g in gt_types if g is not None]
        y_pred=[p==lab for (p,g) in zip(pred_types, gt_types) if g is not None]
        if not y_true: continue
        per[lab] = f1_binary(y_pred, y_true)["f1"]
    if not per: return 0.0
    return sum(per.values())/len(per)
