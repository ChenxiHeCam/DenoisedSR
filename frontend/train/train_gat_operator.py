"""
GAT-based operator predictor.

Idea: operators act on RELATIONSHIPS between variables, which a graph
captures naturally. Per-node behavioral features (periodicity, log-linearity,
power-law slope) + graph attention pooling -> multi-label operator prediction.

Key advantage over global-feature RF: the GAT attends to the TRUE variable
nodes (high support score), ignoring noise nodes, so operator signatures
(e.g. sin's periodicity on theta) aren't washed out by distractor columns.
"""
import sys, re, time, os
sys.path.insert(0, 'src')
sys.path.insert(0, 'D:/Physics Fundation model/src')
sys.path.insert(0, 'D:/Physics Fundation model/scripts')

import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops
from pathlib import Path
from joblib import load as jload

import train_support_predictor_v2 as v2

DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OP_FULL  = ['+','-','*','/','^','sqrt','sin','cos','tan','exp','log',
            'sinh','cosh','tanh','abs','square','cube']
N_OPS    = len(OP_FULL)
NODE_FEAT= 74 + 8       # support features (74) + behavioral (8)
HID      = 128
HEADS    = 4
Q        = 80
EPOCHS   = 40
BATCH    = 32
LR       = 3e-4
SEED     = 42
COOC_T   = 5
OUT      = Path('models/gat_operator.pt')

COOC = jload('models/cooc_graph.joblib')

def infer_ops_full(s):
    s = s.replace('**','^'); found=set()
    for op in ['+','-','*','/','^']:
        if op in s: found.add(op)
    pat={'sqrt':r'\bsqrt\b','sin':r'\bsin\b','cos':r'\bcos\b','tan':r'\btan\b',
         'exp':r'\bexp\b','log':r'\b(log|ln)\b','sinh':r'\bsinh\b','cosh':r'\bcosh\b',
         'tanh':r'\btanh\b','abs':r'\b(abs|Abs)\b'}
    for op,p in pat.items():
        if re.search(p,s): found.add(op)
    if re.search(r'\^\s*2\b',s): found.add('square')
    if re.search(r'\^\s*3\b',s): found.add('cube')
    return found

def behav_node_feat(x, y):
    """8 behavioral features for one column vs target."""
    eps=1e-9; x=np.asarray(x,float); y=np.asarray(y,float); n=len(y)
    if np.std(x)<eps: return np.zeros(8,np.float32)
    order=np.argsort(x); xs,ys=x[order],y[order]
    f=[]
    lc=np.corrcoef(x,y)[0,1]; f.append(0. if not np.isfinite(lc) else lc)
    llc=np.corrcoef(np.log(np.abs(x)+eps),y)[0,1]; f.append(0. if not np.isfinite(llc) else llc)
    elc=np.corrcoef(x,np.log(np.abs(y)+eps))[0,1]; f.append(0. if not np.isfinite(elc) else elc)
    mask=(x>0)&(y>0)
    f.append(np.clip(np.polyfit(np.log(x[mask]),np.log(y[mask]),1)[0],-5,5) if mask.sum()>10 else 0.)
    if n>=16:
        yd=ys-np.polyval(np.polyfit(np.arange(n),ys,1),np.arange(n))
        spec=np.abs(np.fft.rfft(yd-yd.mean()))
        f.append(float(spec[1:].max()/(spec[1:].sum()+eps)) if len(spec)>2 and spec[1:].sum()>eps else 0.)
    else: f.append(0.)
    rc=np.corrcoef(np.argsort(np.argsort(x)),np.argsort(np.argsort(y)))[0,1]; f.append(0. if not np.isfinite(rc) else rc)
    f.append(float(np.clip(np.std(np.diff(ys,2))/(np.std(ys)+eps),0,10)) if n>3 else 0.)
    dy=np.diff(ys); f.append(float(np.sum(np.diff(np.sign(dy))!=0)/max(n,1)))
    return np.array(f,np.float32)

def add_noise(vals,target,n,rng):
    out=dict(vals); q=len(vals[target])
    for i in range(n):
        lo,hi=float(rng.uniform(-5,0)),float(rng.uniform(0,5))
        out[f'__d{i}']=v2.resample(np.array([lo,hi]),q,rng)
    return out

def build_graph_data(aug, target, truth):
    cols=[c for c in aug if c!=target]; y=aug[target]
    feats=[]
    for c in cols:
        sf=v2.column_features(aug[c],y,var_name=c)        # 74
        bf=behav_node_feat(aug[c],y)                        # 8
        feats.append(np.concatenate([sf,bf]))
    feat_mat=np.nan_to_num(np.array(feats,np.float32),nan=0.,posinf=10.,neginf=-10.)
    edges=set()
    for i,a in enumerate(cols):
        for j,b in enumerate(cols):
            if i!=j and COOC.get(a,{}).get(b,0)>=COOC_T: edges.add((i,j))
    if not edges:
        for i in range(len(cols)):
            for j in range(i+1,len(cols)): edges.add((i,j)); edges.add((j,i))
    src,dst=zip(*edges); ei=torch.tensor([list(src),list(dst)],dtype=torch.long)
    ei,_=add_self_loops(ei,num_nodes=len(cols))
    label=torch.tensor([int(op in infer_ops_full(truth)) for op in OP_FULL],dtype=torch.float32)
    return Data(x=torch.tensor(feat_mat),edge_index=ei,y=label.unsqueeze(0))


class GATOperator(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj=nn.Linear(NODE_FEAT,HID)
        self.gat1=GATConv(HID,HID,heads=HEADS,concat=True,dropout=0.1)
        self.norm1=nn.LayerNorm(HID*HEADS)
        self.gat2=GATConv(HID*HEADS,HID,heads=1,concat=False,dropout=0.1)
        self.norm2=nn.LayerNorm(HID)
        # graph-level: concat mean+max pool -> op logits
        self.head=nn.Sequential(nn.Linear(HID*2,128),nn.GELU(),nn.Dropout(0.2),nn.Linear(128,N_OPS))
    def forward(self,x,ei,batch):
        h=F.gelu(self.proj(x))
        h=F.gelu(self.norm1(self.gat1(h,ei)))
        h=F.gelu(self.norm2(self.gat2(h,ei)))
        g=torch.cat([global_mean_pool(h,batch),global_max_pool(h,batch)],dim=-1)
        return self.head(g)   # (B, N_OPS)


def main():
    pool=jload('models/formula_pool.joblib')
    pool=[f for f in pool if f['vals'] is not None]
    print(f"Pool: {len(pool)}  device={DEVICE}  node_feat={NODE_FEAT}  n_ops={N_OPS}")

    model=GATOperator().to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
    # per-op pos_weight to handle imbalance
    freqs=np.zeros(N_OPS)
    for f in pool[:2000]:
        for k,op in enumerate(OP_FULL):
            if op in infer_ops_full(f['truth']): freqs[k]+=1
    freqs/=2000
    pos_w=torch.tensor(np.clip((1-freqs)/(freqs+0.02),1,30),dtype=torch.float32,device=DEVICE)
    bce=nn.BCEWithLogitsLoss(pos_weight=pos_w)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    rng=np.random.default_rng(SEED); idxs=list(range(len(pool)))
    best_f1=0
    for ep in range(1,EPOCHS+1):
        rng.shuffle(idxs); model.train(); losses=[]; t0=time.time()
        TP=np.zeros(N_OPS);FP=np.zeros(N_OPS);FN=np.zeros(N_OPS)
        for bs in range(0,len(idxs)-BATCH,BATCH):
            graphs=[]
            for ii in idxs[bs:bs+BATCH]:
                f=pool[ii]; n_noise=int(rng.integers(2,15))
                aug=add_noise(f['vals'],f['target'],n_noise,rng)
                try: g=build_graph_data(aug,f['target'],f['truth'])
                except: continue
                if g.x.shape[0]<2: continue
                graphs.append(g)
            if not graphs: continue
            batch=Batch.from_data_list(graphs).to(DEVICE)
            opt.zero_grad()
            logits=model(batch.x,batch.edge_index,batch.batch)
            labels=batch.y
            loss=bce(logits,labels)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            losses.append(loss.item())
            pred=(torch.sigmoid(logits)>=0.3).float()
            TP+=((pred==1)&(labels==1)).sum(0).cpu().numpy()
            FP+=((pred==1)&(labels==0)).sum(0).cpu().numpy()
            FN+=((pred==0)&(labels==1)).sum(0).cpu().numpy()
        sched.step()
        rec=TP.sum()/(TP.sum()+FN.sum()+1e-9); prec=TP.sum()/(TP.sum()+FP.sum()+1e-9)
        f1=2*prec*rec/(prec+rec+1e-9)
        print(f"Ep {ep:3d}/{EPOCHS}  loss={np.mean(losses):.4f}  prec={prec:.3f}  rec={rec:.3f}  F1={f1:.3f}  t={time.time()-t0:.1f}s",flush=True)
        if f1>best_f1:
            best_f1=f1
            torch.save({'model':model.state_dict(),'op_vocab':OP_FULL,'node_feat':NODE_FEAT,'epoch':ep},OUT)
    print(f"\nSaved -> {OUT}  best_F1={best_f1:.3f}")

if __name__=='__main__':
    main()
