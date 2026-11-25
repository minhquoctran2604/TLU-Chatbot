import numpy as np
from scipy import stats
M=["bm25","naive","hybrid","mix","graph"]
wins={"factoid":[16,18,28,26,43],"relational":[12,17,37,32,33],"broad":[11,25,24,34,37],
      "aggregate":[13,26,21,34,37],"2-hop":[5,7,12,11,17]}
mr={"factoid":[3.57,3.15,2.79,2.80,2.69],"relational":[3.97,3.18,2.58,2.51,2.76],
    "broad":[3.95,3.11,2.79,2.63,2.52],"aggregate":[3.77,2.95,2.86,2.51,2.91],"2-hop":[3.85,3.31,2.65,2.81,2.39]}
k=5; q05=2.728  # Nemenyi q_alpha, k=5 (Demsar 2006)
def fried(n,r):
    r=np.array(r); chi=12*n/(k*(k+1))*(np.sum(r**2)-k*(k+1)**2/4)
    ff=(n-1)*chi/(n*(k-1)-chi)
    return chi, stats.chi2.sf(chi,k-1), ff, stats.f.sf(ff,k-1,(k-1)*(n-1))
tot_n=0; tot_r=np.zeros(5)
for t in wins:
    n=sum(wins[t]); tot_n+=n; tot_r+=n*np.array(mr[t])
    chi,p,ff,pf=fried(n,mr[t]); cd=q05*np.sqrt(k*(k+1)/(6*n))
    print(f"{t:10s} n={n} chi2={chi:.1f} p={p:.2e} FF={ff:.2f} pF={pf:.2e} CD={cd:.3f}")
omr=tot_r/tot_n
chi,p,ff,pf=fried(tot_n,omr); cd=q05*np.sqrt(k*(k+1)/(6*tot_n))
print("overall n",tot_n,"mean ranks",dict(zip(M,omr.round(3))),f"chi2={chi:.1f} p={p:.2e} CD={cd:.3f}")
for t in list(mr)+["overall"]:
    r=np.array(mr[t]) if t!="overall" else omr
    n=sum(wins[t]) if t!="overall" else tot_n
    cd=q05*np.sqrt(k*(k+1)/(6*n)); best=r.argmin()
    print(t, "best",M[best], "not sig. diff from best:",[M[j] for j in range(5) if r[j]-r[best]<cd])
# overall wins
W=np.array([sum(wins[t][i] for t in wins) for i in range(5)]); N=W.sum(); print("wins",W,N)
print("chi2 GOF uniform wins", stats.chisquare(W))
rng=np.random.default_rng(42); B=10000
bs=rng.multinomial(N,W/N,size=B)
for i in range(5):
    lo,hi=np.percentile(bs[:,i]/N*100,[2.5,97.5]); print(f"{M[i]} share {W[i]/N*100:.1f}% CI [{lo:.1f},{hi:.1f}]")
for a,b in [(4,3),(4,2),(3,2),(2,1),(1,0)]:
    d=bs[:,a]-bs[:,b]; lo,hi=np.percentile(d,[2.5,97.5])
    bt=stats.binomtest(W[a],W[a]+W[b]).pvalue
    print(f"{M[a]}-{M[b]} diff={W[a]-W[b]} CI[{lo:.0f},{hi:.0f}] P(d<=0)={np.mean(d<=0):.4f} binom p={bt:.4f}")
# 2-hop
w=np.array(wins["2-hop"]); bs=rng.multinomial(52,w/52,size=B)
for a,b in [(4,3),(4,2),(4,1)]:
    d=bs[:,a]-bs[:,b]; lo,hi=np.percentile(d,[2.5,97.5]); print(f"2hop {M[a]}-{M[b]} diff={w[a]-w[b]} CI[{lo:.0f},{hi:.0f}] binom p={stats.binomtest(w[a],w[a]+w[b]).pvalue:.3f}")
lo,hi=np.percentile(bs[:,4]/52*100,[2.5,97.5]); print("2hop graph share",17/52*100,lo,hi)
