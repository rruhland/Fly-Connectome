"""Summarize three distinct matched serve cohorts and the contextual learning curve."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

from .tt4_transfer_trial import ROOT,_read,_write

DIR=ROOT/'artifacts/tt22'


def main():
    duration=_read(DIR/'duration-transfer.json.gz')
    winner=int(_read(DIR/'duration-pong.json.gz')['selected'])
    population=_read(DIR/'population-transfer.json.gz')
    contextual=_read(DIR/'contextual-motor.json.gz')
    cohorts=[]
    first=[(o['results']['duration'],o['results']['neutral']) for entries in duration['streams'].values()
           for o in entries]
    cohorts.append(first)
    for rows in (population['held'],contextual['rows']):
        cohorts.append([(r[winner],r[-1]) for r in rows])
    ids=[p['seed'] for rows in cohorts for p,q in rows if p['seed']==q['seed']]
    if len(set(ids))!=sum(map(len,cohorts)):
        raise ValueError('overlapping or mismatched cohorts')
    result={}
    for kind in ('contact','legal'):
        evidence=np.array([[(p[kind]>0),(q[kind]>0)] for rows in cohorts for p,q in rows],int)
        diff=evidence[:,0]-evidence[:,1]
        win,loss=int(sum(diff==1)),int(sum(diff==-1))
        result[kind]=dict(count=evidence.sum(axis=0).tolist(),source_only=win,neutral_only=loss,
                          matched=len(ids), paired_binomial_p=binomtest(win,win+loss).pvalue)
    _write(DIR/'source-summary.json.gz',result)
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.4),layout='constrained')
    blue,grey,orange='#136586','#707985','#b35a1b'
    ax=axes[0]
    x=np.arange(2)
    for j,(label,c) in enumerate([('Pong duration',blue),('Neutral visual gains',grey)]):
        vals=[result[k]['count'][j] for k in ('contact','legal')]
        bars=ax.bar(x+(j-.5)*.36,vals,width=.32,color=c,label=label)
        ax.bar_label(bars,padding=3,fontsize=11)
    ax.set(xlim=(-.55,1.55),ylim=(0,188),xticks=x,xticklabels=['Racket contact','Legal return'],
           ylabel='Serves out of 176',title='Same arm stroke, three fresh serve cohorts')
    ax.legend(loc='upper right',frameon=False)
    ax=axes[1]
    for key,label,c in [('native','Pong rank',blue),('cold','No Pong prior',grey),('permuted','Permuted rank',orange)]:
        r=contextual['curves'][key]
        curve=np.mean([[x['legal'] for x in trial['eval']] for trial in r],axis=0)
        ax.plot([0,4,8,16,32,48],curve,'o-',color=c,lw=2,label=label,markersize=4)
    ax.set(xlabel='New arm practice serves',ylabel='Legal openings out of 64',
           ylim=(0,26),title='One-option-per-serve contextual learner')
    ax.legend(frameon=False,loc='lower right')
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',linewidth=.6,alpha=.2); ax.set_axisbelow(True)
    fig.savefig(DIR/'source-transfer.png',dpi=150)
    plt.close(fig)


if __name__=='__main__':
    main()
