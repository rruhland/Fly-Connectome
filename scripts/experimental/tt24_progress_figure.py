"""Plot independently tested wrist calibration and voluntary actuator adaptation."""
import gzip
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DIR=Path('artifacts/tt24')


def read(name):
    with gzip.open(DIR/name,'rt') as file:
        return json.load(file)


def main():
    data=read('voluntary-independent.json.gz')['rows']
    optics=read('metric-transfer.json.gz')['cohorts']
    fig,axes=plt.subplots(1,2,figsize=(10.4,4.2),layout='constrained',width_ratios=[1.25,1])
    ax=axes[0]
    ax.axvspan(7.5,22.5,color='#f2eddc',zorder=0)
    series=[('voluntary','Learning during play','#095f76'),('frozen','Frozen commands','#a34c40'),
            ('familiar_wiring','Familiar wiring','#717579')]
    for k,label,color in series:
        y=np.mean([[g['score'] for g in row['modes'][k]]for row in data],axis=0)
        ax.plot(np.arange(1,31),y,label=label,c=color,linewidth=1.9,linestyle='--' if k=='familiar_wiring' else '-')
    ax.text(15,ax.get_ylim()[1]-.1,'Altered arm, optics and wiring',ha='center',va='top',fontsize=8)
    ax.set(title='Same wrist policy across 8 changing streams',xlabel='Drop in continuous learning stream',ylabel='Mean verified rebounds',xlim=(1,30))
    ax.legend(frameon=False,fontsize=8,loc='lower right')
    ax=axes[1]
    original=np.array(optics['frozen']['summary']['counts'])
    improved=np.array(optics['public_motion']['summary']['counts'])
    jitter=np.random.default_rng(89114).uniform(-.07,.07,len(original))
    for j,a,b in zip(jitter,original,improved,strict=True):
        ax.plot([j,1+j],[a,b],lw=.9,color=('#478a75' if b>a else '#a7746c'),alpha=.65,zorder=1)
    ax.scatter(jitter,original,color='#6a7478',s=17,label='Fresh drops',zorder=2)
    ax.scatter(1+jitter,improved,color='#095f76',s=17,zorder=2)
    ax.scatter([0,1],[original.mean(),improved.mean()],s=75,marker='D',color='black',label='Mean',zorder=3)
    ax.set_xticks((0,1),('Stock wrist map','After 4.13 s calibration'))
    ax.set(title='Same 16 releases on the changed arm',ylabel='Consecutive wrist-only rebounds',xlim=(-.28,1.3))
    ax.legend(frameon=False,fontsize=8,loc='upper right')
    for ax in axes:
        ax.grid(axis='y',alpha=.17)
        ax.spines[['top','right']].set_visible(False)
        ax.title.set_fontsize(11)
    fig.savefig(DIR/'transfer-progress.png',dpi=165)


if __name__=='__main__':
    main()
