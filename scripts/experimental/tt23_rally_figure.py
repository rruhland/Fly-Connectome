"""Summarize complete physical games and the separate matched opening experiment."""

import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DIR=Path(__file__).resolve().parents[2]/'artifacts/tt22'


def read(name):
    with gzip.open(DIR/name,'rt') as f:
        return json.load(f)


def plot():
    game=read('ordinary-game.json.gz')
    cold=read('ordinary-zero-practice.json.gz')
    if cold['verified']!=11 or game['verified']!={'online':11,'frozen':11}:
        raise ValueError('missing verified complete games')
    fig,ax=plt.subplots(figsize=(8.5,4.5),layout='constrained')
    x=np.arange(1,12)
    ax.axhspan(10,20,color='#d6ebe1',alpha=.60)
    for values,dx,color,label in [(cold['summary']['lengths'],-.19,'#77818c','Before optical practice'),
                                  (game['summary']['frozen']['lengths'],0.,'#156e69','After four practice points'),
                                  (game['summary']['online']['lengths'],.19,'#be662a','Practice + updates in game')]:
        ax.vlines(x+dx,0,values,color=color,alpha=.50,linewidth=1.5)
        ax.scatter(x+dx,values,s=32,color=color,label=label,zorder=3)
    ax.set(xlim=(.45,11.55),ylim=(-.8,25),xticks=x,yticks=[0,5,10,15,20,25],xlabel='Point in the game',ylabel='Consecutive legal returns by both robots')
    ax.spines[['top','right']].set_visible(False)
    ax.set_title('The same 11 points before and after SO-101 practice',loc='left',fontweight='bold',pad=11)
    ax.legend(frameon=False,loc='upper left',fontsize=9)
    ax.text(11.45,19.1,'10–20 return benchmark',ha='right',va='top',fontsize=8,color='#335a48')
    ax.annotate('23',xy=(7,23),xytext=(7.3,23),va='center',fontsize=10,color='#156e69')
    fig.savefig(DIR/'ordinary-game.png',dpi=150)
    plt.close(fig)


if __name__=='__main__':
    plot()
