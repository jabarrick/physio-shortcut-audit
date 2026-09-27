"""Supplementary figure S1: tercile time courses of the cue-signed HEOG deflection and the selection control
(terciles of the same deflection with random signs).  Run from repo root after
scripts/posthoc_revision4_selection_curves.py.  Writes figures/figS_deflection.png"""
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,
    'axes.edgecolor':'#555','axes.linewidth':0.6,'grid.color':'#e3e3e3','grid.linewidth':0.5,'legend.frameon':False,'savefig.dpi':300})
c = np.load('results/analysis/posthoc_revision4_selection_curves.npz'); t = c['t']
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.7, 2.6), sharey=True)
for ax, (lab, key) in zip((a1, a2), (('a  highest tercile', 'top'), ('b  lowest tercile', 'bot'))):
    ax.axvspan(0.3, 1.2, color='#f1f1ef', zorder=0); ax.axhline(0, color='#999', lw=0.6); ax.axvline(0, color='#999', lw=0.6)
    ax.plot(t, c[f'cue_{key}'], color='#2a78d6', lw=1.3, label='signed towards the cue')
    ax.plot(t, c[f'rand_{key}'], color='#777', lw=1.1, ls='--', label='random sign (selection only)')
    ax.plot(t, c[f'cue_{key}'] - c[f'rand_{key}'], color='#eb6834', lw=1.3, label='difference')
    ax.set_xlim(-0.8, 3); ax.set_xlabel('time from cue (s)'); ax.set_title(lab, loc='left', fontsize=7.5)
a1.set_ylabel('AF7 − AF8 (µV)'); a1.legend(fontsize=6, loc='upper right')
fig.tight_layout(); fig.savefig('figures/figS_deflection.png'); print('ok')
