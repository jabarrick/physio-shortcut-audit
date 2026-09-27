"""Supplementary figure: cue-signed HEOG deflection (exploratory). Run from repo root after
scripts/posthoc_revision3_deflection.py. Writes figures/figS_deflection.png"""
import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,
    'axes.edgecolor':'#555','axes.linewidth':0.6,'grid.color':'#e3e3e3','grid.linewidth':0.5,'legend.frameon':False,'savefig.dpi':300})
o = json.load(open('results/analysis/posthoc_revision3_deflection.json'))
c = np.load('results/analysis/posthoc_revision3_deflection_curves.npz'); t = c['t']
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.7, 2.6), gridspec_kw={'width_ratios': [1.25, 1]})
blues = ['#a9c9ef', '#5b9be3', '#1d5fae']
for k, lab in enumerate(['lowest', 'middle', 'highest']):
    a1.plot(t, c[f'AF7-AF8__none_t{k}'], color=blues[k], lw=1.2, label=f'no saccade, {lab} tercile')
a1.plot(t, c['AF7-AF8__congruent'], color='#eb6834', lw=1.4, label='congruent saccade')
a1.plot(t, c['AF7-AF8__incongruent'], color='#777', lw=1.0, ls='--', label='incongruent saccade')
a1.axvspan(0.3, 1.2, color='#f1f1ef', zorder=0); a1.axhline(0, color='#999', lw=0.6); a1.axvline(0, color='#999', lw=0.6)
a1.set_xlim(-0.5, 3); a1.set_xlabel('time from cue (s)'); a1.set_ylabel('AF7 − AF8, signed towards cue (µV)')
a1.set_title('a  deflection time course', loc='left', fontsize=7.5); a1.legend(fontsize=5.8, loc='upper right')
pairs = o['pairs']; x = np.arange(len(pairs)); ref = 'F7-F8'
for lab, col, mk in (('t2_minus_t0', '#2a78d6', 'o'), ('congruent_minus_none', '#eb6834', 's')):
    v = np.array([o['contrast'][lab][p] for p in pairs]); v = v / o['contrast'][lab][ref]
    a2.plot(x, v, color=col, marker=mk, ms=4, lw=1.2,
            label='no saccade: highest − lowest tercile' if lab.startswith('t2') else 'congruent − no saccade')
a2.axhline(0, color='#999', lw=0.6); a2.set_xticks(x); a2.set_xticklabels([p.replace('-', '−') for p in pairs], rotation=60, fontsize=6)
a2.set_ylabel('0.3–1.2 s contrast (relative to F7 − F8)'); a2.set_title('b  front-to-back profile', loc='left', fontsize=7.5)
a2.legend(fontsize=5.8, loc='upper right'); a2.grid(axis='y')
fig.tight_layout(); fig.savefig('figures/figS_deflection.png'); print('ok')
