import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
# File -> manuscript figure: fig1-fig5 -> figures 1-5, fig_eye_posthoc -> figure 6, fig6 -> figure 7.
R='results/'
O='figures/'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,
    'axes.edgecolor':'#555','axes.linewidth':0.6,'xtick.color':'#333','ytick.color':'#333','axes.labelcolor':'#222',
    'grid.color':'#e3e3e3','grid.linewidth':0.5,'legend.frameon':False,'savefig.dpi':300})
COL={'csoanet':'#2a78d6','eegnet':'#eb6834','shallow':'#1baf7a','cbramod':'#eda100','labram':'#e87ba4'}
MK={'csoanet':'o','eegnet':'s','shallow':'^','cbramod':'D','labram':'v'}
NAME={'csoanet':'MS-CNN','eegnet':'EEGNet','shallow':'ShallowConvNet','cbramod':'CBraMod','labram':'LaBraM'}
ORDER=['csoanet','eegnet','shallow','cbramod','labram']
W=6.7

# ---------- Figure 1: design schematic (three columns x three aligned rows; units = inches)
fig = plt.figure(figsize=(W, 3.05)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off'); ax.set_xlim(0, W); ax.set_ylim(0, 3.05)
INK = '#222222'; SUB = '#4a4a4a'; EDGE = '#9a9a9a'
NEUTRAL = '#f4f4f2'; ACCENT = '#2a78d6'; ACCENT_BG = '#e7f0fb'; ARROW = '#6b6b6b'
FT, FB = 7.0, 6.2          # title / body font sizes


def box(x, y, w, h, title, body='', fc=NEUTRAL, ec=EDGE, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.06',
                                fc=fc, ec=ec, lw=0.7))
    ax.text(x + 0.09, y + h - 0.09, title, ha='left', va='top', fontsize=FT, fontweight='bold', color=tc)
    if body:
        ax.text(x + 0.09, y + h - 0.29, body, ha='left', va='top', fontsize=FB, color=SUB, linespacing=1.38)


def arrow(x1, y1, x2, y2, color=ARROW):
    ax.annotate('', (x2, y2), (x1, y1),
                arrowprops=dict(arrowstyle='-|>,head_length=0.35,head_width=0.18', lw=0.9, color=color,
                                shrinkA=0, shrinkB=0))


def plus(x, y):
    ax.text(x, y, '+', ha='center', va='center', fontsize=9, color=ARROW, fontweight='bold')


# column geometry: three columns x three aligned rows
CW = 1.96; GAP = 0.33
X = [0.05, 0.05 + CW + GAP, 0.05 + 2 * (CW + GAP)]
RA, RB, RC = (1.92, 0.72), (0.98, 0.78), (0.04, 0.78)   # (y, h)
TOP = RA[0] + RA[1]
heads = ['1  Generate data', '2  Train and measure truth', '3  Test the audits']
for x, t in zip(X, heads):
    ax.text(x, TOP + 0.24, t, fontsize=7.8, fontweight='bold', color=INK, va='center')
    ax.plot([x, x + CW], [TOP + 0.1, TOP + 0.1], color=INK, lw=0.8)

# ---- column 1: generator
x = X[0]
box(x, RA[0], CW, RA[1], 'Real resting background',
    'PhysioNet rest (T0) windows, 3 s\nrandom balanced pseudo-label y')
plus(x + CW / 2, RA[0] - 0.08)
box(x, RB[0], CW, RB[1], 'Task signal',
    "subject's own mu component (GED)\ngain 1 − s/2 (y = 1), 1 + s/2 (y = 0)")
plus(x + CW / 2, RB[0] - 0.08)
box(x, RC[0], CW, RC[1], 'Confound, injected when z = 1',
    'arm 1: posterior alpha, + a·Δref\narm 2: horizontal saccade, a·149 µV')
ax.text(x + 0.09, RC[0] + 0.1, 'label coupling:  p = P(z = 1 | y = 1)',
        fontsize=FB, color=ACCENT, fontweight='bold', va='bottom')

# ---- column 2: training and ground truth
x = X[1]
box(x, RA[0], CW, RA[1], 'Train decoders',
    'p ∈ {none, .5, .75, .9, 1}\ns × a factorial at p = .9, mismatch cells\n4 core models + LaBraM; 358 units')
box(x, RB[0], CW, RB[1], 'Paired test sets')
ax.text(x + CW - 0.09, RB[0] + RB[1] - 0.1, 'same windows, same y', fontsize=5.6, color=SUB,
        ha='right', va='top', style='italic')
rows = [('matched', 'coupling as in training'), ('neutral', 'z independent of y'), ('reversed', 'coupling inverted')]
for i, (k, v) in enumerate(rows):
    yy = RB[0] + RB[1] - 0.37 - i * 0.14
    ax.text(x + 0.12, yy, k, fontsize=FB, color=INK, fontweight='bold', va='center')
    ax.text(x + 0.72, yy, v, fontsize=FB, color=SUB, va='center')
box(x, RC[0], CW, RC[1], 'Ground truth', 'ΔBA_neu = BA(matched) − BA(neutral)\naccuracy that depends on the z–y relation',
    fc=ACCENT_BG, ec=ACCENT, tc=ACCENT)
arrow(x + CW / 2, RA[0], x + CW / 2, RB[0] + RB[1])
arrow(x + CW / 2, RB[0], x + CW / 2, RC[0] + RC[1])

# ---- column 3: audits
x = X[2]
box(x, RA[0], CW, RA[1], 'Audit metrics',
    'encoding: probe D\nreliance: erasure ΔBA_keep, SRI,\nHEOG regression ΔBA_reg, IG share')
box(x, RB[0], CW, RB[1], 'Validity test (H1–H3r)',
    'metric vs ground truth, cross-fitted,\nacross cells, seeds and subjects',
    fc=ACCENT_BG, ec=ACCENT, tc=ACCENT)
box(x, RC[0], CW, RC[1], 'Real motor imagery (H4, H5)',
    'PhysioNet-MI: natural counterfactual\nwith control models; SHU-MI')
arrow(x + CW / 2, RA[0], x + CW / 2, RB[0] + RB[1])
arrow(x + CW / 2, RB[0], x + CW / 2, RC[0] + RC[1])

# ---- between columns
arrow(X[0] + CW + 0.04, RB[0] + RB[1] / 2, X[1] - 0.04, RA[0] + RA[1] / 2)
arrow(X[1] + CW + 0.04, RA[0] + RA[1] / 2, X[2] - 0.04, RA[0] + RA[1] / 2)
arrow(X[1] + CW + 0.04, RC[0] + RC[1] / 2, X[2] - 0.04, RB[0] + RB[1] / 2, color=ACCENT)

fig.savefig(O+'fig1.png'); plt.close()

# ---------- Figure 2: H1 dose-response
h1=json.load(open(R+'analysis/H1.json'))
fig,axs=plt.subplots(1,2,figsize=(W,2.6),sharey=True)
for ax,conf in zip(axs,['alpha','saccade']):
    d=pd.read_csv(R+f'analysis/H1_{conf}_rows.csv')
    for m in ORDER:
        g=d[d.model==m]
        if g.empty: continue
        per=g.groupby(['amp','subject']).D.mean().reset_index()
        mu=per.groupby('amp').D.mean(); se=per.groupby('amp').D.sem()
        ax.errorbar(mu.index+ (ORDER.index(m)-2)*0.008, mu.values, yerr=1.96*se.values, color=COL[m], marker=MK[m], ms=4, lw=1.2, capsize=0, label=NAME[m])
    ax.axhline(0,color='#999',lw=0.6); ax.set_xlabel('test-time injection amplitude (× a*)')
    ax.set_xticks([0,.125,.25,.5,1]); ax.set_xticklabels(['0','.125','.25','.5','1'])
    s=h1[conf]['lmm']; ax.set_title(f"{'Posterior alpha' if conf=='alpha' else 'Horizontal saccade'}  (slope {s['slope']:.2f}, 95% CI {s['ci_low']:.2f}–{s['ci_high']:.2f})",fontsize=7.5)
    ax.grid(axis='y')
axs[0].set_ylabel('probe encoding D'); axs[0].legend(fontsize=6.5,loc='upper left')
fig.tight_layout(); fig.savefig(O+'fig2.png'); plt.close()

# ---------- Figure 3: H3 forest
h3=json.load(open(R+'analysis/H3.json'))
lab={'erasure_keep':'LEACE ΔBA_keep','sri_post':'SRI (posterior)','ig_share':'IG share','ig_norm_abs':'IG norm-abs','heog_reg':'HEOG regression ΔBA_reg'}
rows=[(k,v) for k,v in h3.items() if k!='degenerate']
fig,ax=plt.subplots(figsize=(W,3.0))
ys=[]
for i,(k,v) in enumerate(rows):
    conf,m=k.split('/'); c=v['corr_cross_fitted']; y=len(rows)-i
    ok=v['decision']=='pass'; col='#2a78d6' if ok else '#9a9a9a'
    ax.plot([c['ci_low'],c['ci_high']],[y,y],color=col,lw=1.6)
    ax.plot(c['estimate'],y,marker='o' if v['confirmatory'] else 'o',mfc=col if v['confirmatory'] else 'white',mec=col,ms=5)
    ax.text(1.02,y,f"{c['estimate']:.2f} [{c['ci_low']:.2f}, {c['ci_high']:.2f}]  {v['decision']}",va='center',fontsize=6.5,transform=ax.get_yaxis_transform())
    ys.append((y,f"{'α' if conf=='alpha' else 'sacc.'} · {lab[m]}"+('' if v['confirmatory'] else ' (expl.)')))
ax.set_yticks([y for y,_ in ys]); ax.set_yticklabels([t for _,t in ys])
ax.axvline(0.3,color='#c0392b',lw=0.8,ls='--'); ax.axvline(0,color='#999',lw=0.6)
ax.text(0.32,len(rows)+0.7,'decision bound 0.3',color='#c0392b',fontsize=6.5,ha='left')
ax.set_xlim(-0.55,1.0); ax.set_ylim(0.3,len(rows)+1.1); ax.set_xlabel('cross-fitted pooled within-model correlation with ΔBA_neu (95% CI)')
ax.grid(axis='x'); fig.tight_layout(); fig.subplots_adjust(right=0.66); fig.savefig(O+'fig3.png'); plt.close()

# ---------- Figure 4: robustness
h3r=json.load(open(R+'analysis/H3r.json')); d=json.load(open(R+'analysis/descriptive.json'))
fig,axs=plt.subplots(1,2,figsize=(W,2.9),gridspec_kw={'width_ratios':[1.25,1]})
ax=axs[0]; items=list(h3r.items())
for i,(k,v) in enumerate(items):
    conf,m,t=k.split('/'); y=len(items)-i
    ax.plot(v['bias'],y,'o',color='#2a78d6',ms=4); ax.plot(v['abs_bias'],y,'x',color='#eb6834',ms=4)
ax.set_yticks([len(items)-i for i in range(len(items))])
ax.set_yticklabels([f"{'α' if k.split('/')[0]=='alpha' else 'sacc.'} {lab[k.split('/')[1]].split(' ΔBA')[0]} · {'mismatch' if k.endswith('mismatch') else 'p = 1'}" for k,_ in items],fontsize=6)
ax.axvspan(-0.05,0.05,color='#e8f0fb'); ax.axvline(0,color='#999',lw=0.6)
ax.set_xlabel('calibration-transfer error (ΔBA_neu units)'); ax.set_title('a  H3r: ● mean bias   × mean |error|',fontsize=7.5,loc='left')
ax=axs[1]; av=pd.DataFrame(d['angle_vs_bias']).dropna(); av['conf']=av.uid.str.split('__').str[1].str.split('_').str[0]
for conf,c,mk in [('alpha','#2a78d6','o'),('saccade','#eb6834','s')]:
    g=av[av.conf==conf]; ax.scatter(g.angle,g.bias,s=8,c=c,marker=mk,alpha=0.7,lw=0,label='alpha' if conf=='alpha' else 'saccade')
ax.axhline(0,color='#999',lw=0.6); ax.set_xlabel('principal angle to task pattern (°)'); ax.set_ylabel('ΔBA_keep − ΔBA_neu')
ax.set_title('b  erasure bias vs principal angle',fontsize=7.5,loc='left'); ax.legend(fontsize=6.5); ax.grid(axis='y')
fig.tight_layout(); fig.savefig(O+'fig4.png'); plt.close()

# ---------- Figure 5: natural counterfactual
t=json.load(open(R+'analysis/real_trials_summary.json')); h4=json.load(open(R+'analysis/H4.json'))
acc=pd.DataFrame([dict(zip(['model','arm','type'],k.split('|')),**v) for k,v in t['accuracy'].items()])
fig,axs=plt.subplots(1,2,figsize=(W,2.6),gridspec_kw={'width_ratios':[1.6,1]})
ax=axs[0]; core=['csoanet','eegnet','shallow','cbramod']; x=np.arange(len(core)); w=0.19
for j,(arm,typ,hatch,col) in enumerate([('main','congruent','','#2a78d6'),('main','none','','#9ec3ee'),('control','congruent','////','#eb6834'),('control','none','////','#f5b59a')]):
    v=[acc[(acc.model==m)&(acc.arm==arm)&(acc.type==typ)].acc.iloc[0] for m in core]
    ax.bar(x+(j-1.5)*w,v,w*0.9,color=col,hatch=hatch,edgecolor='white',lw=0.3,label=f'{arm}, {"congruent saccade" if typ=="congruent" else "no saccade"}')
ax.set_xticks(x); ax.set_xticklabels([NAME[m] for m in core]); ax.set_ylim(0.6,1.0); ax.set_ylabel('trial accuracy (all 90 test subjects)')
ax.legend(fontsize=6,ncol=2,loc='upper center',bbox_to_anchor=(0.5,1.2)); ax.grid(axis='y')
ax=axs[1]; bc=h4['bias_corrected']
for i,m in enumerate(core):
    b=bc[m]; ax.plot([b['ci_low'],b['ci_high']],[i,i],color=COL[m],lw=1.6); ax.plot(b['estimate'],i,MK[m],color=COL[m],ms=5)
ax.set_yticks(range(4)); ax.set_yticklabels([NAME[m] for m in core]); ax.axvline(0,color='#999',lw=0.6)
ax.set_xlabel('bias-corrected contrast\n(main − control)'); ax.set_title(f"H4, {h4['n_eligible']} eligible subjects",fontsize=7.5)
ax.grid(axis='x'); fig.tight_layout(); fig.savefig(O+'fig5.png'); plt.close()

# ---------- Figure 6: real audit
rs=pd.read_csv(R+'analysis/real_summary.csv')
mets=[('heog_reg','HEOG regression ΔBA_reg'),('ig_saccade_share','IG share, frontal-lateral δ'),('ig_alpha_share','IG share, posterior α'),('sri_post','SRI posterior (exploratory)')]
fig,axs=plt.subplots(1,4,figsize=(W,2.3),sharey=True)
for ax,(k,tl) in zip(axs,mets):
    for i,m in enumerate(core):
        v=rs[rs.model==m][k].values; jit=(np.random.default_rng(i).random(v.size)-0.5)*0.3
        ax.scatter(v,i+jit,s=6,color=COL[m],marker=MK[m],alpha=0.6,lw=0); ax.plot(v.mean(),i,'|',color='#222',ms=10,mew=1.4)
    ax.axvline(0,color='#999',lw=0.6); ax.set_title(tl,fontsize=6.8); ax.grid(axis='x')
axs[0].set_yticks(range(4)); axs[0].set_yticklabels([NAME[m] for m in core])
fig.tight_layout(); fig.savefig(O+'fig6.png'); plt.close()
print('ok')

# ---------- Post hoc eye-movement figure (manuscript figure 6; exploratory)
o=json.load(open(R+'analysis/posthoc_revision2_real.json'))
M=['csoanet','eegnet','shallow','cbramod']
fig,axs=plt.subplots(1,2,figsize=(W,2.6),gridspec_kw={'width_ratios':[1.15,1]})
ax=axs[0]; xs=np.arange(3)
for i,m in enumerate(M):
    off=(i-1.5)*0.05
    for arm,ls,key in (('main','-','main'),('control','--','ctrl')):
        v=[o[m][f'{key}_none_t{k}']['est'] for k in range(3)]
        lo=[o[m][f'{key}_none_t{k}']['ci'][0] for k in range(3)]; hi=[o[m][f'{key}_none_t{k}']['ci'][1] for k in range(3)]
        ax.errorbar(xs+off,v,yerr=[np.subtract(v,lo),np.subtract(hi,v)],color=COL[m],ls=ls,marker=MK[m],ms=3.5,lw=1,elinewidth=0.6,capsize=0,
                    mfc=COL[m] if arm=='main' else 'white',label=NAME[m] if arm=='main' else None)
ax.set_xticks(xs); ax.set_xticklabels(['low','middle','high']); ax.set_xlabel('HEOG deflection towards the cued side (tercile)')
ax.set_ylabel('Accuracy, trials without detected saccade'); ax.set_title('(a) solid: main model; dashed: control model',fontsize=7.5,loc='left')
ax.legend(fontsize=6.5,loc='upper left'); ax.grid(axis='y')
ax=axs[1]; ty=[('congruent','Congruent'),('incongruent','Incongruent'),('none','No saccade')]
for i,m in enumerate(M):
    off=(i-1.5)*0.14
    for j,(k,_) in enumerate(ty):
        e=o[m][f'gap_{k}']; ax.errorbar(j+off,e['est'],yerr=[[e['est']-e['ci'][0]],[e['ci'][1]-e['est']]],color=COL[m],marker=MK[m],ms=3.5,elinewidth=0.7,capsize=0)
ax.axhline(0,color='#777',lw=0.6); ax.set_xticks(range(3)); ax.set_xticklabels([t for _,t in ty])
ax.set_ylabel('Main − control accuracy'); ax.set_title('(b) gain from the HEOG signal by trial type',fontsize=7.5,loc='left'); ax.grid(axis='y')
fig.tight_layout(); fig.savefig(O+'fig_eye_posthoc.png'); plt.close()
