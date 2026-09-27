import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
R='results/'
O='figures/'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,
    'axes.edgecolor':'#555','axes.linewidth':0.6,'xtick.color':'#333','ytick.color':'#333','axes.labelcolor':'#222',
    'grid.color':'#e3e3e3','grid.linewidth':0.5,'legend.frameon':False,'savefig.dpi':300})
COL={'csoanet':'#2a78d6','eegnet':'#eb6834','shallow':'#1baf7a','cbramod':'#eda100','labram':'#e87ba4'}
MK={'csoanet':'o','eegnet':'s','shallow':'^','cbramod':'D','labram':'v'}
NAME={'csoanet':'CSOANet','eegnet':'EEGNet','shallow':'ShallowConvNet','cbramod':'CBraMod','labram':'LaBraM'}
ORDER=['csoanet','eegnet','shallow','cbramod','labram']
W=6.7

# ---------- Figure 1: design schematic
fig,ax=plt.subplots(figsize=(W,2.9)); ax.axis('off'); ax.set_xlim(0,100); ax.set_ylim(0,44)
def box(x,y,w,h,t,fc='#f3f3f1'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.3,rounding_size=1.0',fc=fc,ec='#777',lw=0.6))
    ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=5.8,linespacing=1.25)
def arr(x1,y1,x2,y2): ax.annotate('',(x2,y2),(x1,y1),arrowprops=dict(arrowstyle='->',lw=0.7,color='#555'))
box(0.5,29,21,13,'Real resting EEG\n(PhysioNet T0 windows)\n+ random pseudo-labels y')
box(0.5,2,21,22,'Injection, per subject\ntask: mu component,\ngain 1 ∓ s/2\nalpha: EC−EO component, a·Δref\nsaccade: step + spike, a·149 µV\ncoupling P(z = 1 | y) = p')
arr(11,29,11,24.6)
box(26,14,19,16,'Train decoders\n(4 core + LaBraM)\np ∈ {none, .5, .75, .9, 1}\nfactorial s × a at p = .9')
arr(22,13,26,19)
box(49.5,29,21,13,'Paired test sets\nmatched / neutral / reversed\n(same windows, same y)')
box(49.5,2,21,22,'Audit metrics\nencoding: probe D\nreliance: LEACE ΔBA_keep,\nSRI, HEOG regression,\nIG share / norm-abs')
arr(45.5,26,49.5,33); arr(45.5,18,49.5,13)
box(75,29,24.5,13,'Ground truth\nΔBA_neu =\nBA(matched) − BA(neutral)',fc='#e8f0fb')
box(75,2,24.5,22,'Validity test (H1–H3r):\nmetric vs ground truth\nacross cells, seeds, subjects\n\nvalidated metrics →\nreal motor imagery (H4, H5)',fc='#e8f0fb')
arr(71,35.5,75,35.5); arr(71,13,75,13); arr(87,29,87,24.6)
fig.savefig(O+'fig1.png',bbox_inches='tight'); plt.close()

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
ax.text(0.3,len(rows)+0.7,'decision bound 0.3',color='#c0392b',fontsize=6.5,ha='center')
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
ax.set_xlabel('bias-corrected contrast\n(congruent − none), main − control'); ax.set_title(f"H4, {h4['n_eligible']} eligible subjects",fontsize=7.5)
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
