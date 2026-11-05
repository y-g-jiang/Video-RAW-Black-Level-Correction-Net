import os
os.environ.update(TEMP='E:/Temp/bcm18',TMP='E:/Temp/bcm18',MPLCONFIGDIR='E:/Caches/ble_paper_matplotlib',OPENBLAS_NUM_THREADS='1')
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

R=Path(__file__).resolve().parents[1];E=R/'evidence';F=R/'source/figures';T=R/'source/tables'
font_manager.fontManager.addfont(str(R/'source/fonts/msyh.ttc'))
d=json.loads((E/'FILTER_COMPARISON_105K.json').read_text('utf-8'))
raw=json.loads((E/'FILTER_UNCALIBRATED_105K.json').read_text('utf-8'))
runtime=json.loads((E/'NEURAL_RUNTIME_105K.json').read_text('utf-8'))
methods=['T1','mean','median','exponential','kalman','jump','T7']
colors={'T1':'#697386','T7':'#1566a6','mean':'#937041','median':'#895bad','exponential':'#28836b','kalman':'#c54b68','jump':'#d28a25'}
labels_en={'T1':'T1','T7':'T7','mean':'T1 + mean','median':'T1 + median','exponential':'T1 + exponential','kalman':'T1 + Kalman','jump':'T1 + jump model'}
labels_zh={'T1':'T1','T7':'T7','mean':'T1 + 均值','median':'T1 + 中值','exponential':'T1 + 指数','kalman':'T1 + Kalman','jump':'T1 + 跳变模型'}
cost={m:runtime['modes']['T7_cached' if m=='T7' else 'T1_frame']['median_ms_per_center']+d['scores'][m]['filter_us_per_center']/1000 for m in methods}
d['streaming_cost_ms']=cost
d['aggregate_response_match_max_per_cfa']=max(max(np.max(abs(np.array(d['scores'][m][g])-np.array(d['scores']['T7'][g]))) for g in ['g_common','g_relative']) for m in methods if m not in ['T1','T7'])
z=np.load(E/'FILTER_PREDICTIONS_105K.npz');valid=z['valid'];truth=z['truth']
figures={}
for lang in ['en','zh']:
    zh=lang=='zh';labels=labels_zh if zh else labels_en
    rc={'font.family':'Microsoft YaHei' if zh else 'DejaVu Sans','font.size':8,'pdf.fonttype':42,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False}
    with plt.rc_context(rc):
        fig=plt.figure(figsize=(7.1,3.4))
        gs=fig.add_gridspec(2,2,width_ratios=[1.65,1.15],hspace=.28,wspace=.42,left=.09,right=.99,bottom=.15,top=.84)
        a0=fig.add_subplot(gs[0,0]);a1=fig.add_subplot(gs[1,0],sharex=a0);a2=fig.add_subplot(gs[:,1])
        for c,(name,col) in enumerate(zip(['R','G1','G2','B'],['#a74136','#377d4b','#76529a','#24679b'])):
            a0.plot(valid,truth[0,:,c],color=col,lw=1,label=name)
        a0.legend(ncol=4,loc='upper right',frameon=False,fontsize=6.5,handlelength=1,columnspacing=.6)
        a1.plot(valid,truth[0,:,0],color='#171d28',lw=1.1,label='注入真值' if zh else 'Injected target')
        for m,style in [('T1','--'),('jump',':'),('T7','-')]:
            a1.plot(valid,z[m][0,:,0],style,color=colors[m],lw=1.15,label=labels[m])
        for ax in [a0,a1]:ax.set(xlim=(6,113),ylim=(-2,2),yticks=[-2,0,2]);ax.grid(alpha=.2,lw=.5)
        a0.set_ylabel('四 CFA 偏移（DN）' if zh else 'CFA offsets (DN)')
        a1.set_ylabel('R 偏移（DN）' if zh else 'R offset (DN)');a1.set_xlabel('帧' if zh else 'Frame')
        for i,m in enumerate(methods):
            v=d['scores'][m]['J_dn'];a2.barh(i,v,color=colors[m],height=.61)
            a2.text(v+.001,i,f'{v:.4f}',va='center',fontsize=6.5)
        a2.set(yticks=np.arange(7),yticklabels=[labels[m] for m in methods],xlim=(0,.079))
        a2.invert_yaxis();a2.set_xlabel('相邻变化误差 J（DN）' if zh else 'Adjacent-change error J (DN)');a2.grid(axis='x',alpha=.2,lw=.5)
        a2.tick_params(axis='y',labelsize=7,length=0)
        fig.suptitle('Z63 衍生连续序列：滑窗中心响应校准，105k' if zh else 'Z63-derived continuous sequences: sliding-center response calibration, 105k',fontsize=9,x=.09,ha='left')
        fig.legend(*a1.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.50,.94),ncol=4,frameon=False,fontsize=7)
        base=F/('z63_empirical_t1_t7_'+lang);fig.savefig(base.with_suffix('.pdf'),bbox_inches='tight',pad_inches=.04);fig.savefig(base.with_suffix('.png'),dpi=190,bbox_inches='tight',pad_inches=.04);plt.close(fig)
        # Full grid and selected controls, with response and streaming cost visible.
        fig,axes=plt.subplots(1,2,figsize=(7.1,3.1))
        points=raw['candidate_frontier']
        axes[0].scatter([p['A_dn'] for p in points],[p['J_dn'] for p in points],s=9,c='#b5bdc7',alpha=.65,label='原始滤波网格' if zh else 'Uncalibrated grid')
        for i,m in enumerate(methods):
            v=d['scores'][m];marker='*' if m=='T7' else 'o'
            axes[0].scatter(v['A_dn'],v['J_dn'],s=90 if m=='T7' else 33,marker=marker,c=[np.mean(v['g_relative'])],cmap='viridis',vmin=0,vmax=.85,edgecolor='#222',lw=.45,zorder=3)
            axes[1].scatter(cost[m],v['J_dn'],s=90 if m=='T7' else 33,marker=marker,c=colors[m],edgecolor='#222',lw=.45,label=labels[m])
        axes[0].set(xlabel='A (DN)',ylabel='J (DN)',ylim=(0,.125));axes[1].set(xlabel='每中心帧耗时（ms）' if zh else 'Milliseconds per center',ylabel='J (DN)',ylim=(0,.08))
        axes[1].set_xscale('log');axes[1].legend(frameon=False,fontsize=6.5,loc='lower left',ncol=1)
        for ax in axes:ax.grid(alpha=.2,lw=.5)
        sm=plt.cm.ScalarMappable(norm=plt.Normalize(0,.85),cmap='viridis');fig.colorbar(sm,ax=axes[0],fraction=.04,pad=.02,label='平均相对增益' if zh else 'Mean relative gain')
        fig.subplots_adjust(left=.09,right=.99,bottom=.2,top=.94,wspace=.46)
        base=F/('filter_frontier_'+lang);fig.savefig(base.with_suffix('.pdf'),bbox_inches='tight',pad_inches=.04);fig.savefig(base.with_suffix('.png'),dpi=190,bbox_inches='tight',pad_inches=.04);plt.close(fig)
    # Compact primary table: full per-CFA values remain in the evidence JSON.
    rows=[]
    for m in methods:
        v=d['scores'][m]
        rows.append(f"{labels[m]} & {v['A_dn']:.4f} & {v['J_dn']:.4f} & {np.mean(v['g_relative']):.3f} & {np.mean(v['g_common']):.3f} & {cost[m]:.2f} \\\\")
    caption=('七帧响应校准对照的留一场景开发集结果。$A,J$ 单位为 DN，滑窗中心扰动增益取场景与四 CFA 平均，耗时为流式每中心帧毫秒数。' if zh else 'Leave-one-scene-out development results for seven-frame response-calibrated controls. $A,J$ are in DN; sliding-center probe gains average the scenes and four CFAs; time is streaming milliseconds per center.')
    table='\\begin{table}[!t]\n\\centering\n\\caption{'+caption+'}\\label{tab:filters}\n\\footnotesize\n\\setlength{\\tabcolsep}{2.6pt}\n\\begin{tabular}{lrrrrr}\n\\toprule\n'+('方法' if zh else 'Method')+' & $A$ & $J$ & $\\bar g_r$ & $\\bar g_c$ & ms \\\\\n\\midrule\n'+'\n'.join(rows)+'\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'
    (T/('filters_'+lang+'.tex')).write_text(table,'utf-8')
    rows=[]
    for m in methods:
        v=raw['scores'][m]
        gr=[0. if abs(x)<1e-12 else x for x in v['g_relative']]
        rows.append(f"{labels[m]} & {v['A_dn']:.4f} & {v['J_dn']:.4f} & {min(gr):.3f}--{max(gr):.3f} & {min(v['g_common']):.3f}--{max(v['g_common']):.3f} \\\\")
    caption=('逐 CFA 响应校准前的五类七帧滤波对照。参数由另外七个场景选择，均值与中值的窗口固定。' if zh else 'Five seven-frame filter families before response calibration. Parameters are selected on the other seven scenes, while mean and median use fixed windows.')
    table='\\begin{table*}[!t]\n\\centering\n\\caption{'+caption+'}\\label{tab:filters_raw}\n\\footnotesize\n\\begin{tabular}{lrrrr}\n\\toprule\n'+('方法' if zh else 'Method')+' & $A$ (DN) & $J$ (DN) & $g_r$: min--max & $g_c$: min--max \\\\\n\\midrule\n'+'\n'.join(rows)+'\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n'
    (T/('filters_raw_'+lang+'.tex')).write_text(table,'utf-8')
    figures[lang]={n:hashlib.sha256((F/n).read_bytes()).hexdigest() for n in ['z63_empirical_t1_t7_'+lang+'.pdf','filter_frontier_'+lang+'.pdf']}
d['figures']=figures
(E/'FILTER_COMPARISON_105K.json').write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),'utf-8')
print('Aggregate maximum gain difference:',d['aggregate_response_match_max_per_cfa']);print('Streaming cost:',cost)
