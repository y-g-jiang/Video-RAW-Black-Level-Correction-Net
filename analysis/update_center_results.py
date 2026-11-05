"""Synchronize manuscript claims with the corrected sliding-center experiment."""
from pathlib import Path
import json
R=Path(__file__).resolve().parents[1]
D=json.loads((R/'evidence/FILTER_COMPARISON_105K.json').read_text('utf8'))
S=D['scores'];filters=['mean','median','exponential','kalman','jump']
alo,ahi=min(S[m]['A_dn'] for m in filters),max(S[m]['A_dn'] for m in filters)
jlo,jhi=min(S[m]['J_dn'] for m in filters),max(S[m]['J_dn'] for m in filters)
from correct_manuscript import replace_paragraph

def edit(name,pairs):
    p=R/'source'/name;s=p.read_text('utf8')
    for start,new in pairs:s=replace_paragraph(s,start,new)
    p.write_text(s,'utf8')

edit('main.tex',[
('Sub-DN black-level errors',r'Sub-DN black-level errors produce shadow chromaticity shifts after exposure amplification and temporal fluctuations in RAW video. We estimate four CFA offsets in native sensor units using multiscale spatial features and direct attention over signed Bayer observations. Queries exchange information across frames and reread each frame, supporting independent single-frame (T1) and joint seven-frame (T7) inference with shared weights. On a single-scene benchmark with 14 photon-transfer-curve profiles and 13 offset patterns, the pretrained estimator reduces nonzero-offset MAE from 0.649947 to 0.054768 DN. Three-camera +6 EV exposure-ladder examples show shadow chromaticity-distance reductions of 16.54\%--36.09\% relative to aligned 0-EV scene references. At the 105k development operating point, T7 reduces adjacent-change error from 0.073666 to 0.054778 DN and absolute error from 0.108795 to 0.102115 DN across eight scenes. On Z63-derived continuous sequences, five seven-frame T1 postfilters use source-held-out parameter selection and response calibration. Paired long-sequence probes measure all responses at the same sliding-window centers used for scoring. Aggregate per-CFA gains differ from T7 by at most 0.0353. T7 attains absolute/adjacent-change errors of 0.1270/0.0480 DN, compared with 0.1328--0.1336/0.0543--0.0587 DN for the filters. These results quantify the accuracy, temporal response, and computational cost of the shared-weight estimator.'),
('On the eight Z63-derived continuous sequences',r'On the eight Z63-derived continuous sequences, T1 and T7 attain $A=0.132780/0.126973\DN$ and $J=0.060524/0.048024\DN$, respectively. The fixed seven-frame mean gives $A/J=0.163986/0.115446\DN$. Table~\ref{tab:filters} compares all five filter families after calibration using paired sliding-center responses. Their aggregate common and relative gains differ from T7 by at most 0.0353 per CFA. Their $A$ spans 0.132834--0.133597 DN and $J$ spans 0.054323--0.058703 DN. The calibrated jump smoother gives $A/J=0.132834/0.054323\DN$; T7 lowers $A$ in seven scenes and $J$ in six. Paired scene-bootstrap 95\% ranges for T7 minus this control are $[-0.008743,-0.003141]\DN$ for $A$ and $[-0.010941,-0.002215]\DN$ for $J$. Against the other four calibrated families, T7 lowers $J$ in all eight scenes.'),
('We estimate four native-unit CFA offsets',r'We estimate four native-unit CFA offsets by combining spatial context with direct retrieval of signed Bayer observations. Shared queries support single-frame and seven-frame inference. Single-scene PTC tests quantify offset accuracy, and three-camera exposure-ladder examples show reduced shadow chromaticity error. On Z63-derived development sequences, T7 achieves lower mean absolute and adjacent-change errors than the five selected, response-calibrated T1 postfilters. Paired sliding-center probes quantify common and empirical-trajectory response alongside these errors. The temporal analysis characterizes the accuracy--response trade-off and the additional computation of joint inference.')])
p=R/'source/main.tex';s=p.read_text('utf8');s=s.replace('That leave-one-scene-out','That leave-one-scene-out')
s=s.replace('The model is fixed at the 105k checkpoint.', 'The model is fixed at the 105k checkpoint.')
needle='The continuous suite contains eight 120-frame sequences'
# The full protocol paragraph remains authoritative; add one short operator statement.
lines=s.splitlines()
for i,line in enumerate(lines):
    if line.startswith('The PTC matrix combines'):
        lines[i]=line.replace('The network is fixed at the 105k checkpoint.','The network is fixed at the 105k checkpoint.')+' Paired 120-frame perturbation probes measure filter and network response at these same sliding-window centers.'
s='\n'.join(lines)+'\n';p.write_text(s,'utf8')

edit('chinese.tex',[
('亚 DN 级黑电平误差',r'亚 DN 级黑电平误差经曝光提亮后会引起暗部偏色，并在 RAW 视频中产生时域波动。本文结合多尺度空间特征与带符号 Bayer 观测的直接注意力检索，以传感器原生数码值预测四个 CFA 通道偏移。查询在帧间交换信息并重新读取各帧观测，在共享权重下支持独立单帧（T1）与联合七帧（T7）推断。在单场景、14 组光子转移曲线与 13 种偏移模式组成的测试矩阵上，预训练模型将非零偏移 MAE 从 0.649947 DN 降至 0.054768 DN。三台相机的 +6 EV 曝光阶梯实例相对对齐的同场景 0 EV 参考图，暗部色度距离降低 16.54\%--36.09\%。在八场景开发集的 105k 工作点，T7 将相邻变化误差从 0.073666 DN 降至 0.054778 DN，绝对误差从 0.108795 DN 降至 0.102115 DN。在 Z63 衍生连续序列上，五类七帧 T1 后处理滤波器采用留一场景选参与响应校准。长序列配对扰动在与评分相同的滑动窗中心测量响应，汇总逐 CFA 增益与 T7 的最大差为 0.0353。T7 的绝对／相邻变化误差为 0.1270/0.0480 DN，滤波器为 0.1328--0.1336/0.0543--0.0587 DN。这些结果量化了共享权重估计器的精度、时域响应与计算成本。'),
('在八段 Z63 衍生连续序列上',r'在八段 Z63 衍生连续序列上，T1 与 T7 的 $A$ 分别为 0.132780/0.126973 DN，$J$ 为 0.060524/0.048024 DN，固定七帧均值的 $A/J$ 为 0.163986/0.115446 DN。表\ref{tab:filters}比较以配对滑窗中心响应完成校准的五类滤波器，其汇总共同与相对增益在各 CFA 上与 T7 的最大差为 0.0353，$A$ 为 0.132834--0.133597 DN，$J$ 为 0.054323--0.058703 DN。校准跳变平滑器的 $A/J$ 为 0.132834/0.054323 DN，T7 在七个场景降低 $A$、六个场景降低 $J$。T7 减去该对照的配对场景 Bootstrap 95\% 区间为：$A$ 的 $[-0.008743,-0.003141]\DN$、$J$ 的 $[-0.010941,-0.002215]\DN$。相对其余四类校准滤波器，T7 在全部八个场景降低 $J$。'),
('本文结合空间上下文',r'本文结合空间上下文与带符号 Bayer 观测检索，以传感器原生单位估计四通道偏移，并通过共享查询支持独立单帧与联合七帧推断。单场景 PTC 测试量化偏移估计精度，三台相机曝光阶梯实例展示暗部色度误差下降。在 Z63 衍生开发序列上，T7 的平均绝对误差与相邻变化误差低于五类选参并校准响应的 T1 后处理滤波器。配对滑窗中心扰动同步量化共同响应与经验轨迹响应，刻画联合推断的精度、响应与计算成本关系。')])
p=R/'source/chinese.tex';s=p.read_text('utf8');lines=s.splitlines()
for i,line in enumerate(lines):
    if line.startswith('PTC 矩阵由 OM-1'):lines[i]=line+' 120 帧配对扰动序列在相同滑动窗中心测量网络与滤波器响应。'
p.write_text('\n'.join(lines)+'\n','utf8')

edit('appendix_en.tex',[
('The two coefficients are solved',r'The two coefficients are solved from the common and relative gain targets on the other seven scenes. Configurations with maximum per-CFA mean-gain difference at most 0.05 are ranked by continuous $J$, then $A$; the nearest-response configuration is used when this set is empty. All five families meet the training-mean tolerance in each of the eight folds. Coefficients and configurations are frozen for the held-out scene. On the pooled held-out predictions, the largest per-CFA gain difference is 0.0353; the largest individual scene--CFA difference is 0.1729. These values measure the finite perturbations specified above. The evidence archive records per-scene gains, coefficients, scores, and candidate operating points.'),
('T7 attains lower $J$',r'Against the calibrated jump smoother, T7 lowers $J$ in six scenes and $A$ in seven. The paired scene-bootstrap 95\% ranges for $J_{T7}-J_{jump}$ and $A_{T7}-A_{jump}$ are $[-0.010941,-0.002215]\DN$ and $[-0.008743,-0.003141]\DN$, respectively. The calculation uses 20,000 paired resamples of the eight development scenes with fixed fold predictions. The other four calibrated families have higher $J$ than T7 in all eight scenes.')])
edit('appendix_zh.tex',[
('两个系数由另外七个场景',r'两个系数由另外七个场景的共同与相对增益目标求解，最大逐 CFA 平均增益差不超过 0.05 的候选按连续 $J$、$A$ 依次排序，候选集为空时选取响应差最小者。五类滤波器在全部八折中均满足训练场景平均增益容限，随后固定系数与参数并应用于留出场景。汇总留出预测的逐 CFA 增益与 T7 的最大差为 0.0353，单个场景与 CFA 组合的最大差为 0.1729。这些增益对应前述有限扰动。随附证据记录逐场景增益、校准系数、得分及候选工作点。'),
('相对每一类校准滤波器',r'相对校准跳变平滑器，T7 在六个场景降低 $J$、七个场景降低 $A$。$J_{T7}-J_{jump}$ 与 $A_{T7}-A_{jump}$ 的配对场景 Bootstrap 95\% 区间分别为 $[-0.010941,-0.002215]\DN$ 和 $[-0.008743,-0.003141]\DN$，由八个开发场景的 20,000 次配对重采样计算，各折预测保持固定。其余四类校准滤波器在全部八个场景的 $J$ 高于 T7。')])

assert abs(alo-.13283373248216634)<1e-12 and abs(ahi-.13359651198006448)<1e-12
assert abs(jlo-.05432264220719743)<1e-12 and abs(jhi-.05870333786956435)<1e-12
print('Updated bilingual results and finite-probe response scope')
