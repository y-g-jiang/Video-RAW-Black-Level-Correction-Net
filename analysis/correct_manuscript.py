"""Apply evidence-preserving corrections to the user-edited bilingual sources."""
from pathlib import Path
import json
R=Path(__file__).resolve().parents[1]

def replace_paragraph(s,start,new):
    lines=s.splitlines()
    indices=[i for i,v in enumerate(lines) if v.startswith(start)]
    assert len(indices)==1,(start,indices)
    lines[indices[0]]=new
    return '\n'.join(lines)+'\n'

def main():
    for lang,name in [('en','main.tex'),('zh','chinese.tex')]:
        p=R/'source'/name;s=p.read_text('utf8')
        if lang=='en':
            s=replace_paragraph(s,'The network has 2.32',r'The network has 2.32 million trainable parameters. The 105,000-update checkpoint was designated as the temporal operating point after development evaluation. Table~\ref{tab:metrics} also reports the early 46k model selected by the baseline error constraints; Supplementary Fig.~S2 compares the 102k, 105k, and 106k operating points.')
            s=s.replace('Paired scene-level 95\\% confidence intervals','Paired scene-bootstrap 95\\% intervals')
            s=s.replace('an ablation on paired sequences','a paired condition analysis')
            s=s.replace('designed synthetic step and ramp stress interventions','six designed channel-wise step interventions')
            s=s.replace('joint T7 inference demonstrates superior tracking fidelity under dynamic transitions','joint T7 inference gives lower errors in the designed dynamic condition')
        else:
            s=replace_paragraph(s,'模型包含 232',r'模型包含 232 万个可训练参数。本文在开发评估后将 105,000 次更新检查点确定为时域工作点。表\ref{tab:metrics}同时报告按基线误差容限准则选出的早期 46k 模型，补充图 S2 比较 102k、105k 与 106k 工作点。')
            s=s.replace('配对场景级 95\\% 置信区间','配对场景 Bootstrap 95\\% 区间')
            s=s.replace('人工阶跃和斜坡测试干预','六次指定通道阶跃干预')
            s=s.replace('人工设计的斜坡变动与阶跃干预测试段','六次指定阶跃干预的序列')
        p.write_text(s,'utf8')
    for lang in ['en','zh']:
        p=R/'source'/('appendix_'+lang+'.tex');s=p.read_text('utf8')
        if lang=='en':
            s=replace_paragraph(s,'The baseline model selection criterion',r'The baseline selection criterion limits $C$, $J$, and zero MAE to within 0.005 DN of the initial baseline and zero P95 to within 0.01 DN, then ranks checkpoints by $(A_{T1}+A_{T7})/2$, selecting 46k. Development evaluation subsequently led to designation of the recoverable 105k checkpoint for the temporal analysis. T7 $A/J$ is 0.098377/0.055296 DN at 102k, 0.102115/0.054778 DN at 105k, and 0.096073/0.052858 DN at 106k. The minimum common/relative gains at 2 DN are 0.745/0.505 at 105k and 0.655/0.496 at 106k. Figure~\ref{fig:operating} displays the measured accuracy--stability trajectory.')
            s=s.replace('Visual demonstration on a synthetic Z63-domain sequence','Visual demonstration using the 46k checkpoint on a synthetic Z63-domain sequence')
            s=replace_paragraph(s,'Table~\\ref{tab:sources} reports',r'Table~\ref{tab:sources} reports all eight scene-level results, and Table~\ref{tab:gains} lists the four-channel short-window perturbation gains. The paired scene-bootstrap analysis first averages the two offset trajectories within each source, then resamples the eight equally weighted scenes 20,000 times with seed 20261101. The reported 2.5th--97.5th percentile ranges summarize variation across these development scenes at the fixed, development-selected checkpoint.')
            s=s.replace('Every state is conditioned on exactly the same seven measurements. Continuous scoring reads the center state. The original paired response probes read all seven states from their fixed segment, matching the support of the original T7 gain measurement.',r'Every state is conditioned on the same seven measurements. Both continuous scoring and response measurement read only the sliding-window center state. Paired 120-frame response sequences reuse the base geometry, photon draws, and read-noise draws, with perturbations applied before integer quantization. The common perturbation adds 0.3 DN to each CFA. The relative perturbation centers each empirical CFA trajectory over frames 6--113 and scales its maximum absolute amplitude over all 120 frames to 0.3 DN. Common gain uses the mean paired change; relative gain regresses the temporally centered paired change on the known perturbation. Both are measured at the same 108 centers used for $A$ and $J$.')
            s=s.replace('To compare at the T7 response level','To calibrate the finite-probe response toward T7')
            s=s.replace('Network training code and model-weight distribution are maintained separately in the project repository.',r'The \texttt{neural/} directory contains the inference implementation, and \texttt{weights/nafble\_105000.pt} contains the verified frozen 105k model. The array interface accepts signed reference-subtracted packed Bayer data in DN. The source-image generator used for neural probe re-inference is identified separately; frozen probe predictions support the filter analysis directly.')
        else:
            s=replace_paragraph(s,'早期基线选模准则',r'早期基线选模准则要求 $C$、$J$ 与零偏移 MAE 相对初始基线上浮不超过 0.005 DN，零偏移 P95 上浮不超过 0.01 DN，再按 $(A_{T1}+A_{T7})/2$ 排序，据此选出 46k。经过后续开发评估，本文将可恢复的 105k 检查点确定为时域分析工作点。102k、105k、106k 的 T7 $A/J$ 分别为 0.098377/0.055296、0.102115/0.054778、0.096073/0.052858 DN。在 2 DN 中位亮度下，105k 的最小共同／相对增益为 0.745/0.505，106k 为 0.655/0.496。图\ref{fig:operating}展示训练过程中测得的精度与稳定性关系。')
            s=s.replace('在 Z63 域合成序列上的视觉对比示例。','46k 检查点在 Z63 域合成序列上的视觉对比示例。')
            s=replace_paragraph(s,'表\\ref{tab:sources}给出',r'表\ref{tab:sources}给出八个场景的结果，表\ref{tab:gains}列出四通道短窗扰动增益。配对场景 Bootstrap 分析先在每个源场景内平均两条偏移轨迹，再以随机种子 20261101 对八个等权场景进行 20,000 次有放回重采样。第 2.5--97.5 百分位区间描述固定开发选定工作点下，这些开发场景之间的结果变动。')
            s=s.replace('每个状态均由相同七次观测条件化。连续评分读取中心状态，原有配对响应探针则从固定片段读取全部七个状态，与原始 T7 增益测量的支持范围一致。',r'每个状态均由相同七次观测条件化。连续评分与响应测量均只读取滑动窗中心状态。120 帧配对响应序列共用基础序列的几何、光子和读出噪声抽样，在整数量化前加入扰动。共同扰动为各 CFA 恒定增加 0.3 DN，相对扰动将各通道经验轨迹按第 6--113 帧的均值中心化，再将整段 120 帧的最大绝对幅值归一至 0.3 DN。共同增益取配对输出变化的均值，相对增益对中心化配对输出变化与已知扰动作回归，两者均使用与 $A$、$J$ 相同的 108 个中心帧。')
            s=s.replace('为在 T7 响应水平下比较','为使有限扰动下的响应接近 T7')
            s=s.replace('网络训练代码和模型权重在项目仓库中分别管理。',r'\texttt{neural/} 提供推断实现，\texttt{weights/nafble\_105000.pt} 提供已核验的冻结 105k 模型。数组接口接收以 DN 表示、减去参考黑电平的带符号 Bayer 打包数据。神经网络响应重推断所需的本地源图生成器单独列明，随包冻结响应预测可直接用于滤波分析复算。')
        p.write_text(s,'utf8')

if __name__=='__main__':main()
