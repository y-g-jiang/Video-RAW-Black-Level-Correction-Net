# 近邻论文：定位、章法与引用入口

检索与阅读日期：2026-11-01。共整理21篇，全文文本、版本和获取回执保存在[sources](sources/)。优先复用已有SID、ELD PDF，只保存新论文的文本与小型回执，未重复归档大批PDF。以下“主文核读”指原文文本、公式、图注与结构阅读，不等同于逐页检查出版版式。BLBE、ELD、CRVD为主文深读，FastDVDnet核读§1–5，YOND、NAFNet及其他文献采用注明的定向深度。

## 1 四个最直接的章法参照

### R1 BLBE：先承认最接近先例，再把研究问题推进一步

**Mohammadi et al., Why Low-Light Cameras Go Color Blind: Removing Color Bias in Raw Denoising.** ICCP 2026，本次对象为 [arXiv:2607.11090v1](https://arxiv.org/html/2607.11090v1)。[作者页](https://mhdmohammadi.github.io/SIDD-CC/)标示ICCP及推荐PAMI专刊，不能将推荐写成已发表PAMI。

**原文怎样推进。**引言从低照度颜色偏差和逐相机数据负担切入，§III-A建立噪声形成，III-B区分dark shading中的全局BLE，III-C解释仅向训练噪声增加BLE仍会出现局部色偏，III-D提出显式估计—校正—去噪。§IV讨论SIDD参考生成中的颜色偏差，§V按相机和数据依赖类别比较，并用V-D消融回到机制。

**图表职责。**方法图定义全局偏移模块的位置，颜色图检验系统性偏色，消融表同时测试偏移监督、裁零、模型容量、训练和推断分辨率。训练数据、参数采样、网络和伪代码集中在Appendix A，主文只保留解释方法所需的监督式。

**可借鉴。**从可见颜色现象进入观测层的全局偏移，先给方法机制，再让消融回答同一个问题。把类别不同的基线按数据依赖说明清楚，避免看一列平均数。

**对本稿的直接修正。**显式BLE与去噪解耦、四通道输出、全局上下文和偏移辅助监督均已有先例。我们的差异放在独立校正器的双通路读取、查询重读及共享特征的T1/T7。BLBE没有证明所有分辨率差异只来自某一模块，我们也不这样外推。

**阅读范围。**主文引言、II–V、结论及Appendix A中的推断和训练说明，本次补核III-D、V-C/D、参考文献与预处理伪代码。保留[sources/BLBE.json](sources/BLBE.json)和全文文本。

### R3 ELD：让模拟器按成像过程出现

**Wei, Fu, Yang, Huang, A Physics-Based Noise Formation Model for Extreme Low-Light Raw Denoising.** CVPR 2020，实际阅读为本地arXiv:2003.12751v2作者稿。[公开入口](https://openaccess.thecvf.com/content_CVPR_2020/html/Wei_A_Physics-Based_Noise_Formation_Model_for_Extreme_Low-Light_Raw_Denoising_CVPR_2020_paper.html)。

**原文怎样推进。**§1指出常用合成噪声遗漏电子学因素，§3从光子到电子、读出到量化建立模型，3.2随后解释怎样获得参数。§4的数据集承担跨设备验证，§5先以逐项噪声模型消融连接物理因素与恢复质量，再给真实相机与校准敏感性。

**图表职责。**电子成像链与噪声分布图用于说明模型为何这样定义，噪声逐项添加表回答每一因素带来什么，真实图像与跨相机表提供应用验证。训练参数集中在§5.1。

**可借鉴。**我们的模拟器按运动／积分／CFA／加噪／偏移／量化讲述，每一步与目标相机的观测对应，主文不以脚本模块名组织。一个场景的多PTC矩阵是受控条件图，独立场景的实拍端点承担另一类证据。

**不照搬。**本文的Gaussian+Poisson设置没有包含ELD的全部Tukey长尾、row noise等成分，不能借引用宣称已建立完整真实传感器模型。

**阅读范围。**作者稿主文1–8页，方法、校准、数据、§5.1–5.3及结论，本地原PDF复用，页数不当作正式版页数。

### R17 CRVD／RViDeNet：合成时域证据与真实动态场景各司其职

**Yue et al., Supervised Raw Video Denoising With a Benchmark Dataset on Dynamic Scenes.** CVPR 2020。[原文](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yue_Supervised_Raw_Video_Denoising_With_a_Benchmark_Dataset_on_Dynamic_CVPR_2020_paper.pdf)。

**原文怎样推进。**§3先解决动态序列参考难取得的问题，分别介绍可控拍摄和合成视频，§4围绕预去噪、对齐、非局部关联、时间融合、空间融合与ISP建立方法。§5集中训练设置、消融和量化／真实展示。

**图表职责。**数据样例解释运动参考如何生成，总框架图解释四CFA处理和融合位置，消融表连接模块与任务，室内有参考量化和室外无参考展示分开。原文 temporal loss 约束的是恢复图像的一致性。

**可借鉴。**视频论文需要比静态PSNR多一层时间证据，参考来源必须先定义。我们的主图同时报告校正序列、变化误差和响应，以“偏移随时间怎样变化”串联实验。

**不照搬。**本项目没有生成去噪图像或训练ISP，也没有该文式样的实拍动态GT。已有经验黑峰轨迹是模拟标签来源，不能借CRVD的语境写成动态黑位真值数据集。

**阅读范围。**CVF主文§1–6，重点§3、4.1–4.7和5，图注与表体核读，未宣称复现其结果。

### R16 FastDVDnet：时间一致性本来就是一条结果轴

**Tassano, Delon, Veit, FastDVDnet: Towards Real-Time Deep Video Denoising Without Flow Estimation.** CVPR 2020。[原文](https://openaccess.thecvf.com/content_CVPR_2020/papers/Tassano_FastDVDnet_Towards_Real-Time_Deep_Video_Denoising_Without_Flow_Estimation_CVPR_2020_paper.pdf)。

**原文怎样推进。**引言明确单帧与视频的三个差别：多观测、时间一致性、计算成本。§2给出五帧两级模型，§3以设计对照解释两级结构、多尺度及训练方式，§4集中训练细节，§5同时讨论数值质量、时域伪影和不同结构场景的表现。

**图表职责。**框架图解释时间邻域的使用，运动／遮挡例子定位光流伪影，平坦区域的时域残留和视频质量指标说明为什么单帧指标不足。文中残余相关性与闪烁的讨论尤其贴近本项目。

**可借鉴。**在引言就把稳定性作为明确用途，并用时间指标和可视化支持。平滑是有效应用结果，其代价可通过真实变化响应量化。

**不照搬。**无光流视频处理已有先例，不能作为我们的独占创新。其运行时间、AWGN条件和像素级去噪任务均不同，本文5.946秒冷推理不继承其“real-time”结论。

**阅读范围。**CVF §1–5及对应图注（PDF1–6页），下载文本包含后文但本轮不将未重点核读的runtime细节写成已比对结果。

## 2 扩展阅读矩阵

|编号与文献|本次实际深度／版本|与本文的关系|适合借鉴的组织方式|
|---|---|---|---|
|R2 Chen et al., **Learning to See in the Dark**, CVPR2018，[arXiv1805.01934](https://arxiv.org/abs/1805.01934)|已有v1 PDF，本轮补读摘要／引言，旧稿已有引用|低照度RAW与长曝光参考先例|首图先呈现问题和实际成像条件，不把RAW到sRGB网络当四数估计器|
|R4 Chen et al., **Simple Baselines for Image Restoration**, ECCV2022，[2204.04676v1](https://arxiv.org/html/2204.04676v1)|引言、§2–3指定段落与框架图注|SimpleGate编码器来源|按设计问题组织对照，我们的无SCA变体不叫原封不动NAFNet|
|R5 Feng et al., **Learnability Enhancement for Low-light Raw Denoising: Where Paired Real Data Meets Noise Modeling**, ACM MM2022，[DOI](https://doi.org/10.1145/3503161.3548186)|Crossref题录独立核验，方法位置参考BLBE引用，原文未深读|暗场／相机相关噪声建模近邻|作为待深入的直接比较族，不得声称本轮已核其全部校准设置|
|R6 Feng et al., **YOND: Practical Blind Raw Image Denoising Free from Camera-Specific Data Dependency**，[2506.03645v1](https://arxiv.org/html/2506.03645v1)|摘要、结构、III-A/B、III-C指定推导、V-A困难条件|未知相机噪声估计及VST偏差|模块由观察到的失败驱动，困难条件连接可观察中间量，它估计噪声参数，目标不同|
|R7 Zhang et al., **Rethinking Noise Synthesis and Modeling in Raw Denoising**, ICCV2021，[CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Zhang_Rethinking_Noise_Synthesis_and_Modeling_in_Raw_Denoising_ICCV_2021_paper.html)|出版页摘要，先前Crossref题录核验|从真实传感器噪声采样，pattern-aligned sampling/high-bit reconstruction|说明物理模拟与真实噪声采样的不同数据依赖|
|R8 Abdelhamed et al., **Noise Flow: Noise Modeling With Conditional Normalizing Flows**, ICCV2019，[CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Abdelhamed_Noise_Flow_Noise_Modeling_With_Conditional_Normalizing_Flows_ICCV_2019_paper.html)|出版页摘要|学习噪声分布与显式参数模型的邻近路线|噪声拟合指标与成像恢复指标分清，不能把likelihood改善说成BLE精度|
|R9 Brooks et al., **Unprocessing Images for Learned Raw Denoising**, CVPR2019，[1811.11127v1](https://arxiv.org/html/1811.11127v1)|摘要与§3/4目录结构|RGB代理到RAW训练的前提与处理流程|按光度处理环节解释代理生成，我们的线性化代理不自动等同完整unprocessing|
|R10 Abdelhamed, Lin, Brown, **A High-Quality Denoising Dataset for Smartphone Cameras**, CVPR2018，[CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Abdelhamed_A_High-Quality_Denoising_CVPR_2018_paper.pdf)|摘要、引言与首图说明，PDF文本取得|参考图像生成与质量|把参考获取的物理含义说清，结合BLBE的SIDD-CC讨论审视裁零顺序|
|R11 Foi, **Clipped Noisy Images: Heteroskedastic Modeling and Practical Denoising**, Signal Processing2009，[DOI](https://doi.org/10.1016/j.sigpro.2009.04.035)|本次题录核验，结构建议来自EDITORIAL_SKILL，未再核原文|截码与观测分布关系|参考其观测模型→推断→合成/真实结构，具体公式需原文进一步核读|
|R12 Foi et al., **Practical Poissonian-Gaussian Noise Modeling and Fitting for Single-Image Raw-Data**, TIP2008|仅BLBE参考文献[41]题录追溯，原文未核|PTC、均值—方差与读噪基础|背景候选，不以二手引用替代方法比较|
|R13 Lee et al., **Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks**, ICML2019，[1810.00825v1](https://arxiv.org/html/1810.00825v1)|v1标题为Set Transformer，摘要和结构|查询聚合／集合注意力先例|解释读取结构的学术来源，避免将少量查询汇聚本身宣称首创|
|R14 Jaegle et al., **Perceiver: General Perception with Iterative Attention**, ICML2021，[2103.03206v1](https://arxiv.org/html/2103.03206v1)|摘要、结构，v1 HTML标题抽取异常但正文内容正确|大量输入到小潜变量、迭代读取先例|说明瓶颈与重读设计，创新要落到偏移任务的具体约束|
|R15 Zaheer et al., **Deep Sets**, NeurIPS2017，[1703.06114v1](https://arxiv.org/html/1703.06114v1)|摘要、结构|无序观测聚合基础|相关工作简短背景，带位置的整网不据此直接宣称排列不变|
|R18 Chen et al., **Seeing Motion in the Dark**, ICCV2019，[CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Chen_Seeing_Motion_in_the_Dark_ICCV_2019_paper.html)|出版页摘要|极暗RAW视频、静态训练到动态应用、时间一致性|独立强调时间稳定性，并解释动态GT难取得|
|R19 Lehtinen et al., **Noise2Noise: Learning Image Restoration without Clean Data**, ICML2018，[1803.04189v1](https://arxiv.org/html/1803.04189v1)|摘要、结构|噪声期望与监督目标关系|只用于说明依赖噪声统计的训练假设，不自动解决未知系统偏移|
|R20 Batson & Royer, **Noise2Self: Blind Denoising by Self-Supervision**, ICML2019，[1901.11365v1](https://arxiv.org/html/1901.11365v1)|摘要、结构|噪声维度独立与自监督先例|共享全局偏移需要单独讨论，不可泛称自监督能识别任意黑位|
|R21 Gnanasambandam, Sanghvi, Chan, **The Secrets of Non-Blind Poisson Deconvolution**, TCI10(2024),343–356，[2309.03105v1](https://arxiv.org/html/2309.03105v1)|摘要、III–V章节标题与组织，章法信息与skill对照|写作结构参考，任务为去卷积|III逐个设计问题→IV组合→V整体评价，不移植其性能或“Poisson likelihood不必需”结论|

## 3 相关工作建议成段写法

**低照度成像与噪声建模。**SID推动了以真实RAW配对数据学习极暗场景处理，ELD将传感器噪声因素与合成训练连接起来。随后工作分别探索暗场相关数据、真实噪声采样和学习型噪声模型，YOND则研究未知相机条件下的噪声估计与方差稳定变换。这条文献线说明噪声和处理流程会影响恢复结果。本文针对其中的全局通道偏移建立直接估计任务，评价对象是校正量及其成像作用。

**显式黑电平与颜色偏差。**BLBE是最近的直接先例：用辅助偏移监督训练全局四通道估计器，先校正再去噪，并分析输入裁零和上下文大小。本文延续显式偏移表示，研究空间网格、原生像素证据及查询重读的组合，再将其扩展到共享空间特征的单帧和七帧推断。

**时间一致性。**Seeing Motion in the Dark、FastDVDnet及RViDeNet分别从极暗视频学习、无显式光流的时空网络和真实RAW视频数据出发研究时间一致性。本文在四维偏移序列上使用时间上下文，以校正误差的变化和真实扰动响应评价其作用。像素级恢复与低维校正承担不同的成像职责。

**全局证据读取。**Deep Sets、Set Transformer和Perceiver为集合汇聚、小查询集和迭代注意力提供背景。本文通过原生Bayer单元描述子与空间网格定义两路证据，令时间上下文更新后的查询回读自身帧。方法贡献应由这些具体选择及其匹配对照支撑。

## 4 投稿前引用核对

主文推荐先用R1–4、R6、R9、R13–14、R16–18形成紧凑关系，其他放到针对性讨论。R5、R11–12在需要具体技术陈述时应补核原文。R21作为内部章法参考，可不进入最终学术引文表。阅读笔记不等于正式参考文献排版，终稿统一作者名、出版页码和版本后再导出BibTeX。

本轮发现并纠正两类容易影响成文的问题：不能把BLBE已有的显式估计称为我们的首次提出，不能把“无需逐输入真实噪声参数”扩大为Z6 III适配完全不依赖目标条件。两项均已落实到主稿和主张映射。
