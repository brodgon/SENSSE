<h1 align="center">Ablation Studies</h1>

SENSSE integrates multiple architectural and methodological components, each designed to address a specific aspect of the framework. These components include multitask learning, evidential learning strategies, decoder interaction mechanisms and uncertainty-aware regularization terms, all of which contribute to the final model performance.

To better understand the individual contribution of each design choice, a series of ablation studies was conducted. Rather than focusing exclusively on overall performance, these experiments aim to investigate the role of each component and quantify its impact on image synthesis quality, anatomical segmentation accuracy, and uncertainty estimation.

Three complementary ablation studies were performed. First, a **Component Ablation** analyses the contribution of the main architectural elements composing the SENSSE framework. Second, a **Connection Ablation** evaluates different decoder interaction strategies and their influence on information exchange between synthesis and segmentation tasks. Finally, a **Regularization Ablation** investigates the effect of evidential regularization on predictive performance and uncertainty estimation. Together, these experiments provide a assessment of the design decisions underlying SENSSE and offer insights into the mechanisms responsible for its performance.

---

# Component Ablation

SENSSE combines several key components, including multitask learning, evidential learning for synthesis and segmentation, decoder interactions and auxiliary optimization objectives. While the complete framework integrates all these elements simultaneously, it is important to determine the individual contribution of each component. The purpose of the component ablation study is therefore to answer the following questions:

- Does multitask learning provide a benefit compared to single-task models?
- Is simultaneous optimization preferable to independent synthesis and segmentation networks?
- Does the auxiliary cross-entropy loss improve evidential segmentation?
- What is the contribution of the complete SENSSE formulation?

To investigate these questions, we compare the full SENSSE framework against a series of simplified variants in which individual components are removed.

## Evaluated Models

### Syn-EDL
This model isolates the image synthesis task by removing the segmentation branch and training the network exclusively for synthetic image generation supported by evidential deep learning. The objective of this experiment is to establish a synthesis-only baseline and evaluate the extent to which image reconstruction benefits from the multitask framework proposed in SENSSE. 

### Seg-EDL
This configuration focuses exclusively on anatomical segmentation by removing the synthesis branch. The resulting model provides a segmentation-only baseline that enables the contribution of multitask learning to be quantified. Evidential Deep Learning remains the underlying segmentation framework, allowing evaluation of segmentation performance and uncertainty estimation without any influence from image synthesis.

### Multitask
This model simultaneously performs image synthesis and segmentation using a shared encoder and dual-decoder architecture, but without evidential learning or decoder interaction mechanisms. By jointly optimizing both tasks while removing the uncertainty-aware components of SENSSE, this experiment allows the isolated contribution of multitask learning to be evaluated.

### SENSSE-noCE

This variant corresponds to the complete SENSSE framework with the exception of the auxiliary cross-entropy loss incorporated into the segmentation objective. The purpose of this experiment is to investigate whether explicit probabilistic supervision contributes to improved optimization, segmentation accuracy and evidential learning stability when combined with the EDL formulation.

### SENSSE
SENSSE represents the complete proposed framework

---

## Quantitative Results

```{=latex}

\begin{table*}
\caption{Ablation studies evaluating architectural components, decoder interactions and synthesis evidential regularization.}
\label{tab:ablation_results}
\small \centering \begin{tabular}{c|ccc|ccc}
\hline \multirow{2}{*}{\textbf{Model}} & \multicolumn{3}{c|}{\textbf{Head \& Neck}} & \multicolumn{3}{c}{\textbf{Pelvis}} \\
\cline{2-7} & \textbf{MAE(HU)$\downarrow$} & \textbf{Dice$_{\mathrm{HU>300}}$$\uparrow$} & \textbf{DSC$\uparrow$} & \textbf{MAE(HU)$\downarrow$} & \textbf{Dice$_{\mathrm{HU>300}}$$\uparrow$} & \textbf{DSC$\uparrow$} \\ 
\hline\hline
\multicolumn{7}{c}{\textbf{A) Component Ablation}}\\ 
\hline
Syn-EDL & \cellcolor[HTML]{DCF5F2}{\textbf{44.8077 $\pm$ 9.7229}}  & 0.8976 $\pm$ 0.0509 & -- & 33.2246 $\pm$ 7.8606 & 0.8816 $\pm$ 0.0771 & -- \\ 
Seg-EDL & -- & -- & 0.5918 $\pm$ 0.3497 & -- & -- & \cellcolor[HTML]{DCF5F2}{\textbf{0.9261 $\pm$ 0.2723}} \\
Multitask & 48.8613 $\pm$ 9.7413 & 0.8712 $\pm$ 0.0599 & 0.5609 $\pm$ 0.3305 & 35.7525 $\pm$ 6.1784 & 0.8795 $\pm$ 0.0651 & 0.9012 $\pm$ 0.1823 \\
SENSSE-noCE & 46.9852 $\pm$ 9.8133 & 0.8653 $\pm$ 0.0654 & 0.6356 $\pm$ 0.2974 & 30.2214 $\pm$ 7.5586 & 0.8772 $\pm$ 0.0705 & 0.8974 $\pm$ 0.3114 \\
SENSSE & 46.8729 $\pm$ 8.0024 & \cellcolor[HTML]{DCF5F2}{\textbf{0.9003 $\pm$ 0.0744}} & \cellcolor[HTML]{DCF5F2}{\textbf{0.6889 $\pm$ 0.1752}} & \cellcolor[HTML]{DCF5F2}{\textbf{26.1256 $\pm$ 6.1239}} & \cellcolor[HTML]{DCF5F2}{\textbf{0.8979 $\pm$ 0.0624}} & 0.9164 $\pm$ 0.2275 \\
\hline\hline
\end{table*}
```
---
75
 
76
## Qualitative Analysis
77
 
78
The following qualitative comparisons are recommended:
79
 
80
### Figure A1
81
 
82
Representative synthesis results.
83
 
84
Display:
85
 
86
```text
87
CBCT
88
Ground-truth CT
89
Syn-EDL
90
Multitask
91
SENSSE
92
Error map
93
```
94
 
95
This figure highlights the effect of multitask learning and evidential optimization on image reconstruction quality.
96
 
97
### Figure A2
98
 
99
Representative segmentation results.
100
 
101
Display:
102
 
103
```text
104
Ground-truth
105
Seg-EDL
106
Multitask
107
SENSSE
108
```
109
 
110
Overlay contours on challenging anatomical regions.
111
 
112
---
113
 
114
# Connection Ablation
115
 
116
## Motivation
117
 
118
A central hypothesis of SENSSE is that image synthesis and segmentation can benefit from exchanging information during feature reconstruction. To evaluate this hypothesis, different decoder interaction mechanisms are investigated.
119
 
120
The objective of this study is to determine whether synthesis-guided segmentation, segmentation-guided synthesis, or bidirectional information exchange is most beneficial for adaptive radiotherapy applications.
121
 
122
---
123
 
124
## Evaluated Interaction Strategies
125
 
126
### None
127
 
128
The synthesis and segmentation decoders operate independently after the shared encoder.
129
 
130
No information exchange is performed.
131
 
132
This configuration serves as the interaction-free baseline.
133
 
134
### Syn→Seg
135
 
136
Features from the synthesis decoder are transferred to the segmentation decoder.
137
 
138
This configuration reflects the primary hypothesis of SENSSE, namely that intensity-related information can improve anatomical delineation.
139
 
140
### Seg→Syn
141
 
142
Features from the segmentation decoder are transferred to the synthesis decoder.
143
 
144
This experiment evaluates whether anatomical information can improve synthetic image reconstruction.
145
 
146
### Bidirectional
147
 
148
Feature exchange occurs in both directions.
149
 
150
This configuration enables maximum interaction between tasks.
151
 
152
---
153
 
154
## Quantitative Results
155
 
156
*Insert Table B here*
157
 
158
---
159
 
160
## Methodological Figure
161
 
162
### Figure B1
163
 
164
Illustration of decoder interaction mechanisms.
165
 
166
Display:
167
 
168
```text
169
None
170
 
171
Syn → Seg
172
 
173
Seg → Syn
174
 
175
Bidirectional
176
```
177
 
178
using schematic arrows between decoder branches.
179
 
180
---
181
 
182
## Qualitative Analysis
183
 
184
### Figure B2
185
 
186
Representative Head & Neck example.
187
 
188
Display:
189
 
190
```text
191
Ground-truth CT
192
None
193
Syn→Seg
194
Seg→Syn
195
Bidirectional
196
```
197
 
198
alongside corresponding segmentation masks.
199
 
200
### Figure B3
201
 
202
Uncertainty comparison across interaction strategies.
203
 
204
Display:
205
 
206
```text
207
Prediction
208
Error map
209
Uncertainty map
210
```
211
 
212
to assess whether feature interactions influence uncertainty quality.
213
 
214
---
215
 
216
# Regularization Ablation
217
 
218
## Motivation
219
 
220
Deep Evidential Regression introduces an evidential regularization term that penalizes unsupported confidence and encourages uncertainty estimates to remain consistent with prediction errors.
221
 
222
The regularization strength is controlled by the parameter
223
 
224
```math
225
\eta.
226
```
227
 
228
Selecting an appropriate value is essential because insufficient regularization may lead to overconfident predictions, whereas excessive regularization can degrade reconstruction quality.
229
 
230
The objective of this study is to characterize the influence of evidential regularization on synthesis and segmentation performance.
231
 
232
---
233
 
234
## Evaluated Values
235
 
236
The following values were investigated:
237
 
238
```math
239
\eta = 0
240
```
241
 
242
```math
243
\eta = 10^{-4}
244
```
245
 
246
```math
247
\eta = 10^{-3}
248
```
249
 
250
```math
251
\eta = 10^{-2}
252
```
253
 
254
```math
255
\eta = 10^{-1}
256
```
257
 
258
---
259
 
260
## Quantitative Results
261
 
262
*Insert Table C here*
263
 
264
---
265
 
266
## Qualitative Analysis
267
 
268
### Figure C1
269
 
270
Synthesis predictions as a function of regularization strength.
271
 
272
Display:
273
 
274
```text
275
Ground-truth CT
276
 
277
η = 0
278
η = 10^-4
279
η = 10^-3
280
η = 10^-2
281
η = 10^-1
282
```
283
 
284
This comparison illustrates the trade-off between reconstruction fidelity and uncertainty calibration.
285
 
286
### Figure C2
287
 
288
Corresponding uncertainty maps.
289
 
290
Display:
291
 
292
```text
293
Epistemic uncertainty
294
Aleatoric uncertainty
295
Total uncertainty
296
```
297
 
298
for each regularization value.
299
 
300
---
301
 
302
## Discussion
303
 
304
The ablation studies collectively demonstrate the contribution of the main architectural and optimization choices incorporated into SENSSE. Component ablations quantify the benefit of multitask evidential learning, connection ablations evaluate the effect of information exchange between tasks, and regularization ablations characterize the role of evidential constraints in controlling predictive uncertainty. Together, these experiments provide a comprehensive assessment of the design decisions underlying the proposed framework.
