B="%"*66
secs=[("Since 2017: What Changed Inside the Transformer?","l3",3,8),
("Architecture Variations: Normalization","l3",9,14),
("Architecture Variations: Activations and Layers","l3",20,29),
("Position Embeddings: RoPE","l3",30,35),
("Hyperparameters","l3",36,52),
("Stability Tricks","l3",53,57),
("Attention Heads: MQA, GQA, Sliding Windows","l3",58,58),
("Mixture of Experts","l4",2,12),
("MoE Routing","l4",13,23),
("Training MoEs","l4",24,37),
("Upcycling and DeepSeek MoE","l4",38,47)]
O_OPEN=r'''% [O] original bridge frame (draft for the user to rewrite); listed in order.txt
\begin{frame}{Since 2017: What Changed Inside the Transformer?}
\begin{itemize}
  \item Session 7 introduced the original 2017 block: post-norm with LayerNorm, sinusoidal position embeddings, a ReLU feed-forward network. GPT, BERT and T5 (session 8) already changed some of these details, e.g.\ learned or relative position embeddings.
  \item Today's large models (e.g.\ the LLaMA family, DeepSeek) keep the overall design, but change several components \emph{inside} the block.
  \item This session goes through these changes and the evidence behind them: normalization, activations, position embeddings, hyperparameters, stability, attention heads.
  \item Then a change at the level of the whole layer: \textbf{mixture of experts}.
  \item Source: Stanford CS336, Lectures 3 and 4 (Spring 2025).
\end{itemize}
\end{frame}

'''
O_MEM=r'''% [O] original bridge frame (draft for the user to rewrite); listed in order.txt
\begin{frame}{Compute vs.\ Memory}
\begin{itemize}
  \item A GPU can do arithmetic much faster than it can move data between its memory and its compute units.
  \item The number of arithmetic operations per byte moved is an operation's \textbf{arithmetic intensity}. With low arithmetic intensity, an operation is limited by \textbf{memory traffic}, not by the number of floating-point operations (FLOPs).
  \item Matrix multiplications do many operations per byte loaded; normalizations and element-wise operations do few.
  \item So an operation can account for a tiny share of the FLOPs and still a noticeable share of the runtime --- the next slides use exactly this argument.
  \item Hardware details follow in session 12.
\end{itemize}
\end{frame}

'''
O_KV=r'''% [O] original bridge frame (draft for the user to rewrite); listed in order.txt
\begin{frame}{The KV Cache}
\begin{itemize}
  \item Autoregressive generation (session 8, decoding) produces one token at a time; each new token attends to all previous tokens.
  \item The keys and values of the previous tokens do not change, so they are stored instead of recomputed: the \textbf{KV cache}.
  \item Its size grows with sequence length $\times$ number of layers $\times$ number of heads $\times$ head dimension (per sequence in the batch).
  \item At inference time, reading this cache from memory --- not the arithmetic --- is often the bottleneck. The following attention variants reduce it.
\end{itemize}
\end{frame}

'''
O_END=r'''% [O] original bridge frame (draft for the user to rewrite); listed in order.txt
\begin{frame}{Where This Is Going}
\begin{itemize}
  \item The changes in this session act on the Transformer \emph{block}, not on the type of data.
  \item Vision Transformers use them too, e.g.:
  \begin{itemize}
    \item ViT-22B: parallel layers, QK normalization, no biases on the QKV projections and LayerNorms \citebutton{Dehghani et al., 2023}{https://arxiv.org/abs/2302.05442};
    \item V-MoE: a sparse mixture-of-experts Vision Transformer \citebutton{Riquelme et al., 2021}{https://arxiv.org/abs/2106.05974}.
  \end{itemize}
  \item Session 10 takes the Transformer to images, multimodal data and video.
\end{itemize}
\end{frame}

'''
L=r"""\input{../style/preamble}
\input{../latex-math/basic-math}
\input{../latex-math/basic-ml}
\input{../latex-math/ml-nn}

% The course's typewriter font (laett) has no bold shape; map bold typewriter to regular.
\AtBeginDocument{{\ttfamily\selectfont}\DeclareFontShape{T1}{laett}{bx}{n}{<->ssub*laett/m/n}{}\DeclareFontShape{T1}{laett}{m}{bx}{<->ssub*laett/m/n}{}}
\newcommand{\citebutton}[2]{\beamergotobutton{\href{#2}{#1}}}

\newcommand{\titlefigure}{figure/title-modern-variant.png}
\newcommand{\learninggoals}{
  \item What changed in the Transformer block since 2017 and why: pre-norm, RMSNorm, no biases, gated activations
  \item Rotary position embeddings (RoPE)
  \item The consensus hyperparameters of modern language models
  \item Stability tricks and attention variants (MQA/GQA, sliding windows), and why memory matters
  \item Mixture of experts: routing, training, DeepSeek-MoE
}

\title{Advanced Artificial Intelligence and Machine Learning}
\date{}

\begin{document}

\lecturechapter{Transformers Modernized}
\lecture{Advanced AI \& ML}

"""
EXTRA={"l3:3":("pre",O_OPEN),"l3:9":("post",O_MEM+"% FRAMES l3 15 19\n\n"),"l3:58":("post",O_KV+"% FRAMES l3 59 68\n\n")}
for t,src,a,b in secs:
    f="lecture 3 (architecture)" if src=="l3" else "lecture 4 (MoEs)"
    pre,post="",""
    e=EXTRA.get(f"{src}:{a}")
    if e and e[0]=="pre": pre=e[1]
    if e and e[0]=="post": post=e[1]
    L+=f"{B}\n\\section{{{t}}}\n% Source: CS336 Spring 2025, {f}, slides {a}-{b}\n{B}\n\n"+pre+f"% FRAMES {src} {a} {b}\n\n"+post
L+=f"{B}\n\\section{{Where This Is Going}}\n{B}\n\n"+O_END+f"\n{B}\n\\section{{License \\& Attribution}}\n{B}\n\n\\begin{{frame}}{{License \\& Attribution}}\n\\begin{{itemize}}\n\\item \\textbf{{Source.}} All frames except the four listed below are adapted from \\textbf{{Stanford CS336 \\emph{{Language Modeling from Scratch}}}}, Spring 2025, Lectures 3 (\\emph{{Architectures, hyperparameters}}) and 4 (\\emph{{Mixture of experts}}) by Tatsunori Hashimoto, \\url{{https://github.com/stanford-cs336/spring2025-lectures}} --- \\textbf{{Copyright 2025 Stanford University}}, licensed under the \\textbf{{MIT License}} (full notice in \\texttt{{\\_sources/}}). One frame per source slide, in the source's order; text in the source's wording, figures and formulas cropped from the source PDF.\n\\item \\textbf{{Figures}} inside the slides are from the papers credited on each frame, reused with that attribution.\n\\item \\textbf{{Original to this course.}} \\enquote{{Since 2017: What Changed Inside the Transformer?}}, \\enquote{{Compute vs.\\ Memory}}, \\enquote{{The KV Cache}}, \\enquote{{Where This Is Going}}.\n\\end{{itemize}}\n\\end{{frame}}\n\n\\endlecture\n\\end{{document}}\n"
open("session-09-transformers-modernized/slides-transformers-modernized.tex","w").write(L)
