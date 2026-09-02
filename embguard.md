---
layout: project
title: EMBGuard
permalink: /embguard/
blurb: EMBGuard is a hazard-aware safety guardrail for safe planning in embodied agents, accepted at ICML 2026.
og_image: /assets/img/embguard/motivation_figure_tradeoff.png
---

<section class="hero">
  <div class="hero-body">
    <div class="container is-max-desktop">
      <div class="columns is-centered">
        <div class="column has-text-centered">
          <h1 class="title is-1 publication-title">EMBGuard: Constructing Hazard-Aware Guardrails for Safe Planning in Embodied Agents</h1>

          <div class="publication-venue-row">
            <img src="{{ '/assets/img/embguard/icml.png' | relative_url }}" alt="ICML 2026, Seoul">
            <span class="is-size-3 has-text-weight-semibold">ICML 2026</span>
          </div>

          <div class="is-size-5 publication-authors">
            <span class="author-block"><a href="{{ '/' | absolute_url }}">Dongwook Choi</a><sup>1</sup><sup>*</sup>,</span>
            <span class="author-block"><a href="https://connoriginal.github.io/" target="_blank" rel="noopener noreferrer">Taeyoon Kwon</a><sup>1</sup><sup>*</sup>,</span>
            <span class="author-block">Bogyung Jeong<sup>1</sup>,</span>
            <span class="author-block">Minju Kim<sup>1</sup>,</span>
            <span class="author-block">Yeonjun Hwang<sup>1</sup>,</span>
            <span class="author-block">Hyojun Kim<sup>1</sup>,</span>
            <span class="author-block">Byungchul Kim<sup>2</sup>,</span>
            <span class="author-block">Young Kyun Jang<sup>3</sup>,</span>
            <span class="author-block"><a href="https://jinyoungyeo.github.io/" target="_blank" rel="noopener noreferrer">Jinyoung Yeo</a><sup>1</sup></span>
          </div>

          <div class="is-size-5 publication-authors">
            <span class="author-block"><sup>1</sup>Yonsei University,</span>
            <span class="author-block"><sup>2</sup>Sungkyunkwan University,</span>
            <span class="author-block"><sup>3</sup>Independent</span>
          </div>

          <div class="is-size-6 publication-authors">
            <span class="author-block"><sup>*</sup>Equal contribution</span>
          </div>

          <div class="column has-text-centered">
            <div class="publication-links">
              <span class="link-block">
                <a href="https://arxiv.org/abs/2605.30924" class="external-link button is-normal is-rounded is-dark" target="_blank" rel="noopener noreferrer">
                  <span class="icon"><i class="ai ai-arxiv"></i></span>
                  <span>arXiv</span>
                </a>
              </span>
              <span class="link-block">
                <a href="https://github.com/dongwxxkchoi/EMBGuard" class="external-link button is-normal is-rounded is-dark" target="_blank" rel="noopener noreferrer">
                  <span class="icon"><i class="fab fa-github"></i></span>
                  <span>Code</span>
                </a>
              </span>
              <span class="link-block">
                <a href="https://huggingface.co/EMBGuard" class="external-link button is-normal is-rounded is-dark" target="_blank" rel="noopener noreferrer">
                  <span class="icon"><img class="hf-logo" src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt=""></span>
                  <span>Data &amp; Models</span>
                </a>
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</section>

<section class="section">
  <div class="container is-max-desktop">
    <div class="columns is-centered tldr-row">
      <div class="column is-full">
        <div class="section-text tldr-box">
          <p class="is-size-6 has-text-justified">
            <span class="has-text-weight-bold">TL;DR&nbsp;&nbsp;</span>
            <span class="has-text-weight-bold">EMBGuard</span> decouples physical risk reasoning from an embodied agent policy by evaluating each visual observation and candidate action before execution, enabling safer planning without excessive false alarms.
          </p>
        </div>
      </div>
    </div>

    <div class="columns is-centered has-text-centered project-section">
      <div class="column is-full">
        <h2 class="title is-3">Abstract</h2>
        <div class="content has-text-justified section-text">
          <p>MLLM-powered embodied agents deployed in real-world environments encounter physical hazards. However, existing approaches lack explicit mechanisms for identifying hazards and reasoning about action-conditioned risks, leading agents to either miss risky interactions or over-identify risks. We propose <span class="has-text-weight-bold">EMBGuard</span>, the first MLLM-based safety guardrail for embodied agents designed to decouple physical risk reasoning from agent policy. By evaluating a visual observation and action pair, EMBGuard identifies hazardous configurations and provides natural-language explanations of potential risks.</p>
          <p>Alongside EMBGuard, we contribute <span class="has-text-weight-bold">EMBHazard</span>, a training dataset of 15.1K action-conditioned pairs, and <span class="has-text-weight-bold">EMBGuardTest</span>, a benchmark of 329 manually curated real-world scenarios spanning seven physical risk categories. Despite its compact size, EMBGuard achieves performance competitive with proprietary MLLMs while significantly reducing false positives that hinder real-time deployment.</p>
        </div>
      </div>
    </div>

    <div class="columns is-centered has-text-centered project-section">
      <div class="column is-full">
        <h2 class="title is-3">EMBGuard</h2>
        <div class="figure-panel is-motivation">
          <img src="{{ '/assets/img/embguard/motivation_figure_tradeoff.png' | relative_url }}" alt="Safe and hazardous variants of a robot watering plants">
        </div>
        <div class="content has-text-justified section-text">
          <p><span class="has-text-weight-bold">Action-conditioned physical risk.</span> Physical risk does not arise from the environment alone, but from how an agent's action interacts with hazards in the scene. A power strip below a plant is not inherently dangerous; watering the plant creates the risky interaction.</p>
          <p>Given an image observation and a candidate action, EMBGuard determines whether the action poses a risk, identifies the physical risk category, and describes the hazardous configuration in natural language. The guardrail operates independently from the policy model, allowing it to screen actions before execution.</p>
        </div>
      </div>
    </div>

    <div class="columns is-centered has-text-centered project-section">
      <div class="column is-full">
        <h2 class="title is-3">EMBHazard &amp; EMBGuardTest</h2>
        <div class="figure-panel">
          <img src="{{ '/assets/img/embguard/Dataset_Generation.png' | relative_url }}" alt="EMBGuard dataset generation pipeline">
        </div>
        <div class="content has-text-justified section-text">
          <p><span class="has-text-weight-bold">Dataset construction.</span> We define a risk taxonomy grounded in real-world incident reports, generate risk-driven scenarios, produce controlled compositional variants, and verify the rendered images. The resulting EMBHazard training set contains 15.1K action-conditioned pairs, while EMBGuardTest contains 329 manually curated real-world scenarios.</p>
        </div>

        <h3 class="title is-4 subsection-title">Dataset Statistics</h3>
        <div class="figure-panel is-statistics">
          <img src="{{ '/assets/img/embguard/data_statistics.png' | relative_url }}" alt="Risk category and scenario type distributions in EMBHazard and EMBGuardTest">
        </div>
        <div class="content has-text-justified section-text">
          <p><span class="has-text-weight-bold">Balanced risk coverage.</span> EMBHazard and EMBGuardTest cover seven physical risk categories and four scenario conditions: Causal Risky, Selective Risky, Decoupled Benign, and Absent Benign. This distribution tests whether a model understands action-conditioned hazards rather than reacting to visually salient objects alone.</p>
        </div>
      </div>
    </div>

    <div class="columns is-centered has-text-centered project-section">
      <div class="column is-full">
        <h2 class="title is-3">Results</h2>
        <div class="figure-panel">
          <img src="{{ '/assets/img/embguard/benchmark-results.png' | relative_url }}" alt="Performance comparison of EMBGuard and general-purpose multimodal language models">
        </div>
        <div class="content has-text-justified section-text">
          <p><span class="has-text-weight-bold">Safety guardrail performance.</span> Despite their compact 2B and 4B sizes, EMBGuard models remain competitive with substantially larger open and proprietary MLLMs. General-purpose models frequently over-identify risk, whereas EMBGuard better balances risk detection and precision for practical deployment.</p>
        </div>
      </div>
    </div>
  </div>
</section>

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@inproceedings{choi2026embguard,
  title={EMBGuard: Constructing Hazard-Aware Guardrails for Safe Planning in Embodied Agents},
  author={Choi, Dongwook and Kwon, Taeyoon and Jeong, Bogyung and Kim, Minju and Hwang, Yeonjun and Kim, Hyojun and Kim, Byungchul and Jang, Young Kyun and Yeo, Jinyoung},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning},
  year={2026}
}</code></pre>
  </div>
</section>

<footer class="footer">
  <div class="container">
    <div class="columns is-centered">
      <div class="column is-8">
        <div class="content">
          <p>This website is licensed under a <a rel="license" href="https://creativecommons.org/licenses/by-sa/4.0/" target="_blank" rel="noopener noreferrer">Creative Commons Attribution-ShareAlike 4.0 International License</a>.</p>
        </div>
      </div>
    </div>
  </div>
</footer>
