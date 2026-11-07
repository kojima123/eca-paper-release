# Reference verification (checked 2026-11-03)

Status key: VERIFIED = the guess matched the primary source (fields may have been added from it); CORRECTED = something in the guess was wrong; NOT FOUND = no primary source found.
Author rule: all authors if 6 or fewer, otherwise the first 3 + et al.
Sources: arXiv abstract pages and the arXiv API; Crossref metadata (the record the publisher deposits behind each DOI); doi.org (every DOI below was checked and redirects to the publisher); publisher pages (PMLR, NeurIPS proceedings, AAAI OJS, JAIR); OpenReview; the official project and competition pages; Kaggle.
Springer link pages were behind a bot check, so Springer fields come from Crossref. Library catalogue records (CiNii) were used for the two pre-DOI books.

**Totals: 34 VERIFIED, 1 CORRECTED (#35), 0 NOT FOUND.**

---

### 1. Chollet 2019 — VERIFIED
- Chollet, F. (2019). On the Measure of Intelligence. arXiv:1911.01547 [cs.AI]. (v1 2019-11-05; v2 2019-11-25)
- DOI: none (arXiv only)
- Checked: https://arxiv.org/abs/1911.01547

### 2. ARC-AGI-3 benchmark (official reference) — VERIFIED (found)
- ARC Prize Foundation (2026). ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence. arXiv:2603.24621 [cs.AI]. (v1 2026-03-24; v2 2026-04-17)
- DOI: none
- Checked: https://arxiv.org/abs/2603.24621 ; https://arcprize.org/arc-agi/3 (its "Technical Paper" link points to https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf)
- Notes: The only author listed is the corporate author "ARC Prize Foundation". The PDF on arcprize.org is dated April 22, 2026 and has the same title; a footnote names the development team (lead environment designer Hunter Henry; ...; François Chollet, Mike Knoop, Gregory Kamradt). Neither arcprize.org nor the paper gives a "how to cite" text or BibTeX, so the arXiv record is the canonical citable form. The same paper cites the competitions in this form: "ARC Prize 2024 Competition. https://arcprize.org/competitions/2024, 2024."

### 3. ARC Prize 2026 competition (website) — VERIFIED
- ARC Prize (2026). ARC Prize 2026. ARC Prize, Inc. https://arcprize.org/competitions/2026 (accessed 2026-11-03).
  - Shorter form, in the organizers' own style (see #2): "ARC Prize 2026 Competition. https://arcprize.org/competitions/2026, 2026."
- Kaggle track page: "ARC Prize 2026 - ARC-AGI-3", https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3 (host shown as "Abstraction and Reasoning Corpus"; opened 2026-03-25; final submission deadline 2026-11-02 23:59 UTC).
- Checked: https://arcprize.org/competitions/2026 (HTML title "ARC Prize 2026"; footer "© 2026 ARC Prize, Inc."); https://arcprize.org/competitions/2026/arc-agi-3 ; Kaggle competition metadata.
- Notes: The page gives three tracks (ARC-AGI-3, ARC-AGI-2, Paper Prize) and these dates: start Mar 25; Milestone #1 Jun 30; Milestone #2 Sep 30; submissions Nov 2; papers Nov 8; results Dec 4, 2026. Neither page gives a recommended citation.

### 4. Logan 1988 — VERIFIED
- Logan, G. D. (1988). Toward an instance theory of automatization. Psychological Review, 95(4), 492–527.
- DOI: 10.1037/0033-295X.95.4.492
- Checked: https://doi.org/10.1037/0033-295X.95.4.492 ; https://api.crossref.org/works/10.1037/0033-295x.95.4.492

### 5. Schmidt 1975 — VERIFIED
- Schmidt, R. A. (1975). A schema theory of discrete motor skill learning. Psychological Review, 82(4), 225–260.
- DOI: 10.1037/h0076770
- Checked: https://doi.org/10.1037/h0076770 ; Crossref

### 6. Norman & Shallice 1986 — VERIFIED (pages and DOI added)
- Norman, D. A., & Shallice, T. (1986). Attention to action: Willed and automatic control of behavior. In R. J. Davidson, G. E. Schwartz, & D. Shapiro (Eds.), Consciousness and Self-Regulation: Advances in Research and Theory (Vol. 4, pp. 1–18). New York: Plenum Press.
- DOI: 10.1007/978-1-4757-0629-1_1 (Springer US digital edition; ISBN 978-1-4757-0631-4)
- Checked: https://doi.org/10.1007/978-1-4757-0629-1_1 ; https://api.crossref.org/works/10.1007/978-1-4757-0629-1 (editors and "Advances in Research and Theory Volume 4")
- Notes: Crossref now lists the publisher as "Springer US, Boston, MA". The original imprint, Plenum (New York), comes from library catalogue records found by search, not from the publisher page.

### 7. Wolpert, Ghahramani & Jordan 1995 — VERIFIED
- Wolpert, D. M., Ghahramani, Z., & Jordan, M. I. (1995). An internal model for sensorimotor integration. Science, 269(5232), 1880–1882.
- DOI: 10.1126/science.7569931
- Checked: https://doi.org/10.1126/science.7569931 ; Crossref

### 8. Daw, Niv & Dayan 2005 — VERIFIED
- Daw, N. D., Niv, Y., & Dayan, P. (2005). Uncertainty-based competition between prefrontal and dorsolateral striatal systems for behavioral control. Nature Neuroscience, 8(12), 1704–1711.
- DOI: 10.1038/nn1560
- Checked: https://doi.org/10.1038/nn1560 ; Crossref (online 2005-11-06; print Dec 2005)

### 9. Anderson 1982 — VERIFIED
- Anderson, J. R. (1982). Acquisition of cognitive skill. Psychological Review, 89(4), 369–406.
- DOI: 10.1037/0033-295X.89.4.369
- Checked: https://doi.org/10.1037/0033-295X.89.4.369 ; Crossref

### 10. Fitts & Posner 1967 — VERIFIED
- Fitts, P. M., & Posner, M. I. (1967). Human Performance. Belmont, CA: Brooks/Cole. (Basic Concepts in Psychology Series; x + 162 pp.)
- DOI: none
- Checked: https://ci.nii.ac.jp/ncid/BA05593082 (CiNii catalogue record: "Human performance / Paul M. Fitts [and] Michael I. Posner. Belmont: Brooks/Cole, c1967")

### 11. Newell & Rosenbloom 1981 — VERIFIED (pages added)
- Newell, A., & Rosenbloom, P. S. (1981). Mechanisms of skill acquisition and the law of practice. In J. R. Anderson (Ed.), Cognitive Skills and Their Acquisition (pp. 1–55). Hillsdale, NJ: Lawrence Erlbaum Associates.
- DOI: none for the 1981 printing. The Psychology Press reissue has 10.4324/9780203728178-6.
- Checked: https://ci.nii.ac.jp/ncid/BA0039298X (book: L. Erlbaum Associates, Hillsdale, N.J., 1981, xiii + 386 pp., ISBN 0-89859-093-0; 16th Carnegie Symposium on Cognition, 1980); https://api.crossref.org/works/10.4324/9780203728178-6 (chapter in the reissue)
- Notes: The pages 1–55 come from a library record (SWOV) seen in search results; the page itself was behind a bot check. The 2013 reissue uses different pagination, so cite the 1981 pages.

### 12. McCloskey & Cohen 1989 — VERIFIED
- McCloskey, M., & Cohen, N. J. (1989). Catastrophic interference in connectionist networks: The sequential learning problem. In G. H. Bower (Ed.), Psychology of Learning and Motivation (Vol. 24, pp. 109–165). Academic Press.
- DOI: 10.1016/S0079-7421(08)60536-8
- Checked: https://doi.org/10.1016/S0079-7421(08)60536-8 ; Crossref (pages 109–165, ISBN 978-0-12-543324-2, 1989); Open Library ISBN record (ed. Gordon H. Bower, Academic Press, Nov 1989)
- Notes: Crossref does not carry the volume number. Vol. 24 is supported only by secondary bibliographies (ScienceDirect was blocked). The place of publication was not confirmed, so add it only after checking if your style requires it.

### 13. Kirkpatrick et al. 2017 — VERIFIED
- Kirkpatrick, J., Pascanu, R., Rabinowitz, N., et al. (2017). Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy of Sciences, 114(13), 3521–3526.
- DOI: 10.1073/pnas.1611835114
- Checked: https://doi.org/10.1073/pnas.1611835114 ; Crossref (14 authors)

### 14. Liu et al. 2024 (Lost in the Middle) — VERIFIED
- Liu, N. F., Lin, K., Hewitt, J., et al. (2024). Lost in the Middle: How Language Models Use Long Contexts. Transactions of the Association for Computational Linguistics, 12, 157–173.
- DOI: 10.1162/tacl_a_00638 ; arXiv:2307.03172
- Checked: https://doi.org/10.1162/tacl_a_00638 ; Crossref ; https://arxiv.org/abs/2307.03172 (7 authors)

### 15. Wang et al. 2023/2024 (Voyager) — VERIFIED (TMLR confirmed)
- Wang, G., Xie, Y., Jiang, Y., et al. (2024). Voyager: An Open-Ended Embodied Agent with Large Language Models. Transactions on Machine Learning Research (TMLR). arXiv:2305.16291.
- DOI: none
- Checked: https://arxiv.org/abs/2305.16291 (v1 2023-05-25, v2 2023-10-19; 8 authors); OpenReview https://openreview.net/forum?id=ehfRiF0R3a (venue "Accepted by TMLR", posted 2024-03)
- Notes: TMLR has no volume or page numbers. To cite the preprint only, use (2023), arXiv:2305.16291.

### 16. SkillDroid 2026 — VERIFIED
- Chen, Q., Bellucci, A., Sun, Z., & Jacucci, G. (2026). SkillDroid: Compile Once, Reuse Forever. arXiv:2604.14872.
- Checked: https://arxiv.org/abs/2604.14872 (2026-04-16; authors Qijia Chen, Andrea Bellucci, Zhida Sun, Giulio Jacucci)

### 17. MobiMem 2025 — VERIFIED
- Liu, Z., Zhang, C., Zhao, X., et al. (2025). Beyond Training: Enabling Self-Evolution of Agents with MOBIMEM. arXiv:2512.15784.
- Checked: https://arxiv.org/abs/2512.15784 (2025-12-15; 9 authors, first: Zibin Liu, Cheng Zhang, Xi Zhao)
- Notes: The arXiv title writes the name "MOBIMEM" in capitals.

### 18. Schema harness (Zeng et al. 2026) — VERIFIED (+ a later full paper exists)
- Project report (blog): Zeng, G., Wang, J., Ma, W., et al. (2026, July 15). Frontier Models with Our Harness Achieve ~99% on ARC-AGI-3 Public. Impossible Research. https://schema-harness.github.io/
  - The page's own BibTeX ("schema2026blog") gives the title as "[schema]: Frontier Models with the Right Harness Achieve ∼99% on ARC-AGI-3 Public" and howpublished "Impossible Research".
- Full paper (preferred if citing the method): Zeng, G., Wang, J., Ma, W., et al. (2026). Schema: Discovering Unknown Environments via Agentic Program Induction. arXiv:2609.39140 [cs.AI]. (2026-09-30)
- Checked: https://schema-harness.github.io/ (dated July 15, 2026; "Update Sep. 30, 2026: full paper & code release"); https://arxiv.org/abs/2609.39140 ; https://guanning03.github.io/schema-webpage/
- Notes: There are 11 authors: Guanning Zeng, Jiani Wang, Wenjie Ma, Shaofeng Yin, Chenyang Wang, Shichen Liu, Angjoo Kanazawa, Wode Ni, Xiuyu Li, Andrea Zanette, Haiwen Feng. The visible blog title ("Our Harness") and its BibTeX title ("the Right Harness") differ; quote whichever one you cite.

### 19. Rodionov 2026 (arXiv:2605.05138) — VERIFIED (title/author found; also published)
- Rodionov, S. (2026). Executable World Models for ARC-AGI-3 in the Era of Coding Agents. arXiv:2605.05138. Published in: Artificial General Intelligence, 19th International Conference, AGI 2026, San Francisco, CA, USA, July 27–30, 2026, Proceedings, Part II (M. Iklé, A. Franz, A. Kemendo, H. Lowy, Eds.), Lecture Notes in Computer Science, pp. 198–210. Cham: Springer.
- DOI: 10.1007/978-3-032-33195-3_15
- Checked: https://arxiv.org/abs/2605.05138 (v2 2026-06-06, "Accepted for publication at AGI-2026"); https://api.crossref.org/works/10.1007/978-3-032-33195-3_15 and the book record 10.1007/978-3-032-33195-3
- Notes: Your description matches the arXiv full text: "If the prediction diverges from the observation, the executor stops immediately and records mismatch artifacts." The LNCS volume number was not confirmed from a primary source (Springer was blocked), so leave it out or mark it unverified. The same author has a follow-up ablation paper, arXiv:2607.15439, "Do Coding Agents Need Executable World Models, Simplification, and Verification to Solve ARC-AGI-3?"; do not mix the two up.

### 20. Twin (arXiv:2608.14490) — VERIFIED (title/authors found)
- Skoutnev, A., Acharya, K., Longhitano, G., Udell, M., Ellis, K., & Drori, I. (2026). Twin: Playing an Unknown Game with a Test-Time Digital Twin. arXiv:2608.14490.
- Checked: https://arxiv.org/abs/2608.14490 (2026-08-14)
- Notes: The abstract confirms harness-enforced validation: "The harness enforces that an action is not made until the program reproduces every previous observed game transition."

### 21. Tycho (arXiv:2607.28287) — VERIFIED (title/authors found)
- Lehmann, J., Aioanei, A., & Vahdati, S. (2026). Tycho: Active Abstraction with Programmatic World Models for ARC-AGI-3. arXiv:2607.28287.
- Checked: https://arxiv.org/abs/2607.28287 (2026-07-30)
- Notes: The paper's own wording is that it formalizes the environments (games) as "parameterized rendered deterministic Moore machines" (Section 3). Use that phrase rather than "levels as Moore machines".

### 22. SwiftSage (Lin et al. 2023) — VERIFIED
- Lin, B. Y., Fu, Y., Yang, K., et al. (2023). SwiftSage: A Generative Agent with Fast and Slow Thinking for Complex Interactive Tasks. In Advances in Neural Information Processing Systems 36 (NeurIPS 2023), pp. 23813–23825. arXiv:2305.17390.
- DOI: 10.52202/075280-1034 (proceedings DOI)
- Checked: https://proceedings.neurips.cc/paper_files/paper/2023/hash/4b0eea69deea512c9e2c469187643dc2-Abstract-Conference.html ; https://arxiv.org/abs/2305.17390 (9 authors; spotlight)

### 23. SOFAI — VERIFIED (main reference = Booch et al. 2021)
- Main: Booch, G., Fabiano, F., Horesh, L., et al. (2021). Thinking Fast and Slow in AI. Proceedings of the AAAI Conference on Artificial Intelligence, 35(17), 15042–15046. arXiv:2010.06002.
- DOI: 10.1609/aaai.v35i17.17765
- Checked: https://ojs.aaai.org/index.php/AAAI/article/view/17765 ; https://arxiv.org/abs/2010.06002 (11 authors)
- SOFAI architecture paper led by Fabiano (optional second citation): Fabiano, F., Pallagani, V., Ganapini, M. B., et al. (2023). Fast and Slow Planning. arXiv:2303.04283 (8 authors). Checked: https://arxiv.org/abs/2303.04283. No peer-reviewed venue was confirmed for it.
- Related (not checked further): Ganapini, M. B., Campbell, M., Fabiano, F., et al. (2021). Thinking Fast and Slow in AI: the Role of Metacognition. arXiv:2110.01834.
- Notes: The AAAI page spells some names "Andreas Loreggia" and "Keerthiram Murgesan"; arXiv has "Andrea Loreggia" and "Keerthiram Murugesan". With et al. after the third author, this does not affect the citation.

### 24. AdaPlanner (Sun et al. 2023) — VERIFIED
- Sun, H., Zhuang, Y., Kong, L., Dai, B., & Zhang, C. (2023). AdaPlanner: Adaptive Planning from Feedback with Language Models. In Advances in Neural Information Processing Systems 36 (NeurIPS 2023), pp. 58202–58245. arXiv:2305.16653.
- DOI: 10.52202/075280-2537 (proceedings DOI)
- Checked: https://proceedings.neurips.cc/paper_files/paper/2023/hash/b5c8c1c117618267944b2617add0a766-Abstract-Conference.html ; https://arxiv.org/abs/2305.16653

### 25. WebATLAS (arXiv:2510.22732) — VERIFIED (title/authors found)
- Cheng, J., Kumar, A., Lal, R., et al. (2025). WebATLAS: An LLM Agent with Experience-Driven Memory and Action Simulation. NeurIPS 2025 Workshop on Language Agents and World Models. arXiv:2510.22732.
- Checked: https://arxiv.org/abs/2510.22732 (v1 2025-10-26, v2 2025-12-19; 10 authors; venue from the arXiv comment)

### 26. Akyürek et al. (test-time training) — VERIFIED (title and venue settled)
- Akyürek, E., Damani, M., Zweiger, A., et al. (2025). The Surprising Effectiveness of Test-Time Training for Few-Shot Learning. In Proceedings of the 42nd International Conference on Machine Learning (ICML 2025), PMLR 267, 942–963. arXiv:2411.07279.
- Checked: https://proceedings.mlr.press/v267/akyurek25a.html ; https://arxiv.org/abs/2411.07279 (8 authors)
- Notes: The published title is "...for Few-Shot Learning". arXiv v1 (2024-11-11) was titled "...for Abstract Reasoning"; v2 (2025-03-25) and the ICML version use "Few-Shot Learning". Spell the first author "Akyürek".

### 27. SOAR (Pourcel, Colas, Oudeyer 2025) — VERIFIED (ICML 2025 confirmed)
- Pourcel, J., Colas, C., & Oudeyer, P.-Y. (2025). Self-Improving Language Models for Evolutionary Program Synthesis: A Case Study on ARC-AGI. In Proceedings of the 42nd International Conference on Machine Learning (ICML 2025), PMLR 267, 49659–49688. arXiv:2507.14172.
- Checked: https://proceedings.mlr.press/v267/pourcel25a.html ; OpenReview (ICML 2025 poster); arXiv API

### 28. Sha 2001 (Simplex) — VERIFIED
- Sha, L. (2001). Using simplicity to control complexity. IEEE Software, 18(4), 20–28.
- DOI: 10.1109/MS.2001.936213
- Checked: https://doi.org/10.1109/MS.2001.936213 ; Crossref

### 29. Machado et al. 2018 — VERIFIED
- Machado, M. C., Bellemare, M. G., Talvitie, E., Veness, J., Hausknecht, M., & Bowling, M. (2018). Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents. Journal of Artificial Intelligence Research, 61, 523–562.
- DOI: 10.1613/jair.5699 ; arXiv:1709.06009
- Checked: https://jair.org/index.php/jair/article/view/11182 ; Crossref ; arXiv full text
- Notes: The paper text confirms that it introduces sticky actions and uses the Brute. It also says the Brute was "first briefly introduced in Bellemare et al., 2015", so cite that paper too if you attribute the Brute's origin.

### 30. Vaandrager et al. 2022 (L#) — VERIFIED
- Vaandrager, F., Garhewal, B., Rot, J., & Wißmann, T. (2022). A New Approach for Active Automata Learning Based on Apartness. In D. Fisman & G. Rosu (Eds.), Tools and Algorithms for the Construction and Analysis of Systems (TACAS 2022), Part I, Lecture Notes in Computer Science, vol. 13243, pp. 223–243. Cham: Springer.
- DOI: 10.1007/978-3-030-99524-9_12 ; arXiv:2107.05419
- Checked: https://doi.org/10.1007/978-3-030-99524-9_12 ; Crossref chapter and book records (Part I, editors, pages)
- Notes: The volume number 13243 comes from library catalogue records for TACAS 2022 Part I seen in search results. The Springer page was blocked, so this is not a primary-source confirmation.

### 31. Angluin 1987 — VERIFIED
- Angluin, D. (1987). Learning regular sets from queries and counterexamples. Information and Computation, 75(2), 87–106.
- DOI: 10.1016/0890-5401(87)90052-6
- Checked: https://doi.org/10.1016/0890-5401(87)90052-6 ; Crossref

### 32. Yao et al. 2024 (τ-bench) — VERIFIED (+ ICLR 2025)
- Yao, S., Shinn, N., Razavi, P., & Narasimhan, K. (2024). τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains. arXiv:2406.12045. (Published at ICLR 2025, poster.)
- Checked: https://arxiv.org/abs/2406.12045 (the abstract introduces the pass^k metric); OpenReview https://openreview.net/forum?id=roNSXZpUDN (ICLR 2025 Poster)
- Notes: For the conference version, cite (2025) In International Conference on Learning Representations (ICLR 2025).

### 33. Laird, Rosenbloom & Newell 1986 — VERIFIED
- Laird, J. E., Rosenbloom, P. S., & Newell, A. (1986). Chunking in Soar: The anatomy of a general learning mechanism. Machine Learning, 1(1), 11–46.
- DOI: 10.1007/BF00116249 (the Springer DOI; Crossref also has a duplicate Kluwer DOI, 10.1023/A:1022639103969)
- Checked: https://doi.org/10.1007/BF00116249 ; Crossref

### 34. Daniel Franzen, ARC-AGI-3 Milestone #2 notebook — VERIFIED (found)
- Notebook: Franzen, D. (2026). ARC-AGI-3 Milestone 2 Solution [Kaggle notebook]. https://www.kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution (Kaggle user dfranzen; the winning version is scriptVersionId=354090904; latest version 2026-10-03). Code: https://github.com/da-fr/arc-agi-3-solution (write-up: WRITEUP.md).
- Announcement: ARC Prize (@arcprize), "ARC Prize 2026: ARC-AGI-3 Milestone #2 Winners", post on X, 2026-10-01, https://x.com/arcprize/status/2105737436450201734. It reads: "1. Daniel Franzen - 27.9%, $25K; 2. Lord Han Solo - 23.8%, $7.5K; 3. Lohit Siriki - 22.5%, $5K".
- Checked: the Kaggle notebook page (title "ARC-AGI-3 Milestone 2 Solution"); the Kaggle notebook listing for dfranzen; the X post text (via X's public embed endpoint); Kaggle discussion "Comparing the Milestone 2 winning solutions" (Tong Hui Kang, 2026-10-01), https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/744792
- Notes: The arcprize.org blog has no Milestone #2 post (only arc-prize-2026-milestone-1), so the X post is the official announcement. The prize amounts it gives for 2nd and 3rd ($7.5K / $5K) differ from the $10K / $2.5K on the competition page. According to the comparison thread, all three winners run Tufa Labs' Duck harness on TAAF with Qwen3.8-Flash-Next. This comes from community sources, not ARC Prize.

### 35. The "Duck" harness / TAAF — CORRECTED (the notebook owner is jeroencottaar, not driessmit1)
- Notebook: Cottaar, J. / Tufa Labs (2026). Tufa Labs duck harness [June 30 milestone winner] [Kaggle notebook]. https://www.kaggle.com/code/jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner (the older slug /code/jeroencottaar/taaf-duck-harness-kaggle-share redirects here)
- Write-up: Bessis, H., Cottaar, J., Pressman, I., Smit, D., Tešnar, M., & Viel, S. (2026, July 1). Tufa Labs' Winning Solution for ARC-AGI-3 Milestone 1. Kaggle discussion, https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/717133 (authors listed alphabetically in the post)
- Code (contains TAAF): Tufa Labs. The Duck: ARC-AGI-3 inference harness. https://github.com/Tufalabs/duck-harness. The folder `tufa-arc-agi-framework/` holds TAAF; its pyproject.toml says name "tufa-arc-agi-framework", "ARC-AGI3 orchestration framework (TAAF) by Tufa Labs". On Kaggle, TAAF ships as the dataset "TAAF Kaggle Source Bundle", https://www.kaggle.com/datasets/jeroencottaar/taaf-kaggle-source-share
- Official recognition: Kamradt, G. (2026, July 6). ARC Prize 2026: ARC-AGI-3 Milestone Prize #1. https://arcprize.org/blog/arc-prize-2026-milestone-1 (1st place: Tufa Labs, "The Duck", linking the jeroencottaar notebook above)
- What changed: Kaggle user driessmit1 is Dries Smit (Tufa Labs). He is a co-author of the Duck write-up and the author of the predecessor agent "Stochastic Goose", but he has no public Kaggle notebooks (his notebook listing returned 0). He owns two datasets the Duck notebook uses: driessmit1/arc3-vllm-h100-wheelhouse-v3 and driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot. The public Duck/TAAF notebook belongs to jeroencottaar. To credit Dries Smit specifically, the ARC-AGI-3 technical report (#2) cites "Dries Smit. ARC3 Solution. https://github.com/DriesSmit/ARC3-solution, 2025" (the preview-competition agent).
- Checked: the Kaggle pages and listings above; https://github.com/Tufalabs/duck-harness (README, pyproject.toml); https://arcprize.org/blog/arc-prize-2026-milestone-1

## Addendum 2026-11-07: citations added or re-verified for commit 41991aa2 and after

- Lehmann, J., Aioanei, A., Vahdati, S. (2026). Tycho. arXiv:2607.28287. Re-read in the arXiv HTML (v1) on 2026-11-07. Matched policy comparison with Claude Opus 4.8 on the 25 public games (Table 1 / Section 5.1): no world model 79.07 RHAE, single 85.36, orchestrator (actor-requested delegation) 88.49, trigger (automatic repair after verification failures) 83.07. Verbatim: "Accurate transition prediction therefore does not by itself ensure strong game-playing performance: when the model is built and how its advice is used also matter." The paper cites 83.07 and 88.49 from this table.
- Skoutnev, A., Acharya, K., Longhitano, G., Udell, M., Ellis, K., Drori, I. (2026). Twin. arXiv:2608.14490, v1 submitted 2026-08-14. Title, author order and abstract confirmed on arxiv.org/abs/2608.14490 on 2026-11-07 (the harness does not act until the program reproduces every previous observed transition; 179 of 183 levels).
- Zeng, G., et al. (2026). Schema. arXiv:2609.39140, submitted 2026-09-30. Confirmed on arxiv.org/abs/2609.39140 and github.com/guanning03/Schema on 2026-11-07.
- Sestak, F., Aichberger, L., Ielanskyi, M., Klambauer, G. (2026). Kaggle discussion 756428, "[Public LB: 7] the last dance - Short Overview", posted 2026-11-06T08:02Z. The post opens with the author line "Florian Sestak, Lukas Aichberger, Mykyta Ielanskyi, Günter Klambauer - Institute for Machine Learning, Johannes Kepler University, Linz" and reports 7th on the public leaderboard with 54.52. Full text retrieved through the Kaggle API (competition_list_topic_messages) on 2026-11-07. The cited passage, verbatim, from its section "6. What Did Not Work": "World modeling: We tried it three times: model-written world models evolved as a population, a state program the agent kept up to date, and the game model as code with a planner on top. The models fit the recorded transitions well, but the agent did not clear more levels." Their model and harness differ from ours (REAP-pruned Qwen3.8-Flash-Next, 4-bit KV cache, a Pydantic AI port of the Duck harness), so the paper cites it as a similar report, not a replication.
- Bergamaschi Ganapini, M., Campbell, M., Fabiano, F., et al. (2025). Fast, slow, and metacognitive thinking in AI. npj Artificial Intelligence, 1, 27. doi:10.1038/s44387-025-00027-5. Confirmed on nature.com/articles/s44387-025-00027-5 on 2026-11-07 (SOFAI: fast and slow solvers with a metacognitive module choosing between them). Added as the source for SOFAI's architecture; Booch et al. (2021) remains the source for the fast/slow proposal.
- Removed: Yao et al. (2025), tau-bench, which the current text no longer cites.
- Bessis et al. (2026): the notebook reference now carries the full slug kaggle.com/code/jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner (listed through the Kaggle API on 2026-11-07, last run 2026-07-01).
- Serving stack (Section 2): the submitted and local kernels launch SGLang (the `sglang` binary and SGLANG_* environment, with the Intel AutoRound W4A16 Qwen3.8-Flash-Next checkpoint and albucino's INT4 g32 MTP draft checkpoint); the earlier text said vLLM, which is only the name of the OpenAI-compatible provider setting in the harness. Corrected on 2026-11-07.
