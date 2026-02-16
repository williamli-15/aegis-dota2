我先把你给的 OpenDota match schema 做一次“可建模资产盘点”，然后给出一个**纯结构化、ProAct-style（proactive / plan-then-act / 可离线评测）**的 Dota 2 方案，写到 **PRD 级别**：目标、产品形态、数据/标签、模型、评测、工程架构、里程碑一次写全。

---

# PRD：Aegis（暂名）——基于 OpenDota 结构化日志的 Dota 2 Proactive 决策与复盘系统

## 0. 一句话愿景

把 Dota 2 的比赛日志变成一个**可“预测未来—生成计划—评估反事实—输出可执行建议”**的 Proactive 决策系统：既能做**复盘教练**，也能成为**研究级可评测的“行动规划器”**（先结构化、先离线闭环）。

---

## 1. 你这份数据里，真正“值钱”的东西是什么？

你给的 match schema（OpenDota /matches/{match_id}）里，适合做 **proactive** 的不是“全局统计”，而是这些具有**时间戳 + 可还原决策点**的结构：

### A. 时间线骨架（全局节奏与胜负因果钩子）

* `objectives[]`：塔/基地/人头里程碑/聊天轮盘等带时间的事件

  * **价值**：宏观目标推进的“地标”，可以做 *objective planning*（什么时候该推塔/抱团/换线）。
* `radiant_gold_adv[]`, `radiant_xp_adv[]`：优势曲线（通常按分钟）

  * **价值**：训练一个**价值函数/胜率代理**（Value / WinProb），也能做“拐点检测”（什么时候局势崩了）。
* `teamfights[]`：团队战窗口（start/end/last_death）+ 每人 fight 内统计

  * **价值极高**：天然提供“战斗片段切分”，可以做：

    * 参战/不参战决策（participation）
    * fight 前状态 → fight 结果（团战胜率/损失预测）
    * fight 内“技能/物品使用模式”（结构化 micro 的压缩表示）

### B. 玩家“决策痕迹”（最像策略动作的可学习信号）

每个 `players[i]` 下这些是**可做动作空间**的核心：

* `purchase_log[]`（time, key） + `purchase_time`, `first_purchase_time`

  * **价值**：**物品路线（build order）**是 Dota 里最稳定、最可解释、最可评测的策略动作之一。
* `ability_upgrades_arr[]`

  * **价值**：技能加点路线（skill build）同样稳定、可评测、可个性化。
* `obs_log[]`, `sen_log[]` + `obs_left_log[]`, `sen_left_log[]`（带 x,y,time）

  * **价值爆炸**：**插眼决策=时间+坐标**，并且你还有“被清/自然消失”的 left log，可直接定义“眼的收益”指标（存活时间、覆盖区域、与击杀/反击的相关性）。
* `kills_log[]`, `runes_log[]`

  * **价值**：gank/节奏与地图资源控制的时间线信号，尤其可用于“主动性”评测：是不是提前控符/提前做视野/提前打架。
* `lane_pos`（网格计数热力图）

  * **价值**：虽然不是连续轨迹，但足够捕捉“长期站位/转线倾向”，对宏观计划（去哪打钱/去哪带线/去哪蹲）很有用。
* `actions` / `actions_per_min`

  * **价值**：更像“操作强度/行为指纹”，可作为**玩家水平/风格条件变量**，做“skill-conditioned policy”（同一状态下，高水平会怎么做）。

### C. 不太值钱但可当条件/过滤器的

* `chat[]`：可用于情绪/嘲讽相关分析，但你说先不多模态/不搞文本，这块先不做核心。
* `cosmetics`、`draft_timings`（你样例为空）先忽略。

---

## 2. 产品定义：我们到底在“做什么产品”？

### 2.1 产品形态（MVP → Pro）

**MVP（最强闭环，最可落地）**：

> **复盘教练 + 反事实建议引擎**
> 输入：一场 match（OpenDota JSON）
> 输出：一条“可执行的时间线建议”，聚焦 3 类高杠杆决策：

1. **下一件/下两件装备**（含替代分支）
2. **视野计划（下一分钟插哪里 / 为什么 / 预期收益）**
3. **团战参战与目标选择（这波该来/不该来；来了怎么打）**

**Pro（你想要的 visionary）**：

> **Proactive Planner（短视野→长视野）**
> 给定任意时间点的状态摘要，系统生成：

* **未来 T 秒的“宏观计划（plan tokens）”**：比如 *farm triangle → smoke mid → take tower*
* 每步计划的**价值评估**与**失败模式**（反事实）
* 在离线数据上可严格打分：计划是否提前、是否对齐高水平分布、是否带来价值上升

> 关键点：先不做多模态、也不做真实游戏内控制。我们把“计划与动作”定义在**日志可观测**的空间里（购买、插眼、参战、目标推进等），这样才能像论文一样**可评测、可复现、可迭代**。

---

## 3. 目标与非目标

### 3.1 目标（Phase 1–2）

* **G1：建立 Dota2 Structured Trajectory 数据集**（从 raw match JSON → 统一事件表/状态快照表）
* **G2：训练一个可用的价值函数/胜率代理**

  * 输入：结构化状态（10 人汇总 + 时间/经济/阵容/关键物品）
  * 输出：未来 K 分钟后的胜负/优势变化（可多头）
* **G3：训练 2–3 个高价值策略动作模型（policy）**

  * Next-item policy（装备序列）
  * Next-ward policy（眼位序列）
  * Fight participation policy（参战/不参战）
* **G4：形成 Proactive 评测闭环**

  * “提前量”指标：模型在事件发生前多久就给出正确计划
  * “价值增益”指标：候选计划经价值函数评估后，是否优于真实选择/基线

### 3.2 非目标（先明确不做）

* 不做多模态（图像/小地图/语音）
* 不做实时游戏外挂式提示（合规与工程都先不碰）
* 不做 micro 级操作建议（走位/连招），你当前数据也不支持高保真轨迹

---

## 4. 数据工程：从 OpenDota JSON 到“可训练”的统一表示

### 4.1 原始数据获取

你已有：

* 拉 match_id 列表：`GET /api/parsedMatches?limit=200`
* 拉单场细节：`GET /api/matches/{match_id}`

**PRD 要求的数据管道能力：**

* 增量抓取（按时间/按 match_id 去重）
* 原始 JSON 落地（对象存储/分区）
* ETL 生成训练表
* 数据版本与 patch 对齐（`patch`, `start_time`）

### 4.2 标准化成 5 张核心表（建议）

1. `matches`（match-level）

* match_id, start_time, duration, patch, region, game_mode, radiant_win, …

2. `players`（match-player-level，10 行/场）

* match_id, player_slot, team, hero_id, lane, lane_role, rank_tier / computed_mmr, …
* 最终 KDA、GPM、XPM、net_worth（做 outcome & 过滤）

3. `timeseries_minute`（每分钟快照，10*minutes 行/场）

* match_id, t_minute, player_slot
* gold_t, xp_t, lh_t, dn_t（你数据里有 `times[]` 对齐）
* team 的 radiant_gold_adv[t], radiant_xp_adv[t]
* item slots（如果能从 OpenDota拿到每分钟物品快照最好；拿不到就用 purchase_log 回放近似）

4. `events`（统一事件流，最关键）
   列建议：

* match_id, t, actor_slot, actor_team
* event_type（PURCHASE / WARD_OBS / WARD_SEN / WARD_LEFT / KILL / DEATH / RUNE / OBJECTIVE / TEAMFIGHT_START / TEAMFIGHT_END …）
* event_key（item_name / ward_type / rune_id / objective_key …）
* x, y（如果有）
* target_slot / target_unit（如果有）

5. `teamfights`（片段表）

* match_id, fight_id, start, end, deaths, …
* per-player fight stats（可拆成 `teamfight_players` 子表）

> 这套表的价值：你可以把任何模型都统一成 “给定状态/历史 → 预测下一步事件/未来片段结果”。

---

## 5. 状态表示（State）与动作空间（Action）怎么定义？

### 5.1 我们先定义“宏观动作词表”（完全结构化）

我们不追求覆盖一切，只覆盖日志里**可观测且影响胜率**的动作：

**Action vocab v1（MVP）：**

1. `BUY(item_id)`：购买某物品（来自 purchase_log）
2. `PLACE_OBS(x_bin, y_bin)`：插真眼
3. `PLACE_SEN(x_bin, y_bin)`：插假眼
4. `FIGHT_JOIN(fight_id)` / `FIGHT_SKIP(fight_id)`：团战参与（从 teamfights/玩家 fight damage/ability_uses 推断参与）
5. `OBJECTIVE_COMMIT(type, key)`：推进目标（塔/高地/基地），用 objectives 的时间点做监督信号

**Action vocab v2（Pro）：**

* `SMOKE()`（从 purchase_log 或 item_uses 里 smoke）
* `TP_USE()`（item_uses: tpscroll）
* `STACK_CAMP()`（如果你有 camps_stacked/creeps_stacked 的时间化信号则可做；目前是计数，先放）
* `RUNE_CONTROL(rune_type)`（从 runes_log）

### 5.2 状态（State）v1：足够强、足够稳、足够可解释

对任意时间点 t，我们构造一个“结构化摘要”：

**全局：**

* time (t), patch, game_mode, region
* radiant_gold_adv(t), radiant_xp_adv(t)
* 最近 N 秒 objectives / kills 的计数特征（节奏）

**每个玩家（10 人）：**

* hero_id, level（由 xp_t 推或直接字段）
* gold/xp/last_hits/denies（从 *_t）
* 当前核心物品 embedding（由 purchase_log 截止到 t 回放）
* skill build embedding（ability_upgrades_arr 截到当前等级）
* role/lane（lane, lane_role）
* ward 行为历史（过去 M 分钟插眼次数、存活率）
* fight 行为历史（过去 K 波团战的参与/输出）

**对抗信息（关键）：**

* 阵容对阵（英雄集合）
* 关键道具差（例如 BKB、Blink、Pipe 等的有无——先用 item_id 集合硬规则提取）

> 你会发现：这个 state 不需要真实坐标轨迹，也能做出非常强的“下一件装备/下一波视野/这波团来不来”。

---

## 6. 核心模型套件（像 ProAct 那样写：Value + Policy + Planner）

### 6.1 Value Model：胜率/优势变化预测器（全系统的“评估器”）

**Why**：没有 value，proactive 系统就只能“模仿”，没法做“反事实选择”。

**训练目标（建议多头）：**

* `P(win | state_t)`（最终胜负）
* `Δgold_adv_{t→t+3min}`、`Δxp_adv_{t→t+3min}`（短期优势变化）
* `fight_outcome`（如果 t 位于 teamfight start 前 window 内）

**输入**：state_t
**输出**：value_t / multi-head

**用途：**

* 给候选计划打分（plan selection）
* 做“建议的预期收益”（explainability）
* 做离线策略改进指标（estimated uplift）

### 6.2 Policy Models（可独立上线的 3 个“动作专家”）

#### (1) Next Item Policy（装备规划）

* **监督信号**：purchase_log 的下一次购买（或下一件大件的完成）
* **输出**：Top-K 候选物品 + 概率
* **增强**：做成“序列规划”——预测未来 2–4 个购买（带分支）

离线评测：

* Top-1/Top-5 accuracy（基础）
* 更重要：用 Value Model 评估 “推荐物品 vs 真实物品” 的 value 差（counterfactual uplift）

#### (2) Ward Placement Policy（视野规划）

* **监督信号**：obs_log/sen_log 的 (time, x, y)
* **离散化**：把 (x,y) bin 到网格（例如 32×32 或 64×64）
* **输出**：Top-K 网格点 + 置信度 + 推荐类型（obs/sen）

离线评测（比 accuracy 更贴近真实收益）：

* Ward survival time 预测/提升（用 left_log 计算真实存活）
* 覆盖热区匹配度（与高水平分布 KL / Earth Mover’s Distance）
* “视野→事件”相关性：插眼后 N 秒内己方击杀/被杀变化（做统计评估，不当因果结论）

#### (3) Fight Participation Policy（参战决策）

* **监督信号**：teamfights 中该玩家在窗口内是否有 damage/ability_uses/item_uses（定义参与）
* **输出**：JOIN / SKIP + 理由特征（如经济差/关键技能/TP 是否可用等）

离线评测：

* 与高水平策略一致性（rank-conditioned）
* JOIN/STOP 的价值差（value uplift）

### 6.3 Proactive Planner（系统灵魂）

这是“像 ProAct 一样”的部分：不是只预测下一步，而是预测**未来计划**并提前执行。

**Planner 的基本接口：**

* 输入：state_t
* 输出：一个 plan = `[a_t, a_{t+1}, …, a_{t+H}]`（H 步宏观动作）+ 每步预计收益/风险

**实现路线（从稳到野）：**

1. **Plan-as-Sequence（纯生成）**
   用 Transformer/Decision-Transformer 类结构直接生成动作序列（宏观 action tokens）
2. **Generate-and-Score（候选生成 + 价值评估）**

   * policy 生成 N 个候选 plan
   * value model 对每个 plan 的预期结果打分（可用 learned dynamics 近似 rollout，或用启发式对齐未来状态）
3. **RISE-style 自举循环（不需要多模态）**

   * 生成计划 → 用 value/规则打分 → 选最好 → 作为伪标签回灌训练（self-improve）
   * 你想要的“visionary”就在这：**离线自举迭代**，不是修修补补

---

## 7. Proactive 评测：怎么证明它“更主动”？

只做 accuracy 不够。我们要一组**能写进论文/能做迭代门槛**的指标。

### 7.1 提前量（Anticipation Lead Time）

对每类事件（推塔/团战/关键装备完成）定义：

* 事件发生时间 `t_event`
* 模型第一次高置信预测该事件/对应动作的时间 `t_pred`
* **LeadTime = t_event - t_pred**（越大越 proactive）

例子：

* “下一波关键团战（teamfight start）”是否提前 60s 就提示“该集合/该做视野”
* “对面关键大件（如 blink/bkb）”完成前，你是否提前调整插眼/抱团

### 7.2 计划一致性（Plan Fidelity）

在真实比赛中，从 t 开始的未来 H 步动作序列 `A_real`，模型计划 `A_plan`：

* 序列相似度（edit distance / token F1）
* 更关键：**高价值 token 的命中**（比如关键大件、关键眼位、关键团战 join）

### 7.3 价值增益（Estimated Uplift）

用 value model 评估：

* `V(state_t, plan_model) - V(state_t, plan_real)`
  作为离线迭代的核心指标（注意这是代理，但足够作为研发闭环）

---

## 8. 产品体验（MVP 交付长什么样）

### 8.1 复盘页面：Timeline + 建议卡片

* 左侧：时间线（objectives、teamfights、优势曲线）
* 右侧：每个关键时间点 t 的 “三卡”

  1. **装备卡**：推荐下一件/两件 + 分支 + 预期收益（value delta）
  2. **视野卡**：推荐 1–3 个眼位（网格点/坐标）+ 预期存活 + 覆盖目的
  3. **团战卡**：这波你该来吗？如果来：你在 fight 内的“技能/物品使用对比”（结构化对比）

### 8.2 “反事实”按钮（你想要的核心）

* “如果 8:30 没做这件装备而是做 X，会怎样？”
* “如果 7:10 这颗眼换到这里，会怎样？”
  输出：
* value 变化（代理）
* 相似局检索（从数据库里找相似 state 的高水平轨迹作为证据）

---

## 9. 研发里程碑（不废话，直接可执行拆解）

### Phase 0（数据与基线，1–2 周）

* 抓取 + 原始 JSON 存储
* ETL：生成 `events` / `timeseries_minute` / `teamfights` 表
* 基线：

  * 物品：按英雄/时间段统计最常见 build
  * 眼位：按英雄/时间段/优势统计热力图
  * 团战：简单规则（经济差/等级差）做 join baseline

交付：

* 数据覆盖率报告（哪些字段缺失、哪些 match 要过滤）
* 3 个基线模型 + 离线评测脚本

### Phase 1（Value Model + 两个专家策略，2–4 周）

* 训练 value model（win + Δadv）
* 训练 next-item policy
* 训练 ward policy（obs/sen）
* 建立 counterfactual 评测（value uplift）

交付：

* “装备建议”与“视野建议”可在复盘页面跑通
* 指标门槛：相对基线 uplift、lead time（先定义后追）

### Phase 2（Proactive Planner，4–8 周）

* 引入 plan tokens（多步动作序列）
* Generate-and-Score：候选计划生成 + value 评估选优
* RISE-style 自举：用选优计划回灌训练（迭代 2–3 轮）
* 引入 fight participation policy 并纳入 plan

交付：

* “未来 90 秒计划”模块
* proactive 指标（LeadTime、PlanFidelity、Uplift）达标

---

## 10. 风险与工程注意（不“修修补补”，但得提前钉住）

### 10.1 Patch 漂移

* 你有 `patch` 字段：**必须做 patch-conditioning 或按 patch 分桶训练/评测**
* 否则装备/节奏的分布漂移会直接把模型打废

### 10.2 因果 vs 相关

* 我们的“uplift”来自 value proxy，不声称真实因果
* 但作为研发闭环完全够用；后续再引入更严谨的 off-policy evaluation 也行

### 10.3 日志缺失与噪声

* leaver、短局、非正常模式要过滤（你数据里有 `leaver_status`, `duration`, `lobby_type`）
* ward 的 left_log 可能缺失/异常：评测要允许“unknown”

---

# 附录 A：你这份 schema 各字段的“可用性评级”（快速决策版）

**S 级（立刻用，且能做 proactive）**

* `players[].purchase_log`, `purchase_time`, `first_purchase_time`
* `players[].obs_log/sen_log` + `obs_left_log/sen_left_log`
* `teamfights[]`（start/end + per-player）
* `objectives[]`
* `radiant_gold_adv`, `radiant_xp_adv`
* `players[].ability_upgrades_arr`
* `players[].gold_t/xp_t/lh_t/dn_t/times`

**A 级（强辅助/做条件变量）**

* `players[].lane`, `lane_role`, `is_roaming`
* `rank_tier`, `computed_mmr`（做 skill-conditioned policy）
* `actions`, `actions_per_min`（风格/强度指纹）
* `lane_pos`（长期站位/转线倾向）

**B 级（先放着，未来再说）**

* `chat`（除非你要做行为学/情绪）
* `cosmetics`
* `draft_timings`（你样例为空，得看覆盖率）

---

# 附录 B：我建议你们“第一刀”就砍在哪（最值）

如果你要最快做出“看起来像未来产品”的东西：

1. **Value Model（胜率/优势变化）**：没有它就没有反事实
2. **Next Item（序列）**：最稳、最强可解释
3. **Ward Policy（坐标网格）**：你这份数据独一档的好（还有 left_log）
4. 用 (2)(3) 生成候选动作 → (1) 打分 → 形成 **Generate-and-Score** 的 proactive demo

---

如果你愿意，我下一步可以直接把 **events 表的事件类型定义**、**状态特征列表（字段级）**、以及 **3 个模型的训练样本构造伪代码**也写出来（按你们工程实现习惯：Python/SQL/Parquet）。你不需要再补充数据，按你给的 schema 就能开干。
