# 跨模型内在信号适配器:统一 npz schema

> 上游:[plan §0.3/§4](../../../doc/plan/20260719_ar_model_extension_plan.md)。每个新 AR 模型写一个
> `<model>.py` 适配器,单遍 teacher-forcing 前向同时产出 loss+entropy,落成本 schema 的 npz。
> MusicGen 系不走适配器(沿用 phase7_release 的 extract_loss/entropy_curves.py 两遍流程)。

## 文件布局

```
features/<model_tag>/<benchmark>/<split>/<uid>.npz     # uid = 与既有管线一致的 clip 稳定 id
features/<model_tag>/<benchmark>/manifest_<split>.csv  # audio_path, npz_path, mean_loss, mean_entropy, T_native
```

## npz 内容

| key | shape / dtype | 说明 |
|---|---|---|
| `loss` | `(K, T) float32` | 逐 token NLL;K=码本数(单码本模型 K=1) |
| `entropy` | `(K, T) float32` | 逐 token 预测熵(nats) |
| `mask` | `(T,) uint8` | 1=有效帧;padding/超窗为 0 |
| `meta` | `() object`(json str) | 见下 |

`meta` 必填字段:`model_tag`(如 `yue_s1`)、`model_hf_id`、`codec`、`frame_hz`(每秒 token 帧数,交错多轨记等效音频帧率)、`K`、`clip_id`、`benchmark`、`window`(该库窗口化参数,照抄 plan §0.1b)、`context_limit_tokens`(§1(b) cliff 实测值)、`extracted_at`。

## 约定

0. **原生 pattern 纪律(v2.1,最高优先)**:teacher-forcing 必须按该模型**训练时的 token 排布**(delay/交错/双轨等)构造输入,logits 再对齐回音频帧;禁止"对齐 codes 直接前向"的并行捷径。MusicGen 参考实现:`phase7_release/lib/repro/musicgen_pattern.py`。依据:并行口径系统性抬高 loss 且随规模放大(findings doc §2c)。
1. **T 为模型原生帧数**,不在适配器里池化;训练侧统一 `pool_to_frames(x, 1500)`(均匀 mean-pool,同论文)。
2. 多轨/交错模型(YuE 双轨、SongGen interleave):按**音频时间轴对齐**后堆进 K 维(轨×码本拉平),`meta` 里写清各行含义;不得把不同时间步混进同一行。
3. 分块模型:同论文——按该模型 context 上限切非重叠块,块内 teacher-forcing,特征沿时间拼接后再落盘(mask 标接缝无效帧,若有)。
4. 数值:loss/entropy 一律自然对数;inf/nan 置 0 并把 mask 置 0。
5. 幂等:npz 已存在且 `validate()` 通过则跳过(--skip-existing 语义)。
