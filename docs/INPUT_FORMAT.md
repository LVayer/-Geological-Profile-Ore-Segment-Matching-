# 输入、标注与输出契约

## 图片 manifest

路径必须位于项目内。顺序就是空间顺序，不自动重排。坐标 x 向右、y 向下，`bounds` 是 ROI 在共同物理坐标系的 `[xmin,ymin,xmax,ymax]`，不是原图片像素范围；`roi`/`swatch` 是 `[x,y,width,height]` 像素坐标。

```json
{
  "sections": [
    {
      "id": "S01", "image": "输入数据/真实项目/S01.png",
      "frame": "project_depth_datum", "units": "m", "station": 0,
      "bounds": [0, 0, 500, 200], "roi": [50, 40, 1500, 600],
      "registration_verified": true, "complete": false,
      "legend": [
        {"lithology": "砂岩", "swatch": [1600, 50, 25, 15], "verified": true},
        {"lithology": "泥岩", "rgb": [140, 180, 205], "verified": true}
      ],
      "exclude_polygons": [[[10,10],[30,10],[30,30],[10,30]]],
      "fault_polygons": [],
      "lab_tolerance": 12, "lab_margin": 4,
      "boundary_samples": 64, "min_area_pixels": 60,
      "contact_gap_pixels": 3, "minimum_complete_coverage": 0.85
    }
  ]
}
```

示例仅展示一张的结构，匹配需要至少两张，下一张 station 更大。`complete=true` 必须由数据提供者确认覆盖范围和识别完整性；图像路径下还会检查最小覆盖率。不得为了获得尖灭结果而随意设置 true。颜色岩性必须使用一致标准字符串，系统不会把“中砂岩”和“砂岩”自动视为相同。

目前输入是共享二维物理坐标下的对比剖面，未定义世界坐标原点和剖面走向的地图投影转换。不支持用相同 frame 名称掩盖未配准数据。

## 人工实例掩膜

适合纹理图、同色接触层和专家确认的边界。灰度整数 PNG/TIFF 的 0 为背景，正整数表示独立实例；mask 尺寸本身就是绘图区，bounds 描述完整 mask 范围。

```json
{
  "sections": [
    {
      "id": "S01", "frame": "project_depth_datum", "units": "m", "station": 0,
      "bounds": [0, 0, 500, 200], "instance_mask": "输入数据/真实项目/S01_instances.png",
      "registration_verified": true, "complete": true,
      "instances": [
        {"value": 1, "id": "S01_L1", "lithology": "砂岩", "verified": true},
        {"value": 2, "id": "S01_L2", "lithology": "泥岩", "verified": true}
      ]
    }
  ]
}
```

每个实例可包含不止一个区域，但列内多段区域会触发几何不确定性。不声明的 mask ID、重复 ID、空实例会报错。原始 mask 应作为正式标注保存，轮廓只是派生数据。

## 完整图例预处理

图片导入后会在地层识别前生成 `legend_catalog`。它与旧的 `legend` 色块数组分工不同：`legend` 只保存可用于岩性颜色分类的色块；`legend_catalog.items` 保存完整图例单元，包括地层色块、地层/断层界限和其他符号。其他符号继续分为 `label`、`line`、`label_line`。

每项记录 `box`、OCR 名称建议、符号内部文字、颜色分布、主色、线条颜色、像素线宽、方向、直线/折线/曲线/虚线形态及复核标记。OCR 名称不会自动获得人工确认状态；颜色和几何特征来自原始像素。后续主图处理应优先使用 `geological_boundary` 原型保护地质界面，并把 `other.line` 与 `other.label_line` 作为杂项证据。

## 结构化剖面

`recognize` 输出 `sections.json`；可由人工编辑后使用 `match`。完整可运行例子在 `输出数据/image_pipeline/sections.json`。每层的字段解释见 README；ID 在剖面内唯一，layers 按唯一 order 升序。上/下/接触邻居引用同一剖面的 layer ID。边界点必须为有限数字且不少于 8 个；默认 64 个。读取时不会自行填造缺失几何。

## 真值案例

```json
{
  "name": "project_case_001", "seed": 0,
  "survey_group": "independent_site_01", "exhaustive_annotation": true,
  "source": {"...": "完整 S01 剖面对象"},
  "target": {"...": "完整 S02 剖面对象"},
  "truth": [
    {"sources": ["S01_L1"], "targets": ["S02_L1", "S02_L2"], "relation": "split"},
    {"sources": ["S01_L2", "S01_L3"], "targets": ["S02_L3"], "relation": "merge"},
    {"sources": ["S01_L4"], "targets": [], "relation": "pinch-out"}
  ],
  "must_abstain": ["S01_L5"]
}
```

示例中的 `source`/`target` 省略号是文档占位符，运行时必须替换成完整对象。标签不作为算法输入，只进入独立评测。merge 必须合并为一个真值事件，不要用两条重复目标的 continuous 表示；输出也以 source_layers/target_layers 数组编码，不丢失完整事件成员。

不对应的地层不写正对应边；确实无法从证据决定的源地层加入 must_abstain。未完成标注不得声明 exhaustive_annotation，否则训练会错误地把未标注候选当负例。

训练分区 JSON：

```json
{"train":["输入数据/真实项目/train_case.json"],"validation":["输入数据/真实项目/validation_case.json"],"test":["输入数据/真实项目/test_case.json"]}
```

训练程序仅使用 train；validation/test 用于隔离检查，用户后续应分别评测。benchmark 使用另一个索引 `{"cases":["data/real/test_case.json"]}`，执行：

```powershell
.\.venv\Scripts\python.exe -B -m stratamatch benchmark --input ../输入数据/真实项目/test_index.json
```

注意 benchmark 默认写 outputs/benchmark；进行真实评测前请在本目录内另存已有输出。真实 GNN 模型路径需要在 config 中设置，不会默默覆盖合成模型。

## 输出与复核

- `matches.json`：各算法完整结果、候选特征、拒绝原因、原始建议、最终状态、替代解间隔、Markov 边际、模型来源以及冲突。
- `matches.csv`：一行一个完整事件，sources/targets 数组以 JSON 单元格保留。不会把 merge 拆成不可辨识的几条独立边。
- `review_manifest.json`：软件内审查入口，记录两侧着色图、算法结果和剖面数据的位置。
- `review_source.png` / `review_方法_target.png`：重新着色后的基准剖面与目标剖面。
- `manual_review.json`：人工确认、否决、待复核及人工修正关系；与算法原始结果分开保存。
- `modeling_input_方法.json`：只包含人工确认关系及完整剖面边界的三维建模输入。
- `modeling_input_schema.json`：建模输入字段和安全导出规则。
- `review.html`：备用离线检查页面；`matching_graph.png` 为静态图。
- `recognition/*_labels.npy`：-1 为未分配像素，非负整数查 `recognition.label_id_mapping` 对应地层 ID；排序不会改变此映射。PNG 预览只为显示，不是岩性颜色规范。
- 自动布局同时保存 `roi`、`plot_content_rings` 和 `auto_layout.legend_search_scope`。ROI 是坐标矩形；内容环带有 `hole` 标志并限定实际可分层像素；图例搜索范围为整张图，不能再用 ROI 整体排除图例。
- `relation_type` 是保守最终结论；`proposed_relation` 是供审查的算法建议。不能只读 proposed_relation 并忽略 status。
- `accepted` 才是当前实验规则接受；`uncertain`/`manual_review` 不得自动进入三维建模。`rejected` 的 unrelated 记录在候选拒绝审计里，表示该候选被排除，不表示整层一定消失。
- UI 审查写入独立文件，不改写 `matches.json`。三维建模程序应读取 `modeling_input_方法.json`，不能直接把 `proposed_relation` 当成已确认关系。
