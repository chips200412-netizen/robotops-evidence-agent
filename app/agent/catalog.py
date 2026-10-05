"""设备与故障目录（阶段 3 切片 2）。

内置型号表（01 §10 虚构/公开名称）与 10 类故障分类（01 §4 功能 2），
同时提供规则兜底：LLM 不可用/返回异常时用关键词匹配识别型号与分类
（03 §7 工具/LLM 兜底，保证工作流不中断）。
"""
import re

# 设备型号 → 匹配关键词（含全称与常见别名，小写比对）
DEVICE_MODELS: dict[str, list[str]] = {
    "EduBot A1": ["edubot a1", "edubot-a1", "a1 机器人", "edubot"],
    "EduBot A2 Pro": ["edubot a2 pro", "edubot-a2-pro", "edubot a2", "a2 pro"],
    "EduCar X1": ["educar x1", "educar-x1", "educar", "x1 小车"],
    "Humanoid H1": ["humanoid h1", "humanoid-h1", "humanoid", "h1 人形"],
    "夸父 4 Pro": ["夸父 4 pro", "夸父4pro", "夸父4", "夸父"],
    "Tonybot": ["tonybot", "tony bot", "tonybot 2025"],
}

# 10 类故障分类 → 匹配关键词（A3-4 分类依据）
FAULT_CATEGORIES: dict[str, list[str]] = {
    "网络连接": ["网络", "网线", "wifi", "wi-fi", "无线", "连接不上", "连不上", "断网", "配网", "指示灯闪烁"],
    "运动控制": ["轮子", "电机", "运动", "移动", "行走", "转向", "不动", "卡顿", "失控", "跑偏"],
    "语音交互": ["语音", "说话", "对话", "唤醒", "麦克风", "听不到", "音箱", "播放"],
    "摄像头": ["摄像头", "相机", "画面", "黑屏", "拍照", "图像", "模糊", "识别不到人"],
    "传感器": ["传感器", "超声波", "避障", "红外", "测距", "触觉", "陀螺仪", "失灵"],
    "电池与充电": ["电池", "充电", "电量", "续航", "充不进", "鼓包", "发热"],
    "程序运行": ["程序", "代码", "运行报错", "崩溃", "卡死", "死机", "重启", "无法启动", "报错"],
    "机械结构": ["螺丝", "关节", "机械", "断裂", "松动", "卡住", "齿轮", "外壳"],
    "课程操作": ["课程", "课件", "演示", "实验", "编程课", "上课", "教学"],
    "其他": [],
}

# 风险等级（A3-5）：高 / 中 / 低 关键词表（rule_risk_level 判定，check_solution_risk 工具复用）
RISK_HIGH_KEYWORDS = [
    "220", "高压", "电击", "漏电", "电池鼓包", "冒烟", "起火", "燃烧", "短路",
    "拆机", "失控", "断裂", "夹伤", "高温", "烧焦",
]
RISK_MEDIUM_KEYWORDS = [
    "更换", "拆卸外壳", "螺丝", "工具", "固件", "升级", "刷机", "数据丢失", "反复重启",
]

# 原始输入危险预检（R1-ACC-001）：在首次 LLM 调用前识别不可等待追问的安全信号。
# 直接词覆盖明确危险现象；组合规则允许“电池外壳已经明显鼓起”这类非连续表达。
RAW_INPUT_HAZARD_TERMS: dict[str, tuple[str, ...]] = {
    "smoke": ("冒烟", "烟雾", "浓烟"),
    "burning_odor": ("焦味", "焦糊味", "烧焦味", "糊味"),
    "exposed_wire": ("裸露电线", "电线裸露", "裸露线缆", "线缆裸露", "电线露铜", "线缆露铜"),
    "abnormal_heat": ("异常发热", "异常高温", "过热", "烫手"),
    "fire_or_electric": ("起火", "着火", "燃烧", "火花", "漏电", "触电"),
}

RAW_INPUT_HAZARD_COMBINATIONS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "battery_swelling": (("电池", "电芯", "锂电"), ("鼓包", "鼓起", "膨胀", "变形")),
    "exposed_wire_damage": (
        ("电线", "线缆", "导线"),
        ("裸露", "露铜", "破皮", "外皮破损", "破损"),
    ),
}

_NEGATION_PREFIX_RE = re.compile(
    r"(?:(?:看|闻|嗅|查|检查|检测|观察|发现)(?:不|未)(?:到|见|出)|"
    r"(?:并没有|没有|并未|未|并无|无|不存在|不是|并非|已排除|排除|没|不)"
    r"(?:(?:出现|发生|发现|检测到|观察到|看到|闻到|见到|见|任何|明显|相关|"
    r"这种|上述|持续|检查到|继续|曾|过|有|了|再|的|是))*)$"
)
_POST_NEGATION_TOKEN_RE = re.compile(
    r"(?:均|都|全部|全|一概|仍然|依然|仍|至今|目前|暂时|暂|现)*"
    r"(?:(?:(?:已经|已)?(?:确认)?(?:明确(?:地)?)?(?:被)?(?:排除|否认)(?:了)?)|"
    r"(?:(?:尚未|未|没有)(?:(?:被)|(?:在(?:本次|此次|当前)?(?:巡检|检查|检测|排查)中))*"
    r"(?:发现|检测到|发生|出现))|没有|不存在)"
)
_CAUSE_DETAIL_RE = re.compile(
    r"^(?:(?:的)?(?:现象|问题|情况|迹象))?"
    r"(?:(?:背后)?(?:的)?)?(?:具体|根本|实际|真正)?(?:原因|来源|根源|成因)"
)
_NEGATION_SCOPE_BOUNDARIES = (
    "。", "；", ";", "！", "!", "？", "?",
    "但是", "不过", "然而", "只是", "却", "随后", "现在", "但",
)
_TEMPORAL_HAZARD_BOUNDARIES = ("随后", "后来", "然后", "之后", "此后")
_COMBINATION_SCOPE_BOUNDARIES = (
    "，", ",", "。", "；", ";", "！", "!", "？", "?",
    "但是", "不过", "然而", "但",
    *_TEMPORAL_HAZARD_BOUNDARIES,
)
_LIST_CONNECTOR_RE = re.compile(r"(?:、|，|,|以及|或者|并且|与|和|或|及|也|都)+")
_NEGATED_LIST_CONTINUATION_RE = re.compile(
    r"^(?:(?:、|，|,|以及|或者|并且|与|和|或|及|也|都)|"
    r"(?:(?:的)?(?:现象|情况|迹象|风险|问题))?(?:$|[，,。；;！!？?]))"
)
_HAZARD_CLAUSE_SPLIT_RE = re.compile(
    "|".join(
        re.escape(token)
        for token in sorted(_COMBINATION_SCOPE_BOUNDARIES, key=len, reverse=True)
    )
)
_RAW_INPUT_HAZARD_WORDS = tuple(
    sorted(
        {
            word
            for keywords in RAW_INPUT_HAZARD_TERMS.values()
            for word in keywords
        }
        | {
            word
            for groups in RAW_INPUT_HAZARD_COMBINATIONS.values()
            for group in groups
            for word in group
        },
        key=len,
        reverse=True,
    )
)
_POST_NEGATION_CONTEXT_RE = re.compile(
    r"^(?:(?:但|但是|不过|然而))?"
    r"(?:(?:该|此|这种|这个|上述|相关|此类)?(?:的)?"
    r"(?:情况|现象|风险|可能性?|问题|迹象|疑虑|担忧|嫌疑))?"
    r"(?:(?:(?:在)?(?:本次|此次|当前)?(?:巡检|检查|检测|排查)(?:中)?)|"
    r"(?:(?:经|经过)?(?:检查|检测|确认|排查)(?:后)?(?:确认)?))?$"
)
_POST_NEGATION_REFERENCE_RE = re.compile(
    r"^(?:(?:该|此|这种|这个|上述|相关|此类)?(?:的)?"
    r"(?:情况|现象|风险|可能性?|问题|迹象|疑虑|担忧|嫌疑))(?:了)?$"
)
_POST_NEGATION_CLAUSE_END_RE = re.compile(r"[，,。；;！!？?]")
_POST_NEGATION_CONTRAST_RE = re.compile(r"(?:但是|不过|然而|但)")
_STRONG_LIST_CONNECTOR_RE = re.compile(r"(?:、|以及|或者|与|和|或|及)")
_SHARED_POST_QUANTIFIER_RE = re.compile(r"^(?:均|都|全部|全|一概)")
_CROSS_CLAUSE_ANAPHORA_RE = re.compile(
    r"(?:该|此|这种|这个|上述|相关|此类|这一|这类)(?:的)?"
    r"(?:情况|现象|风险|可能性?|问题|迹象|疑虑|担忧|嫌疑)"
)
_UNRESOLVED_EXCLUSION_RE = re.compile(
    r"(?:(?:不|未|尚未|并未|没有|无法|不能|难以)(?:能|能够)?(?:完全|彻底)?"
    r"(?:(?:明确|确认)(?:地)?)?(?:排除|否认)|"
    r"(?:准备|计划|拟|尝试|试图|待|尚待|需要|将要|即将|正在|正)"
    r".{0,8}?(?:排除|否认)|"
    r"(?:是否)?(?:已经|已)?(?:排除|否认).*?(?:尚不确定|不确定|未知|待确认)|"
    r"(?:没有|无|未有|未见).{0,8}?(?:证据|依据).{0,16}?(?:排除|否认)|"
    r"(?:现有|当前|目前)?(?:的)?(?:证据|依据).{0,8}?(?:尚|仍)?"
    r"(?:不足|不充分|不够)(?:以|来)?.{0,8}?(?:排除|否认)|"
    r"(?:据说|传闻|听说|声称|或许|也许|可能(?!性)|报告.{0,4}?(?:称|写)|"
    r"系统.{0,4}?(?:提示|显示)).{0,16}?(?:排除|否认)|"
    r"(?:若|如果|假如|一旦).{0,20}?(?:排除|否认)|"
    r"(?:未能|不能|无法|无人|没人|尚未).{0,12}?(?:确认|证明|证实|说明|核实)"
    r".{0,16}?(?:排除|否认)|"
    r"(?:不是|并非|不能说|无法说).{0,16}?(?:排除|否认)|"
    r"(?:误写|误报|错误(?:地)?(?:记录|写成)?).{0,16}?(?:排除|否认)|"
    r"(?:排除|否认).{0,20}?(?:尚未完成|未完成|进行中|是否确定|仍需确认|"
    r"尚未核实|之外|以外))"
)
_CROSS_CLAUSE_CAUSE_REFERENCE_RE = re.compile(
    r"(?:该|此|这种|这个|上述|相关|此类|这一|这类)?(?:的)?"
    r"(?:情况|现象|风险|可能性?|问题|迹象|疑虑|担忧|嫌疑)?(?:的)?"
    r"(?:具体|根本|实际|真正)?(?:原因|来源|根源|成因)"
)
_CROSS_CLAUSE_REFERENCE_BRIDGE_RE = re.compile(
    r"(?:(?:随后|后来|然后|目前|现在|最终|经|经过|已|已经|也|并|再|"
    r"技术员|维修人员|现场人员|我们|对此|对其|进一步|完成|进行|开展|"
    r"实施|开始|做|做了|了|一次|检查|检测|确认|排查|核实|复核|复查|"
    r"后|结果|显示))*"
)
_REASSERTION_CONTEXT_PATTERN = (
    r"(?:现在|如今|今天|目前|随后|后来|然后|之后|此后|刚刚|已经|再次|重新|又|"
    r"开始|仍然|依然|仍|继续|现场仍然?|当天(?:晚些时候|稍后)|今早|今晨|"
    r"第二天|次日|第(?:二|三|四|五|六|七|八|九|十|\d+)天|"
    r"(?:(?:充电|运行|启动|开机|使用|放置|等待))?"
    r"(?:半|一|二|两|三|四|五|六|七|八|九|十|几|数|\d+)"
    r"(?:秒钟?|分钟?|小时|天|周|个月)(?:之)?后|"
    r"过了?(?:(?:半|一|二|两|三|四|五|六|七|八|九|十|几|数|\d+)"
    r"(?:秒钟?|分钟?|小时|天|周|个月)|"
    r"一会儿|片刻|不久)|"
    r"(?:充电|运行|启动|开机|使用|放置|等待)(?:之)?后)"
)
_REASSERTION_CONTEXT_RE = re.compile(_REASSERTION_CONTEXT_PATTERN)
_REASSERTION_FILLER_RE = re.compile(
    _REASSERTION_CONTEXT_PATTERN
    + r"|(?:正在|突然|明显|出现|迹象|现象|情况|起来|来|已|正|就|却|了)"
)
_COMBINATION_LINK_RE = re.compile(
    r"(?:(?:的|外壳|表面|本体|包体|壳体|电池仓|接头|接口|绝缘层|外皮|部分|局部|处|位置|"
    r"充电(?:时|后|过程中)|运行(?:时|后|过程中)|使用(?:时|后|过程中)|"
    r"启动后|开机后|放置后|等待后|"
    r"已经|已|现在|目前|随后|后来|刚刚|开始|出现|发生|有|变得|逐渐|明显|严重|轻微|疑似|"
    r"持续|再次|重新|又|突然|正|正在|仍|依然|也|均|都|却|了|起来|来))*"
)
_TEMPORAL_OPERATION_RE = re.compile(
    r"(?:(?:充电|运行|使用)(?:时|后|过程中)|(?:启动|开机|放置|等待)后)"
)
_BATTERY_INTERVENING_COMPONENT_RE = re.compile(
    r"(?:外壳|表面|本体|包体|壳体|电池仓)"
)
_WIRE_EXTERNAL_COMPONENT_RE = re.compile(
    r"(?:外壳|本体|包体|壳体|电池仓)"
)
_INTERVENING_COMPONENT_RES = {
    "battery_swelling": _BATTERY_INTERVENING_COMPONENT_RE,
    "exposed_wire_damage": _WIRE_EXTERNAL_COMPONENT_RE,
}
_EXCLUSION_TARGET_BRIDGE_RE = re.compile(
    r"(?:经|经过)?(?:检查|检测|确认|排查|核实|复核|复查)(?:后)?"
)
_EXCLUSION_TARGET_CONTRAST_RE = re.compile(r"(?:而非|不是|并非)")
_COMBINATION_TARGET_FILLERS = (
    *_TEMPORAL_HAZARD_BOUNDARIES,
    "外壳", "表面", "本体", "包体", "壳体", "电池仓", "接头", "接口",
    "绝缘层", "外皮", "部分", "局部", "位置", "该处", "这里", "那里",
    "已经", "明显", "严重", "轻微", "疑似", "发生", "出现", "开始",
    "情况", "现象", "风险", "可能性", "问题", "迹象", "疑虑", "担忧",
    "嫌疑", "它", "其", "且", "的", "已", "被", "了",
)
_EXCLUSION_POLARITY_BOUNDARIES = (
    "。", "；", ";", "！", "!", "？", "?", "，", ",",
    "但是", "不过", "然而", "随后", "后来", "然后", "最终", "但",
)


def _post_negation_tail_refers_to_keyword(tail: str, keyword: str) -> bool:
    """后置排除的显式宾语必须为空、回指当前现象或重复当前关键词。"""
    clause_tail = _POST_NEGATION_CLAUSE_END_RE.split(tail, maxsplit=1)[0]
    if not clause_tail or _POST_NEGATION_REFERENCE_RE.fullmatch(clause_tail):
        return True
    return re.fullmatch(
        rf"(?:(?:该|此|这种|这个|上述|相关|此类)?)"
        rf"{re.escape(keyword)}"
        r"(?:(?:的)?(?:情况|现象|风险|可能性?|问题|迹象|疑虑|担忧))?(?:了)?",
        clause_tail,
    ) is not None


def _post_negation_prefix_refers_to_current(prefix: str) -> bool:
    """后置排除前只允许当前现象回指、检查桥接词和有限转折词。"""
    if _POST_NEGATION_CONTEXT_RE.fullmatch(prefix):
        return True
    candidate = _POST_NEGATION_CONTRAST_RE.sub("", prefix, count=1)
    candidate = _CROSS_CLAUSE_ANAPHORA_RE.sub("", candidate, count=1)
    return _CROSS_CLAUSE_REFERENCE_BRIDGE_RE.fullmatch(candidate) is not None


def _has_shared_post_negation(
    suffix: str,
    keyword: str,
    preserve_ambiguous_generic: bool = False,
) -> bool:
    """识别单个危险词或并列危险清单末尾共享的否定/排除结论。"""
    unresolved_spans = [
        (match.start(), match.end())
        for match in _UNRESOLVED_EXCLUSION_RE.finditer(suffix)
    ]
    for match in _POST_NEGATION_TOKEN_RE.finditer(suffix):
        if any(start <= match.start() < end for start, end in unresolved_spans):
            continue
        if not _post_negation_tail_refers_to_keyword(
            suffix[match.end():],
            keyword,
        ):
            continue
        prefix = suffix[:match.start()]
        referenced_hazards = [
            word for word in _RAW_INPUT_HAZARD_WORDS if word in prefix
        ]
        shared_scope = bool(referenced_hazards) and bool(
            _STRONG_LIST_CONNECTOR_RE.search(prefix)
            or _SHARED_POST_QUANTIFIER_RE.search(match.group())
        )
        tail_clause = _POST_NEGATION_CLAUSE_END_RE.split(
            suffix[match.end():],
            maxsplit=1,
        )[0]
        explicit_current_tail = keyword in tail_clause
        # 已完成的排除若在宾语中直接重复当前危险词，目标已经消歧；即使前文
        # 同时提到其他危险，也只能撤销被明确点名的这一项。
        if explicit_current_tail:
            return True
        generic_reference = bool(
            _CROSS_CLAUSE_ANAPHORA_RE.search(prefix)
            or _POST_NEGATION_REFERENCE_RE.fullmatch(tail_clause)
        )
        implicit_reference = not referenced_hazards and not explicit_current_tail
        reference_bridge = _CROSS_CLAUSE_ANAPHORA_RE.sub("", prefix, count=1)
        reference_bridge = _POST_NEGATION_CONTRAST_RE.sub(
            "",
            reference_bridge,
            count=1,
        )
        cross_clause_reference = any(
            token in prefix for token in _COMBINATION_SCOPE_BOUNDARIES
        ) or bool(
            reference_bridge
            and _CROSS_CLAUSE_REFERENCE_BRIDGE_RE.fullmatch(reference_bridge)
        )
        if (
            preserve_ambiguous_generic
            and not shared_scope
            and not _SHARED_POST_QUANTIFIER_RE.search(match.group())
            and (generic_reference or implicit_reference)
            and cross_clause_reference
        ):
            continue
        if referenced_hazards:
            if _POST_NEGATION_CONTRAST_RE.search(prefix):
                continue
            if not (
                _STRONG_LIST_CONNECTOR_RE.search(prefix)
                or _SHARED_POST_QUANTIFIER_RE.search(match.group())
            ):
                continue
        for word in _RAW_INPUT_HAZARD_WORDS:
            prefix = prefix.replace(word, "")
        prefix = _LIST_CONNECTOR_RE.sub("", prefix)
        if _post_negation_prefix_refers_to_current(prefix):
            return True
    return False


def _is_negated_hazard(
    text: str,
    start: int,
    end: int,
    preserve_ambiguous_generic: bool = False,
) -> bool:
    """判断一次关键词命中是否处在局部否定/排除语境中。"""
    keyword = text[start:end]
    if _has_shared_post_negation(
        text[end:],
        keyword,
        preserve_ambiguous_generic,
    ):
        return True

    boundary_candidates = []
    for token in _NEGATION_SCOPE_BOUNDARIES:
        position = text.rfind(token, 0, start)
        if position >= 0:
            boundary_candidates.append(position + len(token))
    boundary = max(boundary_candidates, default=0)
    prefix = text[boundary:start][-48:]
    if _NEGATION_PREFIX_RE.search(prefix) and not (
        _UNRESOLVED_EXCLUSION_RE.search(prefix)
    ):
        return _CAUSE_DETAIL_RE.search(text[end:]) is None

    # “没有冒烟、焦味或异常发热”只在第一个词前出现一次否定词；移除已出现的
    # 危险词与列表连接词后再次判断，使否定作用域覆盖同一并列清单。
    reduced = prefix
    for word in _RAW_INPUT_HAZARD_WORDS:
        reduced = reduced.replace(word, "")
    reduced = _LIST_CONNECTOR_RE.sub("", reduced)
    return (
        _NEGATION_PREFIX_RE.search(reduced) is not None
        and _UNRESOLVED_EXCLUSION_RE.search(reduced) is None
        and _NEGATED_LIST_CONTINUATION_RE.search(text[end:]) is not None
    )


def _contains_prefix_unnegated_keyword(text: str, keyword: str) -> bool:
    """忽略后置排除，仅判断关键词出现前是否已被明确否定。"""
    for start, end in _keyword_occurrences(text, keyword):
        if not _is_negated_hazard(text[:end], start, end):
            return True
    return False


def _keyword_occurrences(text: str, keyword: str):
    """依次产出关键词在文本中的起止位置。"""
    start = text.find(keyword)
    while start >= 0:
        yield start, start + len(keyword)
        start = text.find(keyword, start + len(keyword))


def _combination_gap(
    subject_start: int,
    subject_end: int,
    symptom_start: int,
    symptom_end: int,
    clause: str,
) -> str:
    if subject_end <= symptom_start:
        return clause[subject_end:symptom_start]
    if symptom_end <= subject_start:
        return clause[symptom_end:subject_start]
    return ""


def _linked_combination_occurrences(
    signal: str,
    clause: str,
    subjects: tuple[str, ...],
    symptoms: tuple[str, ...],
):
    """产出同一受限语义片段内可配对的主语/症状位置。"""
    for subject in subjects:
        for symptom in symptoms:
            for subject_start, subject_end in _keyword_occurrences(clause, subject):
                for symptom_start, symptom_end in _keyword_occurrences(clause, symptom):
                    gap = _combination_gap(
                        subject_start,
                        subject_end,
                        symptom_start,
                        symptom_end,
                        clause,
                    )
                    linked = _COMBINATION_LINK_RE.fullmatch(gap) is not None
                    operation = (
                        _TEMPORAL_OPERATION_RE.search(gap) if linked else None
                    )
                    component_re = _INTERVENING_COMPONENT_RES.get(signal)
                    switches_component = bool(
                        subject_end <= symptom_start
                        and operation
                        and component_re
                        and component_re.search(
                            gap,
                            operation.end(),
                        )
                    )
                    if (
                        linked and not switches_component
                    ):
                        yield (
                            subject_start,
                            subject_end,
                            symptom_start,
                            symptom_end,
                        )


def _clause_positive_combination_signals(
    clause: str,
    subject_seen: dict[str, bool],
) -> set[str]:
    """识别当前片段中新建立或明确重新发生的组合危险。"""
    positive: set[str] = set()
    for signal, (subjects, symptoms) in RAW_INPUT_HAZARD_COMBINATIONS.items():
        for subject_start, subject_end, symptom_start, symptom_end in (
            _linked_combination_occurrences(
                signal,
                clause,
                subjects,
                symptoms,
            )
        ):
            last_end = max(subject_end, symptom_end)
            if _CAUSE_DETAIL_RE.search(clause[last_end:]):
                positive.add(signal)
                break
            subject_negated = _is_negated_hazard(
                clause[:subject_end],
                subject_start,
                subject_end,
            )
            symptom_negated = _is_negated_hazard(
                clause[:symptom_end],
                symptom_start,
                symptom_end,
            )
            if not subject_negated and not symptom_negated:
                positive.add(signal)
                break

        if subject_seen[signal] and _REASSERTION_CONTEXT_RE.search(clause):
            for symptom in symptoms:
                if not _contains_prefix_unnegated_keyword(clause, symptom):
                    continue
                remainder = clause.replace(symptom, "")
                remainder = _REASSERTION_FILLER_RE.sub("", remainder)
                if remainder in (
                    "", "它", "其", "该处", "这里", "那里",
                    "外皮", "绝缘层", "接头", "接口",
                ):
                    positive.add(signal)
                    break
        subject_seen[signal] = subject_seen[signal] or any(
            subject in clause for subject in subjects
        )
    return positive


def _exclusion_scope_bounds(clause: str, start: int, end: int) -> tuple[int, int]:
    """排除证据极性只读取最近标点/时序转折之间的局部片段。"""
    scope_start = 0
    scope_end = len(clause)
    for boundary in _EXCLUSION_POLARITY_BOUNDARIES:
        before = clause.rfind(boundary, 0, start)
        if before >= 0:
            scope_start = max(scope_start, before + len(boundary))
        after = clause.find(boundary, end)
        if after >= 0:
            scope_end = min(scope_end, after)
    return scope_start, scope_end


def _resolved_exclusion_matches(clause: str):
    """产出当前语义片段内肯定完成的排除；未决证据不得跨片段生效。"""
    for match in _POST_NEGATION_TOKEN_RE.finditer(clause):
        scope_start, scope_end = _exclusion_scope_bounds(
            clause,
            match.start(),
            match.end(),
        )
        scope = clause[scope_start:scope_end]
        local_start = match.start() - scope_start
        local_end = match.end() - scope_start
        if _CROSS_CLAUSE_CAUSE_REFERENCE_RE.search(scope):
            continue
        if any(
            unresolved.start() < local_end and unresolved.end() > local_start
            for unresolved in _UNRESOLVED_EXCLUSION_RE.finditer(scope)
        ):
            continue
        yield match


def _signals_in_exclusion_target(fragment: str) -> set[str]:
    """把排除宾语归一到 signal ID；组合危险允许用症状词显式指代。"""
    target_fragment = _EXCLUSION_TARGET_CONTRAST_RE.split(fragment, maxsplit=1)[0]
    targets = {
        signal
        for signal, keywords in RAW_INPUT_HAZARD_TERMS.items()
        if any(keyword in target_fragment for keyword in keywords)
    }
    for signal, (subjects, symptoms) in RAW_INPUT_HAZARD_COMBINATIONS.items():
        if _fragment_targets_combination(
            target_fragment,
            subjects,
            symptoms,
        ):
            targets.add(signal)
    return targets


def _fragment_targets_combination(
    fragment: str,
    subjects: tuple[str, ...],
    symptoms: tuple[str, ...],
) -> bool:
    """组合症状必须属于明确主语/组件，不能把外物同形词当作危险目标。"""
    for symptom in sorted(symptoms, key=len, reverse=True):
        for start, end in _keyword_occurrences(fragment, symptom):
            remainder = fragment[:start] + fragment[end:]
            for subject in subjects:
                remainder = remainder.replace(subject, "")
            for other_symptom in symptoms:
                remainder = remainder.replace(other_symptom, "")
            remainder = _CROSS_CLAUSE_ANAPHORA_RE.sub("", remainder)
            remainder = _LIST_CONNECTOR_RE.sub("", remainder)
            for filler in sorted(
                _COMBINATION_TARGET_FILLERS,
                key=len,
                reverse=True,
            ):
                remainder = remainder.replace(filler, "")
            if not remainder:
                return True
    return False


def _has_named_exclusion_object(fragment: str) -> bool:
    """区分显式外物宾语与“该情况”/检查桥等泛指表达。"""
    candidate = _CROSS_CLAUSE_ANAPHORA_RE.sub("", fragment)
    candidate = _LIST_CONNECTOR_RE.sub("", candidate)
    return bool(
        candidate
        and not _CROSS_CLAUSE_REFERENCE_BRIDGE_RE.fullmatch(candidate)
    )


def _explicit_exclusion_targets(
    clause: str,
    match: re.Match[str],
) -> tuple[set[str], bool]:
    """返回已识别 signal 目标及是否存在明确宾语；检查桥后的目标优先。"""
    prefix = clause[:match.start()]
    suffix = clause[match.end():]
    targets = _signals_in_exclusion_target(suffix)
    has_named_object = _has_named_exclusion_object(suffix)

    bridges = list(_EXCLUSION_TARGET_BRIDGE_RE.finditer(prefix))
    local_prefix = prefix[bridges[-1].end():] if bridges else prefix
    local_targets = _signals_in_exclusion_target(local_prefix)
    has_named_object = has_named_object or _has_named_exclusion_object(
        local_prefix
    )
    if local_targets:
        targets.update(local_targets)
    elif not bridges or _SHARED_POST_QUANTIFIER_RE.match(match.group()):
        targets.update(_signals_in_exclusion_target(prefix))
    return targets, has_named_object


def _clause_positive_direct_signals(clause: str) -> set[str]:
    """只按危险词出现前的语境建立候选，排除事件随后按时间顺序撤销。"""
    signals: set[str] = set()
    for signal, keywords in RAW_INPUT_HAZARD_TERMS.items():
        for keyword in keywords:
            if _contains_prefix_unnegated_keyword(clause, keyword):
                signals.add(signal)
                break
            if any(
                _CAUSE_DETAIL_RE.search(clause[end:])
                for _, end in _keyword_occurrences(clause, keyword)
            ):
                signals.add(signal)
                break
    return signals


def _is_direct_hazard_list_fragment(clause: str) -> bool:
    """判断跨逗号片段是否只是上一否定清单的危险词续项。"""
    remainder = clause
    for keywords in RAW_INPUT_HAZARD_TERMS.values():
        for keyword in keywords:
            remainder = remainder.replace(keyword, "")
    remainder = _LIST_CONNECTOR_RE.sub("", remainder)
    remainder = re.sub(r"(?:(?:的)?(?:现象|情况|迹象|风险|问题))", "", remainder)
    return remainder == ""


def _opens_negated_direct_list(clause: str) -> bool:
    """首个危险词被前置否定时，允许否定作用域延续到纯列表片段。"""
    for keywords in RAW_INPUT_HAZARD_TERMS.values():
        for keyword in keywords:
            for start, end in _keyword_occurrences(clause, keyword):
                if _is_negated_hazard(clause[:end], start, end):
                    return True
    return False


def _canonical_active_signals(
    direct_signals: set[str],
    combination_signals: set[str],
) -> set[str]:
    """同一裸线危险的直接词与组合词只计为一个活动候选。"""
    canonical = {_canonical_signal(signal) for signal in direct_signals}
    canonical.update(_canonical_signal(signal) for signal in combination_signals)
    return canonical


def _canonical_signal(signal: str) -> str:
    return "exposed_wire_damage" if signal == "exposed_wire" else signal


def _iter_hazard_clauses(text: str):
    """保留每个片段前的真实分隔符，避免否定清单跨越分号或句号。"""
    cursor = 0
    preceding_boundary = ""
    for match in _HAZARD_CLAUSE_SPLIT_RE.finditer(text):
        clause = text[cursor:match.start()]
        if clause:
            yield clause, preceding_boundary
        preceding_boundary = match.group()
        cursor = match.end()
    clause = text[cursor:]
    if clause:
        yield clause, preceding_boundary


def _detect_input_hazards_chronologically(text: str) -> list[str]:
    """统一按文本顺序维护直接/组合危险及已完成排除事件。"""
    active_direct: set[str] = set()
    active_combination: set[str] = set()
    subject_seen = {
        signal: False for signal in RAW_INPUT_HAZARD_COMBINATIONS
    }
    negated_list_open = False
    for clause, preceding_boundary in _iter_hazard_clauses(text):
        if preceding_boundary in _TEMPORAL_HAZARD_BOUNDARIES:
            clause = preceding_boundary + clause
        positive_direct = _clause_positive_direct_signals(clause)
        positive_combination = _clause_positive_combination_signals(
            clause,
            subject_seen,
        )
        list_continuation = (
            negated_list_open
            and preceding_boundary in ("，", ",")
            and _is_direct_hazard_list_fragment(clause)
            and not _REASSERTION_CONTEXT_RE.search(clause)
        )
        if not list_continuation:
            active_direct.update(positive_direct)
            active_combination.update(positive_combination)

        for match in _resolved_exclusion_matches(clause):
            targets, has_named_object = _explicit_exclusion_targets(
                clause,
                match,
            )
            shared_scope = _SHARED_POST_QUANTIFIER_RE.match(match.group()) is not None
            if has_named_object and not targets:
                continue
            if shared_scope:
                active_direct.clear()
                active_combination.clear()
                continue
            if targets:
                canonical_targets = {
                    _canonical_signal(signal) for signal in targets
                }
                active_direct = {
                    signal
                    for signal in active_direct
                    if _canonical_signal(signal) not in canonical_targets
                }
                active_combination = {
                    signal
                    for signal in active_combination
                    if _canonical_signal(signal) not in canonical_targets
                }
                continue

            canonical_active = _canonical_active_signals(
                active_direct,
                active_combination,
            )
            if len(canonical_active) != 1:
                continue
            only_signal = next(iter(canonical_active))
            active_direct = {
                signal
                for signal in active_direct
                if _canonical_signal(signal) != only_signal
            }
            active_combination.discard(only_signal)
        negated_list_open = (
            list_continuation
            or (not positive_direct and _opens_negated_direct_list(clause))
        )

    signals = [
        signal for signal in RAW_INPUT_HAZARD_TERMS if signal in active_direct
    ]
    for signal in RAW_INPUT_HAZARD_COMBINATIONS:
        if signal in active_combination and signal not in signals:
            signals.append(signal)
    if "exposed_wire_damage" in signals and "exposed_wire" in signals:
        signals.remove("exposed_wire")
    return signals


def detect_raw_input_hazards(text: str) -> list[str]:
    """返回原始用户输入命中的确定性危险信号，结果顺序稳定且不重复。"""
    normalized = "".join(str(text or "").lower().split())
    if not normalized:
        return []

    return _detect_input_hazards_chronologically(normalized)


def rule_risk_level(text: str) -> str:
    """规则兜底：按关键词判定风险等级（高→中→低）。"""
    low = text.lower()
    if any(kw in low for kw in RISK_HIGH_KEYWORDS):
        return "高"
    if any(kw in low for kw in RISK_MEDIUM_KEYWORDS):
        return "中"
    return "低"


# 生成节点兜底：检索失败/LLM 失败时的通用诊断与步骤（03 §7 不中断）
GENERIC_DIAGNOSIS = ["知识库未匹配到针对性资料，无法给出精确原因；以下为通用排查建议"]
GENERIC_STEPS = [
    "确认设备电源与指示灯状态，必要时重启设备",
    "检查连接（网络/线缆/外设）是否松动",
    "按设备说明书执行一次恢复默认设置",
    "若问题依旧，联系人工技术支持并附上设备型号与故障现象",
]


def normalize_model(name: str) -> str:
    """型号归一化：去空白、小写（用于测试集比对与关键词匹配）。"""
    return "".join(name.split()).lower()


# 设备参数表（01 §10 虚构型号；query_device_information 工具查询用）
DEVICE_INFO: dict[str, dict] = {
    "EduBot A1": {
        "name": "EduBot A1 教育机器人",
        "params": "尺寸 320×280×420mm / 重量 8kg / 12V 锂电池 / Wi-Fi、蓝牙、USB 接口",
    },
    "EduBot A2 Pro": {
        "name": "EduBot A2 Pro 进阶教育机器人",
        "params": "尺寸 360×300×480mm / 重量 10kg / 24V 锂电池 / Wi-Fi 6、蓝牙 5.0 / 激光雷达",
    },
    "EduCar X1": {
        "name": "EduCar X1 智能小车",
        "params": "尺寸 240×180×140mm / 重量 2.5kg / 7.4V 锂电池 / 超声波避障 / 摄像头",
    },
    "Humanoid H1": {
        "name": "Humanoid H1 人形机器人",
        "params": "高度 600mm / 重量 15kg / 36V 锂电池 / 17 自由度 / 语音交互 / 视觉导航",
    },
    "夸父 4 Pro": {
        "name": "夸父 4 Pro 通用服务机器人",
        "params": "高度 1200mm / 重量 42kg / 48V 锂电池 / 激光 SLAM 导航 / 机械臂 / 触控屏",
    },
    "Tonybot": {
        "name": "Tonybot 人形机器人（2025版）",
        "params": "ESP32 主控 / 16 个高压总线舵机 / 发光超声波、IMU、ESP32-S3 视觉、WonderEcho 语音模块 / 支持 Wonderbot 蓝牙、IoT Control Wi-Fi APP、Python、Scratch、Arduino",
    },
}


def rule_identify_device(text: str) -> str:
    """规则兜底：按关键词长度降序匹配型号（长关键词更具体优先），返回第一个命中。

    裸词别名（如「edubot」）只在前缀无更具体型号时才命中，避免
    「EduBot A2 Pro」被「edubot」误判为 EduBot A1。
    """
    low = text.lower()
    candidates = []
    for model, keywords in DEVICE_MODELS.items():
        for kw in keywords:
            candidates.append((len(kw), model, kw))
    candidates.sort(key=lambda x: x[0], reverse=True)  # 长关键词优先
    for _len, model, kw in candidates:
        if kw in low:
            return model
    return ""


def rule_classify_fault(text: str) -> str:
    """规则兜底：按关键词匹配故障类别，返回第一个命中类别；未命中返回「其他」。"""
    low = text.lower()
    for category, keywords in FAULT_CATEGORIES.items():
        if category == "其他":
            continue
        for kw in keywords:
            if kw in low:
                return category
    return "其他"
