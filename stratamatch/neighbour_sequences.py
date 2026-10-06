"""多层邻域序列与切断带联合排序；允许缺层，不把评分当作校准概率。"""
import numpy as np


def align_costs(costs, gap=.32):
    """带缺项的单调动态规划：每层最多对应一次，允许任意一侧缺失。"""
    n, m = costs.shape
    dp = np.zeros((n+1, m+1)); back = np.zeros((n+1, m+1), np.int8)
    dp[:, 0] = np.arange(n+1)*gap; dp[0, :] = np.arange(m+1)*gap
    back[1:, 0] = 1; back[0, 1:] = 2
    for i in range(1, n+1):
        for j in range(1, m+1):
            options = (dp[i-1, j-1]+costs[i-1, j-1], dp[i-1, j]+gap, dp[i, j-1]+gap)
            k = int(np.argmin(options)); dp[i, j] = options[k]; back[i, j] = k
    matches = []; i, j = n, m
    while i or j:
        k = back[i, j]
        if k == 0:
            matches.append((i-1, j-1)); i -= 1; j -= 1
        elif k == 1: i -= 1
        else: j -= 1
    return float(dp[n, m]), matches[::-1]


def neighbourhood(index, excluded, graph, ds, frame, angle, depth=4):
    """沿局部法向两侧提取多层链；端部切向链作为低权重参照。"""
    root = ds[index]; result = []
    for axis_kind in ('normal', 'tangent'):
        for sign in (-1, 1):
            current = index; seen = set(excluded) | {index}; chain = []; previous = None
            for step in range(depth):
                options = []
                for nxt, edge in graph[current].items():
                    if nxt in seen: continue
                    tangent, width = frame(ds[current], edge['point'])
                    # 对齐局部轴符号，避免 PCA 符号翻转导致遍历掉头。
                    if tangent @ root['axis'] < 0: tangent = -tangent
                    direction = np.array([-tangent[1], tangent[0]]) if axis_kind == 'normal' else tangent
                    anchor = previous if previous is not None else ds[current]['centre']
                    delta = edge['point']-anchor
                    forward = float(delta @ direction)*sign
                    if forward <= 0: continue
                    alignment = forward/max(float(np.linalg.norm(delta)), 1.)
                    nt, nw = frame(ds[nxt], edge['point'])
                    reliability = alignment * (edge['length']/(edge['length']+max(width, 2.)))
                    options.append((reliability, nxt, edge, nw, angle(tangent, nt)))
                if not options: break
                quality, nxt, edge, nw, local_angle = max(options, key=lambda v: v[0])
                _, rw = frame(root, edge['point'])
                chain.append({'part': ds[nxt]['id'], 'lith': ds[nxt]['lith'],
                              'relative_width': nw/max(rw, 2.), 'angle': local_angle,
                              'depth': step+1, 'quality': float(quality)})
                seen.add(nxt); current = nxt; previous = edge['point']
            result.append({'axis': axis_kind, 'side': sign, 'layers': chain})
    return result


def compare_neighbourhoods(a, b):
    """比较有序岩性与厚度/走向，跳层有代价，少量证据不能冒充高置信。"""
    alternatives = []
    for flip in (1, -1):
        total = mass = shape = support = 0.; matched = []
        for chain in a:
            other = next(c for c in b if c['axis'] == chain['axis'] and c['side'] == chain['side']*flip)
            aa, bb = chain['layers'], other['layers']
            if not aa and not bb: continue
            costs = np.full((len(aa), len(bb)), 1.2); shapes = np.ones_like(costs)
            for i, x in enumerate(aa):
                for j, y in enumerate(bb):
                    if x['lith'] != y['lith']: continue
                    # 厚度留给同岩性参考层的相对厚度比；此处只比较局部走向。
                    morph = abs(x['angle']-y['angle'])/90
                    shapes[i, j] = morph; costs[i, j] = .35*morph
            cost, pairs = align_costs(costs)
            factor = 1. if chain['axis'] == 'normal' else .22
            capacity = max(len(aa), len(bb)); mass += factor*capacity
            total += factor*cost
            for i, j in pairs:
                quality = np.sqrt(aa[i]['quality']*bb[j]['quality'])
                support += factor*quality/(1+.2*min(i, j))
                shape += factor*shapes[i, j]
                matched.append({'left': aa[i]['part'], 'right': bb[j]['part'], 'axis': chain['axis']})
        similarity = max(0., 1-total/max(.32*mass, 1e-9)) if mass else 0.
        confidence = support/(support+1.5)
        morphology = shape/max(len(matched), 1)
        alternatives.append((similarity*confidence, similarity, confidence, morphology, matched, flip))
    _, similarity, confidence, morphology, matched, flip = max(alternatives, key=lambda x: x[0])
    return 1-similarity, morphology, similarity, confidence, {'matched_layers': matched, 'side_flip': flip}


def short_axis_penalty(a, b, delta):
    """细长层的侧向续接支持随短/长轴比降低；块状层不施加该惩罚。"""
    unit = delta/max(float(np.linalg.norm(delta)), 1.)
    values = []
    ratios = []
    for d in (a, b):
        width = float(np.median(d['width_profile']))
        length = float(np.linalg.norm(np.diff(d['centreline'], axis=0), axis=1).sum())
        ratio = min(1., width/max(length, width, 1.)); ratios.append(ratio)
        side = 1-float(unit @ d['axis'])**2
        values.append(side*(1-ratio))
    return float(np.mean(values)), ratios


def joint_band_ranking(candidates, ds):
    """同一切断层两侧联合排序。缺层可跳过；只作为软证据，避免强迫匹配。"""
    by_id = {d['id']: d for d in ds}; bands = {}
    for edge in candidates:
        for cut in edge.get('cut_part_ids', []): bands.setdefault(cut, []).append(edge)
    votes = {}
    for cut, edges in bands.items():
        cutter = by_id[cut]; normal = np.array([-cutter['axis'][1], cutter['axis'][0]])
        left = set(); right = set(); usable = []
        for edge in edges:
            u, v = edge['left_part'], edge['right_part']
            su = float((by_id[u]['centre']-cutter['centre']) @ normal)
            sv = float((by_id[v]['centre']-cutter['centre']) @ normal)
            if su*sv >= 0: continue
            if su > 0: u, v = v, u
            left.add(u); right.add(v); usable.append((u, v, edge))
        if min(len(left), len(right)) < 2: continue
        # 沿切断层走向排列两侧的地层，联合求不交叉对应。
        order = lambda p: float(by_id[p]['centre'] @ cutter['axis'])
        left = sorted(left, key=order); right = sorted(right, key=order)
        li = {p:i for i,p in enumerate(left)}; ri = {p:i for i,p in enumerate(right)}
        costs = np.full((len(left), len(right)), 10.)
        for u,v,edge in usable: costs[li[u],ri[v]] = min(costs[li[u],ri[v]], edge['score'])
        optimum, matches = align_costs(costs, gap=.30)
        chosen = set(matches)
        strength = len(matches)/(len(matches)+2.)
        # 通过禁止某一对应重算，衡量整组方案的竞争差距；并列方案不给确定性奖励。
        margins = {}
        for pair in matches:
            alternate = costs.copy(); alternate[pair] = 10.
            alternative, _ = align_costs(alternate, gap=.30)
            margin = max(0., alternative-optimum)
            margins[pair] = margin/(margin+.08)
        average_margin = float(np.mean(list(margins.values()))) if margins else 0.
        for u,v,edge in usable:
            key = (edge['left_part'], edge['right_part'])
            pair = (li[u],ri[v]); reliability = margins.get(pair, average_margin)
            votes.setdefault(key, []).append((strength*reliability, pair in chosen, cut))
    for edge in candidates:
        entries = votes.get((edge['left_part'], edge['right_part']), [])
        if not entries: continue
        weight = sum(v[0] for v in entries)
        support = sum(v[0]*v[1] for v in entries)/max(weight, 1e-9)
        strength = float(np.mean([v[0] for v in entries]))
        edge['individual_score'] = edge['score']
        edge['joint_band'] = {'support': support, 'strength': strength, 'cut_parts': [v[2] for v in entries]}
        edge['score'] = round(max(0., edge['score']+.18*strength*(.5-support)), 5)
