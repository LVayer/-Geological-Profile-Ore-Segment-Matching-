"""跨切断带选择相容的整套对应；像素和局部岩性实例均不在此修改。"""
from collections import defaultdict

import numpy as np

from .band_registration import _band, _hypotheses, _solve, _matrix, _ratio_cost, _width
from .gradient_orientation import GradientOrientation
from .neighbour_sequences import align_costs


def _pair(u, v):
    return (min(u, v), max(u, v))


def _registration_models(band, by_id):
    """由多条候选层形成平移及缓变错移假设，再交给整套层序评判。"""
    axis, centre = band['axis'], band['centre']
    observations = []
    for (u, v), edge in band['edges'].items():
        pu = band['positions']['left', u]
        pv = band['positions']['right', v]
        t = float((pu-centre)@axis)
        displacement = float((pv-pu)@axis)
        observations.append((float(edge.get('individual_score', edge['score'])),
                             t, displacement, u, v))
    if not observations:
        return [(0., 0.)]
    widths = [float(np.median(by_id[p]['width_profile']))
              for p in [*band['left'], *band['right']]]
    tolerance = max(5., .7*float(np.median(widths)))
    span = max(1., np.ptp([row[1] for row in observations]))
    models = [(float(shift), 0.) for shift in _hypotheses(band, by_id)[:4]]
    # 两个不同层位的候选可提出仿射错移，重复岩性造成的假设由多层支持筛除。
    ranked = sorted(observations)[:32]
    trials = []
    for i, (_, ta, da, ua, va) in enumerate(ranked):
        for _, tb, db, ub, vb in ranked[i+1:]:
            if ua == ub or va == vb or abs(tb-ta) < 2*tolerance:
                continue
            slope = (db-da)/(tb-ta)
            if not -.8 < slope < 2.:
                continue  # 保持层序单调；倒转区需单独建模。
            intercept = da-slope*ta
            support = sum(np.exp(-.5*((d-intercept-slope*t)/tolerance)**2) *
                          np.exp(-max(0., score)/.45)
                          for score, t, d, _, _ in observations)
            trials.append((float(support), float(intercept), float(slope)))
    seen = {(round(a/tolerance), round(slope*span/tolerance)) for a, slope in models}
    for _, intercept, slope in sorted(trials, reverse=True):
        key = (round(intercept/tolerance), round(slope*span/tolerance))
        if key in seen:
            continue
        seen.add(key)
        models.append((intercept, slope))
        if len(models) >= 10:
            break
    return models


def _band_options(band, by_id, config, orientation=None):
    """对共同位移及序列缺项保留若干不同解，而不是立即锁定一组锚点。"""
    gap = float(config.get('band_missing_layer_cost', .38))
    limit = int(config.get('global_band_alternatives', 4))
    proposals = {}
    for intercept, slope in _registration_models(band, by_id):
        axis, centre = band['axis'], band['centre']
        displacement = lambda point, a=intercept, b=slope: a+b*float((point-centre)@axis)
        value, matches, costs, _ = _solve(
            band, displacement, [], by_id, gap, orientation)
        variants = [(value, matches)]
        # 禁止原方案中最有影响的几对，产生重复岩性/缺层时的竞争序列。
        ranked = sorted(matches, key=lambda ij: 2*gap-costs[ij], reverse=True)[:3]
        for i, j in ranked:
            alternative = costs.copy()
            alternative[i, j] = 10.
            variants.append(align_costs(alternative, gap))
        for _, pairs in variants:
            chosen = []
            for i, j in pairs:
                u, v = band['left'][i], band['right'][j]
                if (u, v) in band['edges'] and costs[i, j] < 2*gap:
                    chosen.append((u, v, float(costs[i, j])))
            minimum = 3 if abs(slope) > 1e-8 else 2
            if len(chosen) < minimum:
                continue  # 两点可恰好拟合仿射位移，第三层用于检验其泛化。
            key = frozenset(_pair(u, v) for u, v, _ in chosen)
            gain = sum(max(0., 2*gap-cost) for _, _, cost in chosen)
            offsets = [float((band['positions']['right', v] -
                              band['positions']['left', u]) @ band['axis']) -
                       displacement(band['positions']['left', u])
                       for u, v, _ in chosen]
            widths = [float(np.median(by_id[p]['width_profile']))
                      for p in [*band['left'], *band['right']]]
            scale = max(8., 2*float(np.median(widths)))
            spread = float(np.median(np.abs(offsets)))
            gain *= 1/(1+spread/scale+.15*abs(slope))
            item = {'cut': band['cut'], 'pairs': chosen, 'gain': gain,
                    'shift': intercept, 'slope': slope, 'spread': spread}
            if key not in proposals or gain > proposals[key]['gain']:
                proposals[key] = item
    ranked = sorted(proposals.values(), key=lambda item: item['gain'], reverse=True)
    return ranked[:max(1, limit)]


def _root(groups, node):
    return groups[node]


def _merge(groups, u, v, forbidden, support, chosen):
    """合并整条路径，并检查同一切断侧的重复地层是否被错误传递合并。"""
    a, b = _root(groups, u), _root(groups, v)
    if a == b:
        return groups, 0.
    left = [i for i, group in enumerate(groups) if group == a]
    right = [i for i, group in enumerate(groups) if group == b]
    if any(_pair(i, j) in forbidden for i in left for j in right):
        return None, 0.
    # 跨越多个切断带后闭合的独立候选，可以反过来支持前面的歧义选择。
    closure = sum(max(0., .55-cost)*.18 for i in left for j in right
                  if (key := _pair(i, j)) in support and key not in chosen
                  for cost in [support[key]])
    return tuple(a if group == b else group for group in groups), closure


def select_global_groups(candidates, descriptors, band_audit, config=None,
                         lithology_labels=None):
    """返回全图兼容的连接集合。束搜索在多个切断带间保留竞争解。"""
    cfg = config or {}
    orientation = GradientOrientation(lithology_labels) if lithology_labels is not None else None
    by_id = {d['id']: d for d in descriptors}
    index = {d['id']: i for i, d in enumerate(descriptors)}
    strong = {b['cut_part'] for b in band_audit.get('bands', [])
              if b['joint_evidence_sufficient']}
    grouped = defaultdict(list)
    support = {}
    for edge in candidates:
        u, v = edge['left_part'], edge['right_part']
        if u not in index or v not in index:
            continue
        key = _pair(index[u], index[v])
        score = float(edge.get('individual_score', edge['score']))
        support[key] = min(score, support.get(key, float('inf')))
        for cut in edge.get('cut_part_ids', []):
            if cut in strong:
                grouped[cut].append(edge)
    bands = [_band(cut, edges, by_id) for cut, edges in grouped.items()
             if cut in by_id]
    bands = [band for band in bands if band is not None]
    forbidden = set()
    for band in bands:
        # 仅把明显并排且轴向重叠的同色层设为硬冲突；沿同侧断续的碎片仍可续接。
        for side in ('left', 'right'):
            seq = band[side]
            for i, u in enumerate(seq):
                for v in seq[i+1:]:
                    a, b = by_id[u], by_id[v]
                    if a['lith'] != b['lith'] or abs(float(a['axis']@b['axis'])) < .9:
                        continue
                    axis = a['axis']
                    al, ah = np.percentile(a['sample']@axis, [2, 98])
                    bl, bh = np.percentile(b['sample']@axis, [2, 98])
                    if max(0., min(ah, bh)-max(al, bl)) > .65*min(ah-al, bh-bl):
                        forbidden.add(_pair(index[u], index[v]))
    options = [(band, _band_options(band, by_id, cfg, orientation)) for band in bands]
    options.sort(key=lambda item: (-len(item[1]), -len(item[0]['edges'])))
    n = len(descriptors)
    beam = [(0., tuple(range(n)), frozenset(), frozenset(), ())]
    width = int(cfg.get('global_beam_width', 32))
    for band, alternatives in options:
        expanded = []
        for score, groups, used, chosen, trace in beam:
            expanded.append((score, groups, used, chosen, trace + ((band['cut'], -1),)))
            for number, option in enumerate(alternatives):
                current_groups, current_used = groups, set(used)
                current_chosen = set(chosen)
                bonus = gain = 0.
                accepted = 0
                for u, v, cost in option['pairs']:
                    edge = band['edges'][u, v]
                    if edge['left_part'] == u:
                        ports = ((index[u], edge.get('left_tip', -1)),
                                 (index[v], edge.get('right_tip', -1)))
                    else:
                        ports = ((index[u], edge.get('right_tip', -1)),
                                 (index[v], edge.get('left_tip', -1)))
                    # 每个实例端口最多连接一条候选续接。
                    if any(port in current_used for port in ports):
                        continue
                    result, closure = _merge(current_groups, index[u], index[v],
                                             forbidden, support, current_chosen)
                    if result is None:
                        continue
                    current_groups = result
                    bonus += closure
                    gain += max(0., 2*float(cfg.get('band_missing_layer_cost', .38))-cost)
                    accepted += 1
                    current_used.update(ports)
                    current_chosen.add(_pair(index[u], index[v]))
                # 多处切断带可共用实例；只舍弃冲突对，保留仍有整套支撑的部分。
                minimum = 3 if abs(option.get('slope', 0.)) > 1e-8 else 2
                if accepted >= minimum:
                    local_widths = [np.median(by_id[p]['width_profile']) for p in
                                    [*band['left'], *band['right']]]
                    scale = max(8., 2*float(np.median(local_widths)))
                    gain *= 1/(1+option['spread']/scale+
                               .15*abs(option.get('slope', 0.)))
                    expanded.append((score+gain+bonus, current_groups,
                                     frozenset(current_used), frozenset(current_chosen),
                                     trace + ((band['cut'], number),)))
        # 相同分组和已占端口只留最高分；有限束宽控制大图运行时间。
        distinct = {}
        for state in expanded:
            key = (state[1], state[2])
            if key not in distinct or state[0] > distinct[key][0]:
                distinct[key] = state
        beam = sorted(distinct.values(), key=lambda state: state[0], reverse=True)[:max(2, width)]
    score, groups, used, chosen, trace = beam[0]
    decisions = {cut: number for cut, number in trace}
    # 无整套证据的直接续接，仍要求原来的双方互选与歧义检查。
    direct = []
    choices = defaultdict(list)
    for edge in candidates:
        if edge.get('cut_part_ids') and any(cut in strong for cut in edge['cut_part_ids']):
            continue
        if edge.get('band_unselected') or not edge.get('contact_evidence_sufficient', True):
            continue
        if edge['left_part'] not in index or edge['right_part'] not in index:
            continue
        for side in ('left', 'right'):
            choices[(edge[side+'_index'], edge[side+'_tip'])].append(edge)
        direct.append(edge)
    for edges in choices.values():
        edges.sort(key=lambda edge: edge['score'])
    for edge in sorted(direct, key=lambda edge: edge['score']):
        if edge['score'] > float(cfg.get('group_max_score', .68)):
            continue
        ports = ((edge['left_index'], edge['left_tip']),
                 (edge['right_index'], edge['right_tip']))
        if any(port in used or choices[port][0] is not edge for port in ports):
            continue
        if any(len(choices[port]) > 1 and choices[port][1]['score']-edge['score'] <
               float(cfg.get('group_ambiguity_margin', .035)) for port in ports):
            continue
        u, v = edge['left_index'], edge['right_index']
        key = _pair(u, v)
        if key in chosen or groups[u] == groups[v]:
            continue
        result, _ = _merge(groups, u, v, forbidden, support, chosen)
        if result is None:
            continue
        groups = result
        used = frozenset(set(used) | set(ports))
        chosen = frozenset(set(chosen) | {key})
    for edge in candidates:
        if edge['left_index'] < n and edge['right_index'] < n:
            key = _pair(edge['left_index'], edge['right_index'])
            edge['decision'] = 'grouped' if key in chosen else 'not_selected_globally'
    model_by_cut = {band['cut']: alternatives for band, alternatives in options}
    selected_models = [{'cut_part': cut,
                        'intercept_pixels': model_by_cut[cut][number].get('shift', 0.),
                        'slope': model_by_cut[cut][number].get('slope', 0.),
                        'residual_pixels': model_by_cut[cut][number]['spread']}
                       for cut, number in trace if number >= 0]
    return groups, chosen, {'method': 'gradient_affine_cross_band_beam_search',
                            'bands_considered': len(options),
                            'band_alternatives': sum(len(item[1]) for item in options),
                            'selected_bands': sum(number >= 0 for number in decisions.values()),
                            'selected_pairs': len(chosen), 'objective_gain': round(score, 4),
                            'beam_width': width,
                            'gradient_samples': orientation.samples if orientation else 0,
                            'reliable_gradient_samples': orientation.reliable if orientation else 0,
                            'selected_displacement_models': selected_models,
                            'band_choices': [{'cut_part': cut, 'choice': number}
                                             for cut, number in trace]}


def _sequence_cost(band, i, j, by_id, radius=3):
    """沿切断层比较目标前后的岩性层序；动态规划允许任意一侧缺层。"""
    def side_cost(a, b):
        if not a and not b:
            return None
        costs = np.array([[0. if by_id[u]['lith'] == by_id[v]['lith'] else .8
                           for v in b] for u in a], float).reshape(len(a), len(b))
        value, _ = align_costs(costs, .28)
        return min(1., value / max(.28 * max(len(a), len(b)), .28))
    values = []
    for left, right in ((band['left'][max(0, i-radius):i],
                         band['right'][max(0, j-radius):j]),
                        (band['left'][i+1:i+radius+1],
                         band['right'][j+1:j+radius+1])):
        cost = side_cost(left, right)
        if cost is not None:
            values.append(cost)
    return float(np.mean(values)) if values else .5


def _progressive_band(band, by_id, anchors, orientation, config):
    """在当前控制层下重估位移，并只提出本切断带的下一对候选。"""
    gap = float(config.get('band_missing_layer_cost', .38))
    widths = [_width(by_id[p], band['positions'][side, p])
              for side, seq in (('left', band['left']), ('right', band['right'])) for p in seq]
    thickness = float(np.median(widths)) if widths else 1.
    # N 是本切断带两侧受影响的地层数，不是连接路径穿过的切断体数。
    affected = min(len(band['left']), len(band['right']))
    models = _registration_models(band, by_id)[:int(config.get('progressive_models', 6))]
    if anchors:
        offsets = [float((band['positions']['right', v] -
                          band['positions']['left', u]) @ band['axis']) for u, v in anchors]
        models.insert(0, (float(np.median(offsets)), 0.))
    best = None
    for intercept, slope in models:
        shift = lambda point, a=intercept, b=slope: a+b*float((point-band['centre'])@band['axis'])
        _, details = _matrix(band, shift, anchors, by_id, orientation)
        ratio = abs(intercept + slope * float((band['positions']['left', band['left'][0]]-
                                               band['centre']) @ band['axis'])) / max(thickness, 1.)
        # 位移超过一层厚或影响至少三层时，逐渐转向层序和相对厚度。
        disturbance = max(float(affected >= 3), float(np.clip((ratio-.8)/.2, 0., 1.)))
        seq_weight = .45-.05*disturbance
        thick_weight = .10+.20*disturbance
        geom_weight = .40-.30*disturbance
        costs = np.full((len(band['left']), len(band['right'])), 10., float)
        features = {}
        for (u, v), edge in band['edges'].items():
            i, j = band['left'].index(u), band['right'].index(v)
            d = details.get((i, j))
            if d is None or by_id[u]['lith'] != by_id[v]['lith']:
                continue
            sequence = _sequence_cost(band, i, j, by_id)
            relative = _ratio_cost(u, v, anchors, band, by_id)
            # 无同岩性参考层时是证据缺失，厚度权重转给可靠的层序及几何项。
            sw, gw = seq_weight, geom_weight
            if relative is None:
                sw += thick_weight*.7
                gw += thick_weight*.3
            geometry = min(1., .58*d['corrected_direction'] +
                           .42*min(1., d['shift_residual']))
            morphology = min(1., float(edge.get('costs', {}).get('morphology', .5)))
            score = sw*sequence + gw*geometry + .05*morphology
            if relative is not None:
                score += thick_weight*relative
            # 图像接触证据仍作小幅可靠性修正，不能覆盖上述自适应权重。
            score += .07*min(1., float(edge.get('individual_score', edge['score'])))
            costs[i, j] = score
            features[(i, j)] = {'sequence': sequence, 'relative_thickness': relative,
                                'corrected_geometry': geometry, 'morphology': morphology,
                                'weights': {'sequence': sw, 'relative_thickness':
                                            thick_weight if relative is not None else 0.,
                                            'corrected_geometry': gw, 'morphology': .05},
                                'affected_layers': affected, 'displacement_over_thickness': ratio}
        # 已确认的控制层必须保留，并限制本轮其他候选的行列。
        for u, v in anchors:
            i, j = band['left'].index(u), band['right'].index(v)
            costs[i, :] = 10.; costs[:, j] = 10.; costs[i, j] = 0.
        value, matches = align_costs(costs, gap)
        # 多层层序用于选择位移模型；一轮仍只能从该方案中确认一对。
        model_penalty = .04*abs(slope) + .03*abs(intercept)/max(thickness, 1.)
        anchor_penalty = sum(min(2., abs(float((band['positions']['right', v]-
                                              band['positions']['left', u])@band['axis'])-
                                      shift(band['positions']['left', u]))/max(thickness, 1.))
                             for u, v in anchors)
        objective = value + model_penalty + .35*anchor_penalty
        if best is None or objective < best[0]:
            best = (objective, value, matches, costs, features, intercept, slope, disturbance)
    if best is None:
        return []
    _, value, matches, costs, features, intercept, slope, disturbance = best
    proposed = []
    for i, j in matches:
        u, v = band['left'][i], band['right'][j]
        if (u, v) in anchors or costs[i, j] >= 2*gap:
            continue
        alternative = costs.copy(); alternative[i, j] = 10.
        alternative_value, _ = align_costs(alternative, gap)
        margin = max(0., alternative_value-value)
        gain = max(0., 2*gap-costs[i, j])
        # 唯一性与多层共同位移同时支持时才优先成为下一控制层。
        priority = gain*(.35+.65*margin/(margin+.12))
        proposed.append({'u': u, 'v': v, 'cut': band['cut'], 'priority': priority,
                         'score': float(costs[i, j]), 'margin': margin,
                         'shift': intercept, 'slope': slope,
                         'disturbance': disturbance, 'features': features[(i, j)]})
    return proposed


def select_progressive_groups(candidates, descriptors, band_audit, config=None,
                              lithology_labels=None):
    """一次确认一对；新控制层随即重算相关切断带，并检查跨带成组冲突。"""
    cfg = config or {}
    by_id = {d['id']: d for d in descriptors}
    index = {d['id']: i for i, d in enumerate(descriptors)}
    orientation = GradientOrientation(lithology_labels) if lithology_labels is not None else None
    strong = {item['cut_part'] for item in band_audit.get('bands', [])
              if item.get('joint_evidence_sufficient')}
    grouped = defaultdict(list)
    support = {}
    for edge in candidates:
        u, v = edge['left_part'], edge['right_part']
        if u not in index or v not in index or by_id[u]['lith'] != by_id[v]['lith']:
            continue
        pair = _pair(index[u], index[v])
        support[pair] = min(float(edge.get('individual_score', edge['score'])),
                            support.get(pair, float('inf')))
        for cut in edge.get('cut_part_ids', []):
            if cut in strong:
                grouped[cut].append(edge)
    bands = {cut: band for cut, edges in grouped.items()
             if cut in by_id and (band := _band(cut, edges, by_id)) is not None}
    forbidden = set()
    for band in bands.values():
        # 同侧并排且长轴显著重叠的同色层不能经跨带传递合为一个地层。
        for side in ('left', 'right'):
            seq = band[side]
            for k, u in enumerate(seq):
                for v in seq[k+1:]:
                    a, b = by_id[u], by_id[v]
                    if a['lith'] != b['lith'] or abs(float(a['axis']@b['axis'])) < .9:
                        continue
                    axis = a['axis']
                    al, ah = np.percentile(a['sample']@axis, [2, 98])
                    bl, bh = np.percentile(b['sample']@axis, [2, 98])
                    if max(0., min(ah, bh)-max(al, bl)) > .65*min(ah-al, bh-bl):
                        forbidden.add(_pair(index[u], index[v]))
    groups = tuple(range(len(descriptors)))
    chosen, used = set(), set()
    anchors = {cut: [] for cut in bands}
    proposals = {cut: _progressive_band(band, by_id, [], orientation, cfg)
                 for cut, band in bands.items()}
    direct = [edge for edge in candidates if not any(
        cut in strong for cut in edge.get('cut_part_ids', []))
              and edge.get('contact_evidence_sufficient', True)
              and edge['left_part'] in index and edge['right_part'] in index]
    direct_choices = defaultdict(list)
    for edge in direct:
        for side in ('left', 'right'):
            direct_choices[(edge[side+'_index'], edge[side+'_tip'])].append(edge)
    for choices in direct_choices.values():
        choices.sort(key=lambda edge: edge['score'])
    direct = [edge for edge in direct if edge['score'] <= float(cfg.get('group_max_score', .68))
              and all(direct_choices[(edge[side+'_index'], edge[side+'_tip'])][0] is edge
                      for side in ('left', 'right'))]
    direct = [edge for edge in direct if all(
        len(direct_choices[(edge[side+'_index'], edge[side+'_tip'])]) < 2 or
        direct_choices[(edge[side+'_index'], edge[side+'_tip'])][1]['score']-
        edge['score'] >= float(cfg.get('group_ambiguity_margin', .035))
        for side in ('left', 'right'))]
    history = []
    while True:
        available = []
        for cut, items in proposals.items():
            for item in items:
                u, v = item['u'], item['v']
                edge = bands[cut]['edges'][u, v]
                if edge['left_part'] == u:
                    ports = ((index[u], edge.get('left_tip', -1)),
                             (index[v], edge.get('right_tip', -1)))
                else:
                    ports = ((index[u], edge.get('right_tip', -1)),
                             (index[v], edge.get('left_tip', -1)))
                if any(port in used for port in ports):
                    continue
                a, b = index[u], index[v]
                if groups[a] == groups[b]:
                    continue
                result, closure = _merge(groups, a, b, forbidden, support, chosen)
                if result is None:
                    continue
                # 其他切断带已有对应能够为本轮提供独立的闭合支持。
                priority = item['priority'] + closure
                available.append((priority, item['margin'], -item['score'],
                                  cut, item, ports, result))
        for edge in direct:
            a, b = edge['left_index'], edge['right_index']
            ports = ((a, edge['left_tip']), (b, edge['right_tip']))
            if any(port in used for port in ports) or groups[a] == groups[b]:
                continue
            result, closure = _merge(groups, a, b, forbidden, support, chosen)
            if result is None:
                continue
            priority = .4*max(0., .68-float(edge['score']))+closure
            item = {'u': edge['left_part'], 'v': edge['right_part'],
                    'score': float(edge['score']), 'margin': 0., 'features': {}}
            available.append((priority, 0., -float(edge['score']), None,
                              item, ports, result))
        if not available:
            break
        priority, _, _, cut, item, ports, result = max(available,
            key=lambda row: (row[0], row[1], row[2]))
        if priority < float(cfg.get('progressive_min_gain', .055)):
            break
        u, v = item['u'], item['v']
        groups = result
        chosen.add(_pair(index[u], index[v]))
        used.update(ports)
        history.append({'round': len(history)+1, 'left_part': u, 'right_part': v,
                        'cut_part': cut, 'priority': round(float(priority), 5),
                        'score': round(float(item['score']), 5),
                        'alternative_margin': round(float(item['margin']), 5),
                        'features': item['features']})
        # 同一对可能跨越多个切断体；各切断带维护自己的控制层和位移。
        for other, band in bands.items():
            if (u, v) in band['edges'] and (u, v) not in anchors[other]:
                anchors[other].append((u, v))
                proposals[other] = _progressive_band(
                    band, by_id, anchors[other], orientation, cfg)
    for edge in candidates:
        if edge['left_index'] < len(descriptors) and edge['right_index'] < len(descriptors):
            edge['decision'] = ('grouped' if _pair(edge['left_index'], edge['right_index'])
                                in chosen else 'not_selected_globally')
    return groups, frozenset(chosen), {
        'method': 'adaptive_progressive_one_pair_per_round',
        'bands_considered': len(bands), 'selected_pairs': len(chosen),
        'selected_bands': sum(bool(value) for value in anchors.values()),
        'gradient_samples': orientation.samples if orientation else 0,
        'reliable_gradient_samples': orientation.reliable if orientation else 0,
        'rounds': history,
        'selected_displacement_models': [
            {'cut_part': cut, 'control_pairs': len(pairs),
             'apparent_shift_pixels': float(np.median([
                 (band['positions']['right', v]-band['positions']['left', u]) @ band['axis']
                 for u, v in pairs]))}
            for cut, pairs in anchors.items() if pairs for band in [bands[cut]]],
    }
