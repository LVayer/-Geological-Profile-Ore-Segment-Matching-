"""跨多条切断带保留竞争解释；每轮只提交一对可复核的地层控制关系。"""
from collections import defaultdict

import numpy as np

from .band_registration import _band, _frame, _ratio_cost, _width
from .global_layer_matching import _merge, _pair, _registration_models, _sequence_cost
from .gradient_orientation import GradientOrientation
from .neighbour_sequences import align_costs


def _ports(edge, index, u, v):
    if edge['left_part'] == u:
        return ((index[u], edge.get('left_tip', -1)),
                (index[v], edge.get('right_tip', -1)))
    return ((index[u], edge.get('right_tip', -1)),
            (index[v], edge.get('left_tip', -1)))


def _relative_width(u, v, controls, band, by_id):
    """用两侧都存在的同岩性控制层作厚度参照，包括跨切断带传来的控制层。"""
    local = [(a, b) for a, b in controls if ('left', a) in band['positions']
             and ('right', b) in band['positions']]
    value = _ratio_cost(u, v, local, band, by_id)
    if value is not None:
        return value
    axis = band['axis']; normal = np.array([-axis[1], axis[0]])
    origin = band['centre']; lith = by_id[u]['lith']
    pu, pv = band['positions']['left', u], band['positions']['right', v]
    references = []
    for a, b in controls:
        if a in (u, v) or b in (u, v) or by_id[a]['lith'] != lith or by_id[b]['lith'] != lith:
            continue
        ca, cb = by_id[a]['centre'], by_id[b]['centre']
        if (ca-origin)@normal >= 0 or (cb-origin)@normal <= 0:
            continue
        separation = abs(float((ca-pu)@axis))+abs(float((cb-pv)@axis))
        references.append((separation, a, b))
    if not references:
        return None
    _, a, b = min(references)
    ra = _width(by_id[u], pu)/_width(by_id[a], by_id[a]['centre'])
    rb = _width(by_id[v], pv)/_width(by_id[b], by_id[b]['centre'])
    return min(1., abs(float(np.log(max(ra, 1e-6)/max(rb, 1e-6))))/.7)


def _band_options(band, by_id, controls, orientation, cfg, excluded=None):
    """重拟合共同位移，保留整套层序的多种可行解释和缺层方案。"""
    gap = float(cfg.get('band_missing_layer_cost', .38))
    anchors = [(u, v) for u, v in controls if (u, v) in band['edges']]
    models = _registration_models(band, by_id)[:int(cfg.get('hypothesis_models', 6))]
    axis, centre = band['axis'], band['centre']
    if anchors:
        offsets = [float((band['positions']['right', v]-band['positions']['left', u])@axis)
                   for u, v in anchors]
        models.insert(0, (float(np.median(offsets)), 0.))
    left_index = {u: i for i, u in enumerate(band['left'])}
    right_index = {v: j for j, v in enumerate(band['right'])}
    widths = [_width(by_id[p], band['positions'][side, p])
              for side, seq in (('left', band['left']), ('right', band['right'])) for p in seq]
    thickness = max(2., float(np.median(widths)))
    static = {}
    for (u, v), edge in band['edges'].items():
        i, j = left_index[u], right_index[v]
        neighbour = _sequence_cost(band, i, j, by_id)
        # 全同岩性的短序列无法辨认层位；保持中性，不能误报为完美层序。
        nearby = band['left'][max(0, i-3):i+4]+band['right'][max(0, j-3):j+4]
        informative = len({by_id[p]['lith'] for p in nearby}) > 1
        if not informative:
            neighbour = .5
        relative = _relative_width(u, v, controls, band, by_id)
        static[i, j] = (edge, neighbour, relative)
    proposals = {}
    for intercept, slope in models:
        costs = np.full((len(band['left']), len(band['right'])), 10., float)
        features = {}
        for (i, j), (edge, neighbour, relative) in static.items():
            u, v = band['left'][i], band['right'][j]
            if by_id[u]['lith'] != by_id[v]['lith']:
                continue
            if excluded == _pair(u, v):
                continue
            pu, pv = band['positions']['left', u], band['positions']['right', v]
            displacement = intercept+slope*float((pu-centre)@axis)
            corrected = pv-pu-displacement*axis
            if orientation is None:
                ta, tb = _frame(by_id[u], pu), _frame(by_id[v], pv)
            else:
                ta, _ = orientation.sample(by_id[u], pu)
                tb, _ = orientation.sample(by_id[v], pv)
            angle = np.degrees(np.arccos(np.clip(abs(float(ta@tb)), 0., 1.)))/90.
            norm = float(np.linalg.norm(corrected))
            direction = angle if norm < 1. else (.5*angle+
                .25*(1-abs(float(corrected@ta))/norm)+
                .25*(1-abs(float(corrected@tb))/norm))
            residual = min(1., abs(float((pv-pu)@axis)-displacement)/(2*thickness))
            geometry = min(1., .58*direction+.42*residual)
            morphology = min(1., float(edge.get('costs', {}).get('morphology', .5)))
            # 有一层受到切断即使用约定权重；未知厚度记中性，不占为层序的支持。
            score = .40*neighbour+.30*(relative if relative is not None else .5)+\
                    .25*geometry+.05*morphology
            costs[i, j] = score
            features[i, j] = {'sequence': neighbour, 'relative_thickness': relative,
                             'corrected_geometry': geometry, 'morphology': morphology,
                             'weights': {'sequence': .40, 'relative_thickness': .30,
                                         'corrected_geometry': .25, 'morphology': .05}}
        for u, v in anchors:
            i, j = left_index[u], right_index[v]
            costs[i, :] = 10.; costs[:, j] = 10.; costs[i, j] = 0.
        best_value, matches = align_costs(costs, gap)
        variants = [(best_value, matches)]
        for i, j in sorted((p for p in matches if p in features and
                            (band['left'][p[0]], band['right'][p[1]]) not in anchors),
                           key=lambda p: costs[p])[:2]:
            alternate = costs.copy(); alternate[i, j] = 10.
            variants.append(align_costs(alternate, gap))
        for value, pairs in variants:
            selected = [(band['left'][i], band['right'][j], float(costs[i, j]))
                        for i, j in pairs if (i, j) in features and costs[i, j] < 2*gap
                        and (band['left'][i], band['right'][j]) not in anchors]
            if not selected:
                continue
            anchor_error = sum(min(2., abs(float((band['positions']['right', v]-
                                              band['positions']['left', u])@axis)-
                                      (intercept+slope*float((band['positions']['left', u]-centre)@axis))
                                      )/thickness) for u, v in anchors)
            gain = sum(max(0., 2*gap-cost) for _, _, cost in selected)
            gain -= .06*abs(slope)+.03*anchor_error
            # 稳定的共同位移优于靠少数偶然对应拟合出的剧烈剪切。
            residuals = [float((band['positions']['right', v]-
                                band['positions']['left', u])@axis)-
                         (intercept+slope*float((band['positions']['left', u]-centre)@axis))
                         for u, v, _ in selected]
            spread = float(np.median(np.abs(residuals))) if residuals else 0.
            gain /= 1.+spread/max(2*thickness, 1.)
            key = frozenset(_pair(u, v) for u, v, _ in selected)
            item = {'pairs': selected, 'gain': gain, 'shift': intercept,
                    'slope': slope, 'spread': spread, 'features': features}
            if key not in proposals or gain > proposals[key]['gain']:
                proposals[key] = item
    return sorted(proposals.values(), key=lambda item: item['gain'], reverse=True)[
        :int(cfg.get('hypothesis_band_options', 4))]


def _bridge_options(edges, by_id, controls, cfg, excluded=None):
    """跨多条切断带的候选也作为竞争假设；邻层缺失只降低把握，不一票否决。"""
    confirmed = {_pair(u, v) for u, v in controls}
    options = []
    for edge in edges:
        u, v = edge['left_part'], edge['right_part']
        if _pair(u, v) == excluded:
            continue
        matched = edge.get('sequence_alignment', {}).get('matched_layers', [])
        normal = [item for item in matched if item.get('axis') == 'normal']
        support = sum(_pair(item['left'], item['right']) in confirmed for item in normal)
        confidence = min(1., float(edge.get('neighbour_confidence', 0.)))
        similarity = float(edge.get('contact_similarity', .5))
        # 已确认的两侧邻层逐步提高层序可信度；缺层时仍保留中性分数。
        sequence = .5 + confidence*(.5-similarity) - .12*min(2, support)
        sequence = float(np.clip(sequence, 0., 1.))
        reference = []
        pu, pv = (np.asarray(edge['curve_work_pixels'][k], float) for k in (0, -1))
        for a, b in controls:
            if a in (u, v) or b in (u, v) or by_id[a]['lith'] != by_id[u]['lith']:
                continue
            distance = float(np.linalg.norm(by_id[a]['centre']-pu)+
                             np.linalg.norm(by_id[b]['centre']-pv))
            ra = _width(by_id[u], pu)/_width(by_id[a], by_id[a]['centre'])
            rb = _width(by_id[v], pv)/_width(by_id[b], by_id[b]['centre'])
            reference.append((distance, min(1., abs(float(np.log(max(ra, 1e-6)/
                                      max(rb, 1e-6))))/.7)))
        relative = min(reference)[1] if reference else .5
        votes = edge.get('band_registration', {}).get('votes', [])
        selected = [vote for vote in votes if vote.get('selected')]
        base = float(edge.get('costs', {}).get('direction', .5))
        if selected:
            quality = max(float(vote.get('band_quality', 0.)) for vote in selected)
            geometry = .65*base+.35*(1.-min(1., quality))
        else:
            geometry = .75*base+.25*.5
        morphology = float(edge.get('costs', {}).get('morphology', .5))
        cost = .40*sequence+.30*relative+.25*geometry+.05*morphology
        # 极长的搜索路径可误穿许多细层；复杂度是软代价，邻层控制可抵消它。
        excess_cuts = max(0, len(edge.get('cut_part_ids') or [])-3)
        cost += min(.35, .025*excess_cuts)/(1.+.5*support)
        # 不把邻层数量作为门槛。低质量观测只降低收益，仍可与其他方案竞争。
        gain = max(0., .76-cost)*(0.75+.25*confidence)
        edge['multi_cut_sequence_support'] = support
        edge['global_direct_score'] = cost
        options.append({'pairs': [(u, v, cost)], 'gain': gain,
                        'features': {'sequence': sequence,
                                     'relative_thickness': relative,
                                     'corrected_geometry': geometry,
                                     'morphology': morphology}})
    return sorted(options, key=lambda item: item['gain'], reverse=True)[
        :int(cfg.get('hypothesis_bridge_options', 3))]


def _forbidden_neighbours(bands, by_id, index):
    forbidden = set()
    for band in bands.values():
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
    return forbidden


def _search(bands, options, index, n, forbidden, support, width, controls, excluded=None):
    """束搜索保留互不冲突的跨带解释；尚不提交任何整套匹配。"""
    groups = tuple(range(n)); used = set(); chosen = set()
    for u, v, edge in controls:
        pair = _pair(index[u], index[v]); ports = _ports(edge, index, u, v)
        result, _ = _merge(groups, index[u], index[v], forbidden, support, chosen)
        if result is None:
            return []
        groups = result; used.update(ports); chosen.add(pair)
    beam = [(0., groups, frozenset(used), frozenset(chosen), ())]
    for cut, band in sorted(bands.items(), key=lambda x: -max(
            (item['gain'] for item in options.get(x[0], [])), default=0.)):
        alternatives = options.get(cut, [])
        expanded = []
        for score, groups, used, chosen, trace in beam:
            expanded.append((score, groups, used, chosen, trace+((cut, -1),)))
            for number, option in enumerate(alternatives):
                current = groups; ports_used = set(used); picked = set(chosen)
                gain = 0.; accepted = 0
                for u, v, cost in option['pairs']:
                    pair = _pair(index[u], index[v])
                    if pair == excluded or (isinstance(excluded, set) and pair in excluded) or pair in picked:
                        continue
                    edge = band['edges'][u, v]
                    ports = _ports(edge, index, u, v)
                    if any(p in ports_used for p in ports):
                        continue
                    merged, closure = _merge(current, index[u], index[v],
                                             forbidden, support, picked)
                    if merged is None or merged == current:
                        continue
                    current = merged; picked.add(pair); ports_used.update(ports)
                    gain += max(0., 2*.38-cost)+closure; accepted += 1
                if accepted:
                    gain *= max(.45, option['gain']/max(sum(
                        max(0., .76-c) for _, _, c in option['pairs']), .01))
                    expanded.append((score+gain, current, frozenset(ports_used),
                                     frozenset(picked), trace+((cut, number),)))
        distinct = {}
        for state in expanded:
            key = (state[2], state[3])
            if key not in distinct or state[0] > distinct[key][0]:
                distinct[key] = state
        beam = sorted(distinct.values(), key=lambda x: x[0], reverse=True)[:width]
    return beam


def select_hypothesis_groups(candidates, descriptors, band_audit, config=None,
                             lithology_labels=None, virtual_descriptors=()):
    """多方案试算、每轮一对暂定控制；分组直到所有轮次完成后才固定。"""
    cfg = config or {}; n = len(descriptors)
    by_id = {d['id']: d for d in [*descriptors, *virtual_descriptors]}
    index = {d['id']: i for i, d in enumerate(descriptors)}
    orientation = GradientOrientation(lithology_labels) if lithology_labels is not None else None
    grouped = defaultdict(list); support = {}
    for edge in candidates:
        u, v = edge['left_part'], edge['right_part']
        if u not in index or v not in index or by_id[u]['lith'] != by_id[v]['lith']:
            continue
        pair = _pair(index[u], index[v])
        support[pair] = min(float(edge.get('individual_score', edge['score'])),
                            support.get(pair, float('inf')))
        # 跨多切断体的长路径只能提供全局软证据，不能污染单条切断边界的层序。
        cuts = edge.get('cut_part_ids') or []
        if len(cuts) == 1 and cuts[0] in by_id:
            grouped[cuts[0]].append(edge)
    bands = {cut: band for cut, edges in grouped.items()
             if (band := _band(cut, edges, by_id)) is not None
             and len(band['edges']) >= 2}
    forbidden = _forbidden_neighbours(bands, by_id, index)
    # 同一个端点的跨带路径构成互斥备选，和单带层序进入同一全局搜索。
    bridges = defaultdict(list)
    for edge in candidates:
        u, v = edge['left_part'], edge['right_part']
        if u not in index or v not in index or by_id[u]['lith'] != by_id[v]['lith']:
            continue
        cuts = edge.get('cut_part_ids') or []
        if len(cuts) == 1 and cuts[0] in bands:
            continue
        if len(cuts) <= 1 and edge.get('band_unselected'):
            continue
        if not edge.get('contact_evidence_sufficient', True):
            continue
        if not edge.get('curve_work_pixels') or edge['score'] > float(
                cfg.get('group_max_score', .68)):
            continue
        bridges[('bridge', index[u], edge.get('left_tip', -1))].append(edge)
    for key, edges in bridges.items():
        # 每个端点只保留少量有竞争力的解释，避免长路径数量淹没束搜索。
        edges.sort(key=lambda edge: float(edge.get('individual_score', edge['score'])))
        bridges[key] = edges[:int(cfg.get('hypothesis_bridge_candidates', 3))]
    all_bands = dict(bands)
    all_bands.update({key: {'edges': {(e['left_part'], e['right_part']): e for e in edges}}
                      for key, edges in bridges.items()})
    controls = []; excluded = set(); history = []; width = int(cfg.get('hypothesis_beam_width', 6))
    max_rounds = min(n, int(cfg.get('hypothesis_max_rounds', 110)))
    for step in range(max_rounds):
        anchors = [(u, v) for u, v, _ in controls]
        options = {cut: _band_options(band, by_id, anchors, orientation, cfg)
                   for cut, band in bands.items()}
        options.update({key: _bridge_options(edges, by_id, anchors, cfg)
                        for key, edges in bridges.items()})
        beam = _search(all_bands, options, index, n, forbidden, support, width,
                       controls, excluded=excluded)
        if not beam:
            break
        fixed = {_pair(index[u], index[v]) for u, v, _ in controls}
        best = beam[0]
        proposed = [pair for pair in best[3] if pair not in fixed and pair not in excluded]
        if not proposed:
            break
        # 比较“包含这一对”和“允许其他位移/层序解释但排除这一对”的全局最优解。
        votes = defaultdict(float)
        scale = max(.12, best[0]-beam[-1][0])
        for state in beam:
            weight = float(np.exp((state[0]-best[0])/scale))
            for pair in state[3]-fixed:
                votes[pair] += weight
        total = sum(float(np.exp((state[0]-best[0])/scale)) for state in beam)
        ranked = sorted(proposed, key=lambda pair: votes[pair]/max(total, 1e-9), reverse=True)
        selected = None
        for pair in ranked[:1]:
            frequency = votes[pair]/max(total, 1e-9)
            # 排除此对后重新配准并重排层序，避免在被剪枝的旧方案里制造虚假共识。
            excluded_ids = _pair(descriptors[pair[0]]['id'], descriptors[pair[1]]['id'])
            rival_options = {cut: _band_options(band, by_id, anchors, orientation,
                                                cfg, excluded=excluded_ids)
                             for cut, band in bands.items()}
            rival_options.update({key: _bridge_options(edges, by_id, anchors, cfg,
                                                         excluded=excluded_ids)
                                  for key, edges in bridges.items()})
            alternative = _search(all_bands, rival_options, index, n, forbidden,
                                  support, width, controls, excluded=pair)
            margin = best[0]-(alternative[0][0] if alternative else 0.)
            if frequency >= float(cfg.get('hypothesis_min_consensus', .35)) and margin >= float(
                    cfg.get('hypothesis_min_margin', .01)):
                selected = (pair, frequency, margin)
                break
        if selected is None:
            # 暂不确认模糊边，让其余区域的控制层先形成；一旦有新控制就重新评估。
            excluded.update(ranked[:3])
            continue
        pair, frequency, margin = selected
        excluded.clear()
        # 只提交一对；如未来证据推翻它，可从 controls 删除并重算，未使用不可逆并查集。
        chosen_edge = None
        for band in all_bands.values():
            for (u, v), edge in band['edges'].items():
                if _pair(index[u], index[v]) == pair:
                    chosen_edge = (u, v, edge); break
            if chosen_edge is not None:
                break
        if chosen_edge is None:
            break
        controls.append(chosen_edge)
        if len(controls) % 10 == 0:
            print(f'global matching controls: {len(controls)}, trial: {step+1}', flush=True)
        history.append({'round': len(history)+1, 'left_part': chosen_edge[0],
                        'right_part': chosen_edge[1], 'consensus': round(frequency, 4),
                        'global_alternative_margin': round(margin, 4),
                        'candidate_hypotheses': len(beam)})
    # 复查早期较弱的暂定控制层：在保留其他控制的条件下，重新拟合所有局部位移。
    # 如果全局最佳解已不需要该对，就撤销它；最终合组只用复查后的控制集合。
    revisions = []
    review_limit = int(cfg.get('hypothesis_recheck_limit', 12))
    weak = sorted(history,
                  key=lambda item: item['global_alternative_margin'])[:review_limit]
    for item in weak:
        target = next(((u, v, edge) for u, v, edge in controls
                       if u == item['left_part'] and v == item['right_part']), None)
        if target is None:
            continue
        others = [record for record in controls
                  if (record[0], record[1]) != (target[0], target[1])]
        remaining = [(u, v) for u, v, _ in others]
        options = {cut: _band_options(band, by_id, remaining, orientation, cfg)
                   for cut, band in bands.items()}
        options.update({key: _bridge_options(edges, by_id, remaining, cfg)
                        for key, edges in bridges.items()})
        beam = _search(all_bands, options, index, n, forbidden, support,
                       width, others)
        excluded_ids = _pair(target[0], target[1])
        rival_options = {cut: _band_options(band, by_id, remaining, orientation,
                                            cfg, excluded=excluded_ids)
                         for cut, band in bands.items()}
        rival_options.update({key: _bridge_options(edges, by_id, remaining, cfg,
                                                     excluded=excluded_ids)
                              for key, edges in bridges.items()})
        alternative = _search(all_bands, rival_options, index, n, forbidden,
                              support, width, others,
                              excluded=_pair(index[target[0]], index[target[1]]))
        pair = _pair(index[target[0]], index[target[1]])
        margin = (beam[0][0]-(alternative[0][0] if alternative else 0.)) if beam else -1.
        if not beam or pair not in beam[0][3] or margin < float(
                cfg.get('hypothesis_rollback_margin', .015)):
            controls = others
            revisions.append({'left_part': target[0], 'right_part': target[1],
                              'reason': 'later_global_alternative',
                              'rechecked_margin': round(margin, 4)})
    # 所有路径已共同竞争且逐对复查，最终才把控制关系合并为地层组。
    groups = tuple(range(n)); used = set(); chosen = set()
    for u, v, edge in controls:
        pair = _pair(index[u], index[v]); ports = _ports(edge, index, u, v)
        merged, _ = _merge(groups, index[u], index[v], forbidden, support, chosen)
        if merged is None or any(p in used for p in ports):
            continue
        groups = merged; used.update(ports); chosen.add(pair)
    # 高置信全局控制形成后，逐对延伸。每轮重算层序和相对厚度，所有路径争夺同一端点。
    for _ in range(n):
        anchors = [(u, v) for u, v, _ in controls]
        pool = {}
        for cut, band in bands.items():
            for option in _band_options(band, by_id, anchors, orientation, cfg):
                for u, v, cost in option['pairs']:
                    key = _pair(index[u], index[v])
                    if key not in pool or cost < pool[key][0]:
                        pool[key] = (cost, u, v, band['edges'][u, v])
        for key, edges in bridges.items():
            for option in _bridge_options(edges, by_id, anchors, cfg):
                u, v, cost = option['pairs'][0]
                pair = _pair(index[u], index[v])
                if pair not in pool or cost < pool[pair][0]:
                    pool[pair] = (cost, u, v, all_bands[key]['edges'][u, v])
        ranked_pool = []
        for cost, u, v, edge in pool.values():
            if cost > float(cfg.get('hypothesis_extension_max_cost', .62)):
                continue
            pair = _pair(index[u], index[v]); ports = _ports(edge, index, u, v)
            if pair in chosen or any(p in used for p in ports):
                continue
            merged, _ = _merge(groups, index[u], index[v], forbidden, support, chosen)
            if merged is not None and merged != groups:
                ranked_pool.append((cost, u, v, edge, merged, ports))
        if not ranked_pool:
            break
        ranked_pool.sort(key=lambda item: item[0])
        # 相同端点存在近似等分候选时暂缓，待其他控制层改善层序证据。
        picked = None
        for item in ranked_pool:
            cost, u, v, edge, merged, ports = item
            rival = min((other[0] for other in ranked_pool
                         if other is not item and any(p in other[5] for p in ports)),
                        default=float('inf'))
            if rival-cost >= float(cfg.get('hypothesis_extension_margin', .018)) or cost < .32:
                picked = item
                break
        if picked is None:
            break
        cost, u, v, edge, groups, ports = picked
        controls.append((u, v, edge)); used.update(ports); chosen.add(_pair(index[u], index[v]))
        history.append({'round': len(history)+1, 'left_part': u, 'right_part': v,
                        'extension': True, 'calibrated_cost': round(cost, 4),
                        'port_alternative_margin': (round(rival-cost, 4)
                                                    if np.isfinite(rival) else None)})
    for edges in bridges.values():
        for edge in edges:
            matched = edge.get('sequence_alignment', {}).get('matched_layers', [])
            edge['multi_cut_sequence_support'] = sum(
                item.get('axis') == 'normal' and
                item.get('left') in index and item.get('right') in index and
                _pair(index[item['left']], index[item['right']]) in chosen
                for item in matched)
    for edge in candidates:
        if edge['left_index'] < n and edge['right_index'] < n:
            edge['decision'] = ('grouped' if _pair(edge['left_index'], edge['right_index'])
                                in chosen else 'not_selected_globally')
    return groups, frozenset(chosen), {
        'method': 'joint_single_multi_cut_progressive_hypotheses',
        'bands_considered': len(bands), 'selected_pairs': len(chosen),
        'selected_bands': len({cut for cut, band in bands.items() if any(
            _pair(index[u], index[v]) in chosen for u, v in band['edges'])}),
        'rounds': history, 'revisions': revisions, 'beam_width': width,
        'gradient_samples': orientation.samples if orientation else 0,
        'reliable_gradient_samples': orientation.reliable if orientation else 0,
        'weighting': {'cut_affected_min_layers': 1, 'sequence': .40,
                      'relative_thickness': .30, 'corrected_geometry': .25,
                      'morphology': .05},
        'band_sequence_source': 'single-cut contact paths only',
        'multi_cut_paths': 'joint beam alternatives and rollback',
    }
