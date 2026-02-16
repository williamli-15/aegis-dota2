import argparse, os, orjson
from typing import Any, Dict, List, Optional
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

def slot_to_idx(player_slot: int) -> int:
    return player_slot if player_slot < 128 else player_slot - 123  # 128->5 ... 132->9

def slot_to_team(player_slot: int) -> int:
    return 0 if player_slot < 128 else 1

def write_part(out_dir: str, name: str, part_idx: int, rows: List[Dict[str, Any]]):
    if not rows:
        return
    os.makedirs(os.path.join(out_dir, name), exist_ok=True)
    table = pa.Table.from_pylist(rows)
    path = os.path.join(out_dir, name, f"part-{part_idx:05d}.parquet")
    pq.write_table(table, path)
    rows.clear()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--flush_matches", type=int, default=200)  # 每处理多少场 flush 一次
    ap.add_argument("--min_duration", type=int, default=600)   # 过滤短局：10min
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    part = 0
    buf_matches, buf_players, buf_sp, buf_st, buf_events, buf_tf = [], [], [], [], [], []

    with open(args.input, "rb") as f:
        for line in tqdm(f, desc="ETL"):
            if not line.strip():
                continue
            m = orjson.loads(line)

            # basic filters (可按你们需要再加)
            if int(m.get("human_players", 10)) != 10:
                continue
            if int(m.get("duration", 0)) < args.min_duration:
                continue

            match_id = int(m["match_id"])
            patch = int(m.get("patch", -1))
            start_time = int(m.get("start_time", 0))
            duration = int(m.get("duration", 0))

            buf_matches.append({
                "match_id": match_id,
                "patch": patch,
                "start_time": start_time,
                "duration": duration,
                "radiant_win": bool(m.get("radiant_win", False)),
                "game_mode": int(m.get("game_mode", -1)),
                "lobby_type": int(m.get("lobby_type", -1)),
                "region": int(m.get("region", -1)),
            })

            # ---- players_static + per-minute player state + per-minute team state
            players = m.get("players") or []
            # team minute series (use index of times)
            rg = m.get("radiant_gold_adv") or []
            rx = m.get("radiant_xp_adv") or []

            # pick a canonical times list
            canonical_times = None
            if players and isinstance(players[0].get("times"), list):
                canonical_times = players[0]["times"]

            # player static
            for p in players:
                ps = int(p.get("player_slot", -1))
                slot_idx = slot_to_idx(ps)
                team = slot_to_team(ps)
                buf_players.append({
                    "match_id": match_id,
                    "slot_idx": slot_idx,
                    "team": team,
                    "hero_id": int(p.get("hero_id", -1)),
                    "lane": int(p.get("lane", -1)),
                    "lane_role": int(p.get("lane_role", -1)),
                    "rank_tier": p.get("rank_tier"),
                    "computed_mmr": p.get("computed_mmr"),
                })

            # per-minute snapshots
            if canonical_times:
                for ti, t in enumerate(canonical_times):
                    t = int(t)
                    # team minute row
                    buf_st.append({
                        "match_id": match_id,
                        "patch": patch,
                        "t": t,
                        "radiant_gold_adv": int(rg[ti]) if ti < len(rg) else None,
                        "radiant_xp_adv": int(rx[ti]) if ti < len(rx) else None,
                    })
                    # player minute rows
                    for p in players:
                        ps = int(p.get("player_slot", -1))
                        slot_idx = slot_to_idx(ps)
                        team = slot_to_team(ps)
                        hero_id = int(p.get("hero_id", -1))
                        gold_t = p.get("gold_t") or []
                        xp_t = p.get("xp_t") or []
                        lh_t = p.get("lh_t") or []
                        dn_t = p.get("dn_t") or []
                        if ti >= len(gold_t) or ti >= len(xp_t) or ti >= len(lh_t) or ti >= len(dn_t):
                            continue
                        buf_sp.append({
                            "match_id": match_id,
                            "patch": patch,
                            "t": t,
                            "slot_idx": slot_idx,
                            "team": team,
                            "hero_id": hero_id,
                            "gold": int(gold_t[ti]),
                            "xp": int(xp_t[ti]),
                            "lh": int(lh_t[ti]),
                            "dn": int(dn_t[ti]),
                        })

            def _s(v):
                if v is None:
                    return None
                return v if isinstance(v, str) else str(v)

            def _f(v):
                if v is None:
                    return None
                # 有时 x/y 可能是 int，也可能是 str；统一转 float
                return float(v)

            # ---- events: purchases, wards, kills, runes, objectives, teamfights
            def add_event(
                t: int,
                slot_idx: Optional[int],
                team: Optional[int],
                event_type: str,
                key: Optional[str] = None,
                x: Optional[float] = None,
                y: Optional[float] = None,
                target: Optional[str] = None,
                fight_id: Optional[int] = None,
            ):
                buf_events.append({
                    "match_id": match_id,
                    "patch": patch,
                    "t": int(t),
                    "slot_idx": slot_idx,
                    "team": team,
                    "event_type": event_type,
                    "key": _s(key),
                    "x": _f(x),
                    "y": _f(y),
                    "target": _s(target),
                    "fight_id": fight_id,
                })

            # player-driven logs
            for p in players:
                ps = int(p.get("player_slot", -1))
                slot_idx = slot_to_idx(ps)
                team = slot_to_team(ps)

                for it in (p.get("purchase_log") or []):
                    add_event(it.get("time", 0), slot_idx, team, "purchase", key=it.get("key"))

                for w in (p.get("obs_log") or []):
                    add_event(w.get("time", 0), slot_idx, team, "ward_obs_place",
                              key=w.get("key"), x=w.get("x"), y=w.get("y"))

                for w in (p.get("sen_log") or []):
                    add_event(w.get("time", 0), slot_idx, team, "ward_sen_place",
                              key=w.get("key"), x=w.get("x"), y=w.get("y"))

                for w in (p.get("obs_left_log") or []):
                    add_event(w.get("time", 0), slot_idx, team, "ward_obs_left",
                              key=w.get("key"), x=w.get("x"), y=w.get("y"),
                              target=w.get("attackername"))

                for w in (p.get("sen_left_log") or []):
                    add_event(w.get("time", 0), slot_idx, team, "ward_sen_left",
                              key=w.get("key"), x=w.get("x"), y=w.get("y"),
                              target=w.get("attackername"))

                for k in (p.get("kills_log") or []):
                    add_event(k.get("time", 0), slot_idx, team, "kill", key=k.get("key"))

                for r in (p.get("runes_log") or []):
                    add_event(r.get("time", 0), slot_idx, team, "rune", key=str(r.get("key")))

            # match-level objectives
            for o in (m.get("objectives") or []):
                t = o.get("time", 0)
                # 有些 objective 有 player_slot
                ps = o.get("player_slot")
                if ps is not None:
                    ps = int(ps)
                    slot_idx = slot_to_idx(ps)
                    team = slot_to_team(ps)
                else:
                    slot_idx, team = None, None
                # objective 的 type 很重要
                add_event(t, slot_idx, team, "objective",
                          key=o.get("type"),
                          target=o.get("key") or o.get("unit"))

            # teamfights (start/end + table)
            tf_list = m.get("teamfights") or []
            for fid, tf in enumerate(tf_list):
                st_ = int(tf.get("start", 0))
                en_ = int(tf.get("end", 0))
                deaths = int(tf.get("deaths", 0))
                buf_tf.append({
                    "match_id": match_id,
                    "patch": patch,
                    "fight_id": fid,
                    "start": st_,
                    "end": en_,
                    "deaths": deaths,
                })
                add_event(st_, None, None, "teamfight_start", fight_id=fid)
                add_event(en_, None, None, "teamfight_end", fight_id=fid)

            # flush by match count
            if len(buf_matches) >= args.flush_matches:
                write_part(args.out_dir, "matches", part, buf_matches)
                write_part(args.out_dir, "players_static", part, buf_players)
                write_part(args.out_dir, "state_player_minute", part, buf_sp)
                write_part(args.out_dir, "state_team_minute", part, buf_st)
                write_part(args.out_dir, "events", part, buf_events)
                write_part(args.out_dir, "teamfights", part, buf_tf)
                part += 1

    # final flush
    write_part(args.out_dir, "matches", part, buf_matches)
    write_part(args.out_dir, "players_static", part, buf_players)
    write_part(args.out_dir, "state_player_minute", part, buf_sp)
    write_part(args.out_dir, "state_team_minute", part, buf_st)
    write_part(args.out_dir, "events", part, buf_events)
    write_part(args.out_dir, "teamfights", part, buf_tf)

if __name__ == "__main__":
    main()
